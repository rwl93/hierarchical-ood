"""Hierarchy metrics"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

import calculate_log as callog
from protos import main_pb2

from collections import defaultdict


device = 'cuda' if torch.cuda.is_available() else 'cpu'
module_logger = logging.getLogger('__main__.hierarchy_metrics')


def calc_ensemble_accuracy(logits, targets, model='softmax', hierarchy=None, T=1000.):
    """Calculate ensemble accuracy for a tensor of logits and targets

    Parameters
    ----------
    logits : torch.tensor
        tensor containing raw logits from each model. Expecting input with
        size: (Num Models, Num Samples, Num output units)
    targets : torch.tensor
        Tensor containing targets. Shape: (Num Samples)
    """
    # Calculate leaf probabilities
    if model == 'softmax':
        probs = F.softmax(logits, -1)
        probs = probs.mean(0)
        preds = probs.argmax(1)
    elif model == 'cascade':
        probs = torch.empty((0,), device='cpu')
        for l in logits:
            hlogits = hierarchy.split_logits_by_synset(l)
            synset_bscores, _, _, _ = gather_soft_scores(hlogits, hierarchy, model, T)
            leaf_bprobs = hierarchy.to_leafs(synset_bscores)
            probs = torch.cat((probs, leaf_bprobs.expand(1,-1,-1)), 0)
        probs = probs.mean(0)
        preds = probs.argmax(1)
    elif model == 'mos':
        raise NotImplementedError("Ensemble MOS not implemented")
    else:
        raise ValueError("Unsupported model type")
    return preds.eq(targets).sum() / preds.size(0)


def calc_ensemble_ood(val_logits, ood_logits, model='softmax', hierarchy=None, T=1000.):
    """Calculate ensemble accuracy for a tensor of logits and targets

    Parameters
    ----------
    logits : torch.tensor
        tensor containing raw logits from each model. Expecting input with
        size: (Num Models, Num Samples, Num output units)
    targets : torch.tensor
        Tensor containing targets. Shape: (Num Samples)
    """
    # Calculate leaf probabilities
    if model == 'softmax':
        metric_results = {k: {} for k in ['MSP','ODIN']}
        val_scores = []
        val_scores.append(F.softmax(val_logits, dim=-1))
        val_scores.append(F.softmax(val_logits/T, dim=-1))
        val_scores = [s.mean(0).max(1)[0].numpy() for s in val_scores]

        ood_scores = []
        ood_scores.append(F.softmax(ood_logits, -1))
        ood_scores.append(F.softmax(ood_logits/T, -1))
        ood_scores = [s.mean(0).max(1)[0].numpy() for s in ood_scores]

        module_logger.info("Computing OOD Statistics...")
        metric_results["MSP"] = callog.metric(val_scores[0], ood_scores[0])
        metric_results["ODIN"] = callog.metric(val_scores[1], ood_scores[1])
    elif model == 'cascade':
        # Basic prediction method scoring
        val_scores = gen_ensemble_hierarchy_ood_scores(
            val_logits, hierarchy, model=model, T=T)
        ood_scores = gen_ensemble_hierarchy_ood_scores(
            ood_logits, hierarchy, model=model, T=T)
        val_pp, val_meanent, val_minent, val_maxent = val_scores[:4]
        val_meanent_mean, val_minent_mean, val_maxent_mean = val_scores[4:7]
        val_meanent_min, val_minent_min, val_maxent_min = val_scores[7:]
        ood_pp, ood_meanent, ood_minent, ood_maxent = ood_scores[:4]
        ood_meanent_mean, ood_minent_mean, ood_maxent_mean = ood_scores[4:7]
        ood_meanent_min, ood_minent_min, ood_maxent_min = ood_scores[7:]

        metric_results = {}
        module_logger.info("Computing OOD Statistics...")
        metric_results["PP"] = callog.metric(val_pp[0], ood_pp[0])
        metric_results["PP Temp Scaled"] = callog.metric(val_pp[1], ood_pp[1])
        metric_results["HMEAN"] = callog.metric(-val_meanent[0], -ood_meanent[0])
        metric_results["HMEAN Temp Scaled"] = callog.metric(-val_meanent[1], -ood_meanent[1])
        metric_results["HMIN"] = callog.metric(-val_minent[0], -ood_minent[0])
        metric_results["HMIN Temp Scaled"] = callog.metric(-val_minent[1], -ood_minent[1])
        metric_results["HMAX"] = callog.metric(-val_maxent[0], -ood_maxent[0])
        metric_results["HMAX Temp Scaled"] = callog.metric(-val_maxent[1], -ood_maxent[1])

        metric_results["HMEAN MEAN"] = callog.metric(-val_meanent_mean[0], -ood_meanent_mean[0])
        metric_results["HMEAN MEAN TS"] = callog.metric(-val_meanent_mean[1], -ood_meanent_mean[1])
        metric_results["HMIN MEAN"] = callog.metric(-val_minent_mean[0], -ood_minent_mean[0])
        metric_results["HMIN MEAN TS"] = callog.metric(-val_minent_mean[1], -ood_minent_mean[1])
        metric_results["HMAX MEAN"] = callog.metric(-val_maxent_mean[0], -ood_maxent_mean[0])
        metric_results["HMAX MEAN TS"] = callog.metric(-val_maxent_mean[1], -ood_maxent_mean[1])

        metric_results["HMEAN MIN"] = callog.metric(-val_meanent_min[0], -ood_meanent_min[0])
        metric_results["HMEAN MIN TS"] = callog.metric(-val_meanent_min[1], -ood_meanent_min[1])
        metric_results["HMIN MIN"] = callog.metric(-val_minent_min[0], -ood_minent_min[0])
        metric_results["HMIN MIN TS"] = callog.metric(-val_minent_min[1], -ood_minent_min[1])
        metric_results["HMAX MIN"] = callog.metric(-val_maxent_min[0], -ood_maxent_min[0])
        metric_results["HMAX MIN TS"] = callog.metric(-val_maxent_min[1], -ood_maxent_min[1])
    elif model == 'mos':
        raise NotImplementedError("Ensemble MOS not implemented")
    else:
        raise ValueError("Unsupported model type")
    return metric_results


def gen_ensemble_hierarchy_ood_scores(logits, H, model='cascade', T=1000., IPP=False, eps=0.):
    """
    Generate OOD scores for HSC models from probs

    Parameters
    ----------
    """
    num_models = logits.size(0)

    syn_msp = torch.empty((0,), device='cpu')
    syn_odin = torch.empty((0,), device='cpu')
    syn_bents = torch.empty((0,), device='cpu')
    syn_oents = torch.empty((0,), device='cpu')
    for l in logits:
        hlogits = H.split_logits_by_synset(l)
        syn_bscores, syn_oscores, syn_bent, syn_oent = gather_soft_scores(hlogits, H, model, T)
        syn_bents = torch.cat((syn_bents, syn_bent.cpu().expand(1,-1,-1)),0)
        syn_oents = torch.cat((syn_oents, syn_oent.cpu().expand(1,-1,-1)),0)
        syn_btensor = torch.empty((0,), device='cpu')
        syn_otensor = torch.empty((0,), device='cpu')
        for bscore, oscore in zip(syn_bscores, syn_oscores):
            syn_btensor = torch.cat((syn_btensor, bscore), 1)
            syn_otensor = torch.cat((syn_otensor, oscore), 1)
        syn_msp = torch.cat((syn_msp, torch.tensor(syn_btensor).expand(1, -1, -1)), 0)
        syn_odin = torch.cat((syn_odin, torch.tensor(syn_otensor).expand(1, -1, -1)), 0)

    # Average models
    syn_msp = syn_msp.mean(0)
    syn_odin = syn_odin.mean(0)

    # Calculate synset entropy
    syn_mspent = torch.ones((syn_msp.size(0), len(H.synset_offsets,)), device='cpu') * -1.
    syn_odinent = torch.ones((syn_msp.size(0), len(H.synset_offsets,)), device='cpu') * -1.
    for i, (off, bnd) in enumerate(zip(H.synset_offsets, H.synset_bounds)):
        off = int(off)
        bnd = int(bnd) + 1
        syn_mspent[:, i] = -(syn_msp[:, off:bnd] * torch.log(syn_msp[:, off:bnd])).sum(1)
        syn_odinent[:, i] = -(syn_odin[:, off:bnd] * torch.log(syn_odin[:, off:bnd])).sum(1)

    # Get path probabilities
    pp_msp = H.to_leafs(H.split_logits_by_synset(syn_msp))
    preds = pp_msp.argmax(1) # For entropy metric
    pp_msp = pp_msp.max(1)[0].numpy()
    pp_odin = H.to_leafs(H.split_logits_by_synset(syn_odin)).max(1)[0].numpy()

    # Get path entropy
    mean_bentropy = np.ones((preds.size(0),), dtype=float) * -1.
    mean_oentropy = np.ones((preds.size(0),), dtype=float) * -1.
    min_bentropy = np.ones((preds.size(0),), dtype=float) * -1.
    min_oentropy = np.ones((preds.size(0),), dtype=float) * -1.
    max_bentropy = np.ones((preds.size(0),), dtype=float) * -1.
    max_oentropy = np.ones((preds.size(0),), dtype=float) * -1.

    # Get path entropy using the entropy calculated by the individual models
    # not the ensemble w/ min
    mean_bentropy_allmean = np.ones((preds.size(0),), dtype=float) * -1.
    mean_oentropy_allmean = np.ones((preds.size(0),), dtype=float) * -1.
    min_bentropy_allmean =  np.ones((preds.size(0),), dtype=float) * -1.
    min_oentropy_allmean =  np.ones((preds.size(0),), dtype=float) * -1.
    max_bentropy_allmean =  np.ones((preds.size(0),), dtype=float) * -1.
    max_oentropy_allmean =  np.ones((preds.size(0),), dtype=float) * -1.

    # Get path entropy using the entropy calculated by the individual models
    # not the ensemble w/ min
    mean_bentropy_allmin = np.ones((preds.size(0),), dtype=float) * -1.
    mean_oentropy_allmin = np.ones((preds.size(0),), dtype=float) * -1.
    min_bentropy_allmin =  np.ones((preds.size(0),), dtype=float) * -1.
    min_oentropy_allmin =  np.ones((preds.size(0),), dtype=float) * -1.
    max_bentropy_allmin =  np.ones((preds.size(0),), dtype=float) * -1.
    max_oentropy_allmin =  np.ones((preds.size(0),), dtype=float) * -1.

    for i, p in enumerate(preds):
        bentropys = []
        oentropys = []

        plist = list(H.class_parents[H.train_classes[p]])
        plist.append(p)
        ind_bent = torch.empty((0,), device='cpu')
        ind_oent = torch.empty((0,), device='cpu')
        for curr in plist:
            syn_id, _ = H.classid2synsetid_offset(curr)
            bentropys.append(syn_mspent[i, syn_id])
            oentropys.append(syn_odinent[i, syn_id])

            ind_bent = torch.cat(
                (ind_bent, syn_bents[:, i, syn_id].view(-1, 1)), 1)
            ind_oent = torch.cat(
                (ind_oent, syn_oents[:, i, syn_id].view(-1, 1)), 1)

        # Calc mean and min ent for individual models
        mean_ind_bent = ind_bent.mean(-1)
        min_ind_bent, _ = ind_bent.min(-1)
        max_ind_bent, _ = ind_bent.max(-1)
        mean_ind_oent = ind_oent.mean(-1)
        min_ind_oent, _ = ind_oent.min(-1)
        max_ind_oent, _ = ind_oent.max(-1)

        mean_bentropy_allmean[i] = mean_ind_bent.mean()
        mean_oentropy_allmean[i] = mean_ind_oent.mean()
        min_bentropy_allmean[i]  = min_ind_bent.mean()
        min_oentropy_allmean[i]  = min_ind_oent.mean()
        max_bentropy_allmean[i]  = max_ind_bent.mean()
        max_oentropy_allmean[i]  = max_ind_oent.mean()

        mean_bentropy_allmin[i] = mean_ind_bent.min()
        mean_oentropy_allmin[i] = mean_ind_oent.min()
        min_bentropy_allmin[i]  = min_ind_bent.min()
        min_oentropy_allmin[i]  = min_ind_oent.min()
        max_bentropy_allmin[i]  = max_ind_bent.min()
        max_oentropy_allmin[i]  = max_ind_oent.min()

        bentropys = np.array(bentropys)
        oentropys = np.array(oentropys)
        mean_bentropy[i] = bentropys.mean() # Best mean certainty
        mean_oentropy[i] = oentropys.mean()
        min_bentropy[i] = bentropys.min() # Most certain node
        min_oentropy[i] = oentropys.min()
        max_bentropy[i] = bentropys.max() # Least certain node
        max_oentropy[i] = oentropys.max()

    return [pp_msp, pp_odin], \
           [mean_bentropy, mean_oentropy], \
           [min_bentropy, min_oentropy], \
           [max_bentropy, max_oentropy], \
           [mean_bentropy_allmean, mean_oentropy_allmean], \
           [min_bentropy_allmean, min_oentropy_allmean], \
           [max_bentropy_allmean, max_oentropy_allmean], \
           [mean_bentropy_allmin, mean_oentropy_allmin], \
           [min_bentropy_allmin, min_oentropy_allmin], \
           [max_bentropy_allmin, max_oentropy_allmin]


# adapted from pytorch ImageNet example code
class HierarchicalAccuracy:
    """Hierarchical accuracy metric

    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, hierarchy,
                 is_multilabel=False, is_split_logits=False, soft_preds=False,
                 track_hdist=False,
                 ):
        super().__init__()
        self.logger = logging.getLogger('__main__.hierarchy_metrics.HierarchicalAccuracy')
        self._hierarchy = hierarchy
        self._counts = torch.zeros(len(hierarchy.synset_bounds))
        self._running_scores = torch.zeros(len(hierarchy.synset_bounds))
        self._top1_scores = 0.
        self._top1_counts = 0.
        self._pred_scores = 0.
        self._pred_counts = 0.
        self._error_depth = torch.zeros(hierarchy.max_depth+1).to(device)
        self._is_multilabel = is_multilabel
        self._is_split_logits = is_split_logits
        self._soft_preds = soft_preds
        self._track_hdist = track_hdist
        if self._track_hdist:
            self._dist_matrix = torch.zeros((self._hierarchy.max_depth,
                                             self._hierarchy.max_depth),
                                            dtype=int)

    def update_state(self, outputs, targets):
        with torch.no_grad():
            if self._is_multilabel:
                ml, active_synsets = targets
            else:
                ml, active_synsets = self._hierarchy.to_multilabel(targets)
            if self._is_split_logits:
                hlogits = outputs
            else:
                hlogits = self._hierarchy.split_logits_by_synset(outputs)

            if self._soft_preds:
                _, _, tree_preds, _, _, _ = soft_predict(
                    hlogits,
                    self._hierarchy,
                    as_tensor=True)
            else:
                _, _, tree_preds, _, _, _ = hard_predict(
                    hlogits,
                    self._hierarchy,
                    as_tensor=True)
            tp_gts = ml + self._hierarchy.synset_offsets
            tp_gts *= active_synsets
            tp_gts = tp_gts.max(dim=1)[0]
            tp_correct = tree_preds.eq(tp_gts)
            self._pred_scores += torch.sum(tp_correct).float().to('cpu')
            self._pred_counts += tp_correct.size(0)
            # Track hierarchy distance
            if self._track_hdist:
                gt_dists, pred_dists = self.calc_hdists(ml, active_synsets,
                                                        tree_preds)
                for g, p in zip(gt_dists, pred_dists):
                    self._dist_matrix[g, p] += 1
            top1_correct = torch.ones(ml.size(0), dtype=torch.bool).to(device)
            error_depth = torch.ones(ml.size(0)).to(device) * -1.
            for i, hl in enumerate(hlogits):
                _, preds = torch.max(hl, axis=1)
                correct = preds.eq(ml[:, i])
                idxs2set = torch.where(active_synsets[:, i] == 1)
                correct = correct[idxs2set]
                top1_correct[idxs2set] = torch.logical_and(
                        top1_correct[idxs2set], correct)
                error_depth[idxs2set[0][torch.logical_not(correct)]] = \
                    self._hierarchy.get_synsetid_depth(i)
                self._running_scores[i] += torch.sum(correct).float().to('cpu')
                self._counts[i] += correct.shape[0]
            self._top1_scores += torch.sum(top1_correct).float().to('cpu')
            self._top1_counts += top1_correct.size(0)
            edeps, ecounts = torch.unique(error_depth, return_counts=True)
            self._error_depth[edeps.long()] += ecounts

    def calc_hdists(self, gt_ml, gt_act, preds):
        """Calculate the hierarchical distance between groundtruth and preds.

        Parameters
        ----------
        gt_ml :
            Groundtruth multilabels
        gt_act :
            Groundtruth activations
        preds :
            Hierarchicy path predictions 

        Returns
        -------
        (torch.IntTensor torch.IntTensor)
            The first tensor is the gt distance to the first shared node.
            The second tensor is the prediction distance to the shared node.
        """
        return calc_hdists(gt_ml, gt_act, preds, self._hierarchy)

    def reset_state(self,):
        self._running_scores = torch.zeros_like(self._running_scores)
        self._counts = torch.zeros_like(self._counts)
        self._top1_scores = 0.
        self._top1_counts = 0.
        self._pred_scores = 0.
        self._pred_counts = 0.
        self._error_depth = torch.zeros(self._hierarchy.max_depth+1).to(device)
        self._dist_matrix = torch.zeros((self._hierarchy.max_depth,
                                        self._hierarchy.max_depth),
                                       dtype=int)

    def result(self,):
        return np.nanmean((self._running_scores/self._counts).numpy())

    def result_full(self,):
        return np.divide(self._running_scores, self._counts,
                out=np.zeros_like(self._running_scores),
                where=self._counts!=0)

    def result_top1(self,):
        if self._top1_counts == 0:
            return -1
        return self._top1_scores/self._top1_counts

    def result_pred(self,):
        if self._pred_counts == 0:
            return -1
        return self._pred_scores / self._pred_counts

    def result_error_depth(self,):
        return self._error_depth.cpu().numpy()

    def result_hierarchy_distances(self,):
        return self._dist_matrix.cpu().numpy()


class HierarchicalPredAccuracy:
    """Hierarchical accuracy metric whose input is prediction class idxs

    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, hierarchy, track_hdist=False, is_gt_multilabel=False):
        super().__init__()
        self.logger = logging.getLogger(
            '__main__.hierarchy_metrics.HierarchicalPredAccuracy')
        self._hierarchy = hierarchy
        self._counts = 0.
        self._running_scores = 0.
        self._track_hdist = track_hdist
        if self._track_hdist:
            self._dist_matrix = torch.zeros((self._hierarchy.max_depth+1,
                                             self._hierarchy.max_depth+1),
                                            dtype=int)
        self._is_gt_multilabel = is_gt_multilabel

    def update_state(self, preds, targets):
        with torch.no_grad():
            if self._is_gt_multilabel:
                ml, act = targets
            else:
                ml, act = self._hierarchy.to_multilabel(targets,
                                                        is_leaf_idx=False)
            # No prediction -> pred = -1 which is the last index
            pred_ml, pred_act = self._hierarchy.to_multilabel(
                preds, is_leaf_idx=False)

            tp_gts = ml + self._hierarchy.synset_offsets
            tp_gts *= act
            tp_gts = tp_gts.max(dim=1)[0]
            # above problem avoided since we're using train class indexes
            tp_correct = preds.eq(tp_gts)
            self._running_scores += torch.sum(tp_correct).float().to('cpu')
            self._counts += tp_correct.size(0)
            # Track hierarchy distance
            if self._track_hdist:
                gt_dists, pred_dists = self.calc_hdists(ml, act, preds)
                for g, p in zip(gt_dists, pred_dists):
                    self._dist_matrix[g, p] += 1

    def calc_hdists(self, gt_ml, gt_act, preds):
        """Calculate the hierarchical distance between groundtruth and preds.

        Parameters
        ----------
        gt_ml :
            Groundtruth multilabels
        gt_act :
            Groundtruth activations
        preds :
            Hierarchicy path predictions 

        Returns
        -------
        (torch.IntTensor torch.IntTensor)
            The first tensor is the gt distance to the first shared node.
            The second tensor is the prediction distance to the shared node.
        """
        return calc_hdists(gt_ml, gt_act, preds, self._hierarchy)

    def reset_state(self,):
        self._running_scores = 0.
        self._counts = 0.
        self._dist_matrix = torch.zeros((self._hierarchy.max_depth+1,
                                         self._hierarchy.max_depth+1),
                                        dtype=int)

    def result(self,):
        return self._running_scores / self._counts

    def result_hierarchy_distances(self,):
        return self._dist_matrix.cpu().numpy()


class MOSAccuracy:
    """MOS accuracy metric

    Parameters
    ----------
    hierarchy : hierarchy_util.Hierarchy

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, hierarchy):
        super().__init__()
        self.logger = logging.getLogger('__main__.hierarchy_metrics.MOSAccuracy')
        self._hierarchy = hierarchy
        self._mos_off_bounds = hierarchy.get_MOS_offsets_bounds()
        self._group_scores = torch.zeros(
            hierarchy.get_MOS_offsets_bounds().size(0))
        self._scores = 0.
        self._counts = 0.

    def update_state(self, outputs, targets):
        with torch.no_grad():
            gt = self._hierarchy.to_MOSlabel(targets)
            gt_idx = self._hierarchy.to_MOSlabelidx(targets)
            gscores = torch.zeros(
                (outputs.size(0), self._mos_off_bounds.size(0)),
                device=device)
            gpreds = torch.zeros(
                (outputs.size(0), self._mos_off_bounds.size(0)),
                device=device).long()
            for gi, (off, bnd, _) in enumerate(self._mos_off_bounds):
                # MOS Predict
                gsoftmax = F.softmax(outputs[:, off:bnd+1], dim=1)
                gs, gp = gsoftmax[:, :-1].max(1)
                gp += off  # Add group offset to pred labels
                gscores[:, gi], gpreds[:, gi] = gs, gp

                # Group wise acc
                gcorrect = gsoftmax.argmax(1)
                gcorrect = gcorrect.eq(gt[:, gi])
                self._group_scores[gi] += torch.sum(gcorrect).float().to('cpu')

            preds = gpreds[torch.range(0, gpreds.size(0)-1).long(),
                           gscores.argmax(1)]
            correct = preds.eq(gt_idx)
            self._scores += torch.sum(correct).float().to('cpu')
            self._counts += outputs.size(0)

    def reset_state(self,):
        self._group_scores = torch.zeros_like(self._group_scores)
        self._scores = 0.
        self._counts = 0.

    def result(self,):
        return (self._scores/self._counts).numpy()

    def result_groupwise(self,):
        return (self._group_scores/self._counts).numpy()


class HierarchicalOOD:
    """Hierarchical OOD metric

    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, ood_hierarchy, id_hierarchy, model='cascade',
            ood_methods=['MSP', 'ODIN'], soft_preds=False):
        super().__init__()
        self.logger = logging.getLogger('__main__.hierarchy_metrics.HierarchicalOOD')
        self._ood_hierarchy = ood_hierarchy
        self._id_hierarchy = id_hierarchy
        self._ood_methods = ood_methods
        self._soft_preds = soft_preds
        self._hAcc = HierarchicalAccuracy(self._id_hierarchy,
                                          is_multilabel=True,
                                          is_split_logits=True,
                                          soft_preds=soft_preds)
        if type(model) == int:
            ref_config = main_pb2.Main()
            if model == ref_config.CASCADE:
                model = 'cascade'
            elif model == ref_config.CASCADEFCHEAD:
                model = 'cascadefchead'
            else:
                raise ValueError('Invalid model for HierarchicalOOD metrics')
        self._model = model
        self.reset_state()

    def reset_state(self):
        self._metric_results = {k: {} for k in self._ood_methods}
        self._odin_outlier_idxs = {}
        self._msp_outlier_idxs = {}
        self._hAcc.reset_state()
        self._hAcc_computed = False
        self._id_scores = None
        self._ood_dsets = []

    def update_state(self, net, id_loader, ood_loader, ood_dset="OOD"):
        self._ood_dsets.append(ood_dset)
        self._metric_results["MSP"][ood_dset] = defaultdict(list)
        self._metric_results["ODIN"][ood_dset] = defaultdict(list)
        if self._id_scores is None:
            msp_scores, odin_scores, _ = self.gen_msp_odin_scores(
                    net, id_loader, 1000., True, IPP=False, eps=0.01)
            self._id_scores = [msp_scores, odin_scores]
        ood_msp_scores, ood_odin_scores, _ = self.gen_msp_odin_scores(
            net, ood_loader, 1000., False, IPP=False, eps=0.01, dset=ood_dset)

        self.logger.info("Computing OOD Statistics...")
        # NOTE: The last 5 values in metric_results are:
        #    the OOD metric of the max id/ood scores across all active/inactive synset, respectively
        #    the OOD metric of the max path entropy scores
        #    the OOD metric of the prediction scores
        #    the OOD metric of the mean path entropy scores
        #    the OOD metric of the min path entropy scores
        self._metric_results["MSP"][ood_dset] = defaultdict(list)
        self._metric_results["ODIN"][ood_dset] = defaultdict(list)
        for i in range(len(self._id_hierarchy.synset_bounds)+5):
            mspid = self._id_scores[0][i]
            mspood = ood_msp_scores[i]
            if mspood.size > 0:
                msp_res = callog.metric(mspid, mspood)
                odin_res = callog.metric(
                    self._id_scores[1][i],
                    ood_odin_scores[i])
            else:
                msp_res = {'TMP': {}}
                odin_res = {'TMP': {}}
                for met in ['AUROC', 'TNR', 'AUOUT']:
                    msp_res['TMP'][met] = -1
                    odin_res['TMP'][met] = -1
            for met in ['AUROC', 'TNR', 'AUOUT']:
                self._metric_results["MSP"][ood_dset][met].append(
                    msp_res['TMP'][met])
                self._metric_results["ODIN"][ood_dset][met].append(
                    odin_res['TMP'][met])

    def gen_msp_odin_scores(self, net, loader, T, isID, IPP=False, eps=0.,
                            dset="ID"):
        """
        Generate OOD scores with the ODIN detector and the msp detector.

        Parameters
        ----------
        model : nn.Module
            base classifier network to test
        loader : torch.utils.data.DataLoader
            dataloader to compute ood scores over
        T : int
            temperature level to divide logits by
        isID : bool
            if the dataloader is for an in distribution set
        IPP : bool
            whether or not to do the input preprocessing step
        eps : float
            epsilon of perturbation if IPP is set
        """
        net.eval()
        baseline_scores = []
        odin_scores = []
        odin_ipp_scores = []
        for _ in range(len(self._id_hierarchy.synset_bounds)+5):
            baseline_scores.append(np.empty((0,)))
            odin_scores.append(np.empty((0,)))
            odin_ipp_scores.append(np.empty((0,)))

        if IPP:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                self._msp_outlier_idxs[dset] = np.empty((0,))
                self._odin_outlier_idxs[dset] = np.empty((0,))
                for dat, targs in loader:
                    dat, targs = dat.to(device), targs.to(device)
                    # if isID:
                    #     # Only include id synsets that are active
                    #     ml, act = self._id_hierarchy.to_multilabel(targs)
                    # else:
                    #     # Only include ood synsets that are not active
                    #     ml, act = self._ood_hierarchy.to_full_multilabel(targs)
                    #     # map active synsets to id hierarchy sets
                    #     ml, act  = self._id_hierarchy.trim_full_multilabel(ml, act)
                    logits = net(dat)
                    hlogits = self._id_hierarchy.split_logits_by_synset(logits)
                    # if (not isID) and (not self._hAcc_computed):
                    #     self._hAcc.update_state(hlogits, (ml,act))

                    # Basic prediction method scoring
                    if self._soft_preds:
                        pred_bscores, pred_oscores, _, bentropy_data, oentropy_data, entropy_idxs = soft_predict(
                            hlogits, self._id_hierarchy, self._model, T=1000.)
                    else:
                        pred_bscores, pred_oscores, _, bentropy_data, oentropy_data, entropy_idxs = hard_predict(
                            hlogits, self._id_hierarchy, self._model, T=1000.)

                    # initialize tracker for max score across all synsets
                    batch_size = dat.size(0)
                    max_bscores = np.ones((batch_size,))*-1.
                    max_oscores = np.ones((batch_size,))*-1.
                    # Max Non-Activated
                    baseline_scores[-5] = np.concatenate((
                        baseline_scores[-5], max_bscores), axis=0)
                    odin_scores[-5] = np.concatenate((
                        odin_scores[-5], max_oscores), axis=0)
                    # Max entropy
                    baseline_scores[-4] = np.concatenate((
                        baseline_scores[-4], -bentropy_data[2]), axis=0)
                    odin_scores[-4] = np.concatenate((
                        odin_scores[-4], -oentropy_data[2]), axis=0)
                    # Predictions
                    baseline_scores[-3] = np.concatenate((
                        baseline_scores[-3], pred_bscores), axis=0)
                    odin_scores[-3] = np.concatenate((
                        odin_scores[-3], pred_oscores), axis=0)
                    # Mean entropy
                    baseline_scores[-2] = np.concatenate((
                        baseline_scores[-2], -bentropy_data[0]), axis=0)
                    odin_scores[-2] = np.concatenate((
                        odin_scores[-2], -oentropy_data[0]), axis=0)
                    # Min entropy
                    baseline_scores[-1] = np.concatenate((
                        baseline_scores[-1], -bentropy_data[1]), axis=0)
                    odin_scores[-1] = np.concatenate((
                        odin_scores[-1], -oentropy_data[1]), axis=0)
                    # Entropy min idxs
                    self._msp_outlier_idxs[dset] = np.concatenate((
                        self._msp_outlier_idxs[dset], entropy_idxs[0]), axis=0)
                    self._odin_outlier_idxs[dset] = np.concatenate((
                        self._odin_outlier_idxs[dset], entropy_idxs[1]),
                        axis=0)

            # if (not isID) and (not self._hAcc_computed):
            #     self._hAcc_computed = True
        return baseline_scores, odin_scores, odin_ipp_scores

    def print_stats_of_list(self, prefix, dat):
        # Helper to print min/max/avg/std/len of values in a list
        dat = np.array(dat)
        self.logger.info("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(  # noqa: E501
                prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
        )

    def print_result(self):
        for dset in self._ood_dsets:
            max_metrics = []
            max_entropy_metrics = []
            pred_metrics = []
            mean_entropy_metrics = []
            min_entropy_metrics = []
            self.logger.info("OOD Dataset: " + dset)
            for ood in self._ood_methods:
                max_metrics.append(ood+' Metrics using max activations accumulated across all appropriate synsets:')
                max_entropy_metrics.append(ood+' Metrics using min entropy along path:')
                pred_metrics.append(ood+' Metrics using predicted activations:')
                mean_entropy_metrics.append(ood+' Metrics using mean entropy along path:')
                min_entropy_metrics.append(ood+' Metrics using min entropy along path:')
                for met in ['AUROC', 'TNR', 'AUOUT']:
                    self.print_stats_of_list(
                        ood + ' ' + met,
                        [s for s in self._metric_results[ood][met][:-1]
                         if s != -1])
                    max_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-5])
                    mean_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-4])
                    pred_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-3])
                    mean_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-2])
                    min_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][met][-1])
            for o in max_metrics:
                self.logger.info(o)
            for o in max_entropy_metrics:
                self.logger.info(o)
            for o in pred_metrics:
                self.logger.info(o)
            for o in mean_entropy_metrics:
                self.logger.info(o)
            for o in min_entropy_metrics:
                self.logger.info(o)
        self.logger.info("Averaged Accuracy: {}".format(self._hAcc.result()))
        self.logger.info("Top1 Accuracy: {}".format(self._hAcc.result_top1()))

    def print_result_full(self):
        for dset in self._ood_dsets:
            max_metrics = []
            max_entropy_metrics = []
            pred_metrics = []
            mean_entropy_metrics = []
            min_entropy_metrics = []
            self.logger.info("OOD Dataset: " + dset)
            for ood in self._ood_methods:
                max_metrics.append(ood+' Metrics using max activations accumulated across all appropriate synsets:')
                max_entropy_metrics.append(ood+' Metrics using max entropy along path:')
                pred_metrics.append(ood+' Metrics using predicted activations:')
                mean_entropy_metrics.append(ood+' Metrics using mean entropy along path:')
                min_entropy_metrics.append(ood+' Metrics using min entropy along path:')
                for met in ['AUROC', 'TNR', 'AUOUT']:
                    self.logger.info(ood + ' ' + met + ' Synset Performance')
                    self.logger.info(self._metric_results[ood][dset][met][:-1])
                    max_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-5])
                    max_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-4])
                    pred_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-3])
                    mean_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-2])
                    min_entropy_metrics[-1] += '\n\t' + met + ' ' + str(
                        self._metric_results[ood][dset][met][-1])
            for o in max_metrics:
                self.logger.info(o)
            for o in max_entropy_metrics:
                self.logger.info(o)
            for o in pred_metrics:
                self.logger.info(o)
            for o in mean_entropy_metrics:
                self.logger.info(o)
            for o in min_entropy_metrics:
                self.logger.info(o)
            min_bentropy_counts = np.unique(self._msp_outlier_idxs[dset],
                                            return_counts=True)
            min_oentropy_counts = np.unique(self._odin_outlier_idxs[dset],
                                            return_counts=True)
            self.logger.info("OOD Path entropy min idx (msp): {}, {}".format(
                min_bentropy_counts[0], min_bentropy_counts[1]))
            self.logger.info("OOD Path entropy min idx (odin): {}, {}".format(
                min_oentropy_counts[0], min_oentropy_counts[1]))
        # self.logger.info("Synset Accuracy: {}".format(self._hAcc.result_full()))
        # self.logger.info("Top1 Accuracy: {}".format(self._hAcc.result_top1()))

    def result(self):
        return dict(self._metric_results), \
               self._hAcc.result_full(), self._hAcc.result_top1()


class MOSOOD:
    """MOS OOD metric

    Parameters
    ----------
    hierarchy : hierarchy_util.Hierarchy

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, hierarchy):
        super().__init__()
        self.logger = logging.getLogger('__main__.hierarchy_metrics.MOSOOD')
        self._hierarchy = hierarchy
        self._mos_off_bounds = hierarchy.get_MOS_offsets_bounds()
        self._ood_methods = ['MOS']
        self.reset_state()

    def reset_state(self,):
        self._metric_results = {k: dict() for k in self._ood_methods}
        self._ood_dsets = []
        self._id_scores = None

    def update_state(self, net, id_loader, ood_loader, ood_dset="OOD"):
        self._ood_dsets.append(ood_dset)
        self._metric_results["MOS"][ood_dset] = defaultdict(list)
        # Get MOS Scores
        if self._id_scores is None:
            self._id_scores = self.gen_mos_scores(net, id_loader)
        ood_scores = self.gen_mos_scores(net, ood_loader)

        self.logger.info("Computing OOD Statistics...")

        mos_res = callog.metric(self._id_scores, ood_scores)
        for met in ['AUROC', 'TNR', 'AUOUT']:
            self._metric_results["MOS"][ood_dset][met].append(
                    mos_res['TMP'][met])

    def gen_mos_scores(self, net, loader):
        with torch.no_grad():
            scores = np.empty((0,), dtype=float)
            for pkg in loader:
                dat = pkg[0]
                dat = dat.to(device)
                outputs = net(dat)
                for gi, (off, bnd, _) in enumerate(self._mos_off_bounds):
                    outputs[:, off:bnd+1] = F.softmax(outputs[:, off:bnd+1],
                                                      dim=1)
                other_scores = outputs[:, self._mos_off_bounds[:, 1]]
                other_scores, _ = other_scores.min(1)
                other_scores = other_scores.cpu().numpy()
                scores = np.concatenate((scores, other_scores), axis=0)
        return -1. * scores

    def print_result(self):
        mos_metrics = 'MOS OOD Metrics:'
        for dset in self._ood_dsets:
            mos_metrics += '\n\tOOD Dataset: ' + dset
            for met in ['AUROC', 'TNR', 'AUOUT']:
                mos_metrics += '\n\t\t' + met + ' ' + str(
                    self._metric_results['MOS'][dset][met])
        self.logger.info(mos_metrics)

    def result(self):
        return dict(self._metric_results)


def hard_predict(hlogits, hierarchy, model='cascade', T=1000., as_tensor=False,
        soft_preds=False):
    """Generate prediction from hlogits by following decision tree"""
    # Initialize prediction confidences
    batch_size = hlogits[0].size(0)
    current_bscores = torch.ones((batch_size,)).to(device)
    current_oscores = torch.ones((batch_size,)).to(device)
    current_bentropy = [[] for _ in range(batch_size)]
    current_oentropy = [[] for _ in range(batch_size)]
    current_preds = torch.zeros((batch_size,)).long().to(device)
    current_act = torch.zeros((batch_size,)).long().to(device)
    for i, hl in enumerate(hlogits):
        curr_scores = _get_scores(hl, model, T)
        bscores, preds = torch.max(curr_scores[0], dim=1)
        oscores = torch.max(curr_oscores[1], dim=1)[0]
        idxs2set = current_act.eq(i)
        bscores = curr_scores[0][idxs2set]
        oscores = curr_scores[1][idxs2set]
        bentropy = curr_scores[2][idxs2set]
        oentropy = curr_scores[3][idxs2set]

        preds = preds[idxs2set]
        # Update running scores
        current_bscores[idxs2set] *= bscores
        current_oscores[idxs2set] *= oscores
        # get the indices instead of the boolean mask
        idxs2set = torch.where(idxs2set)[0]
        for j, idx in enumerate(idxs2set):
            current_bentropy[idx].append(bentropy[j])
            current_oentropy[idx].append(oentropy[j])
        # offset preds indexes
        preds += hierarchy.synset_offsets[i].long()
        current_preds[idxs2set] = preds
        current_act[idxs2set] = hierarchy.get_hyponym_synsetid(preds)
    mean_bentropy = torch.tensor([torch.tensor(ent).mean() for ent in current_bentropy]).to(device)
    min_bentropy = torch.tensor([torch.tensor(ent).min() for ent in current_bentropy]).to(device)
    mean_oentropy = torch.tensor([torch.tensor(ent).mean() for ent in current_oentropy]).to(device)
    min_oentropy = torch.tensor([torch.tensor(ent).min() for ent in current_oentropy]).to(device)
    min_bentropy_idx = torch.tensor([torch.tensor(ent).argmin() for ent in current_bentropy]).to(device)
    min_oentropy_idx = torch.tensor([torch.tensor(ent).argmin() for ent in current_oentropy]).to(device)
    if as_tensor:
        return current_bscores, current_oscores, current_preds, \
               [mean_bentropy, min_bentropy], [mean_oentropy, min_oentropy], \
               [min_bentropy_idx, min_oentropy_idx]
    return current_bscores.cpu().numpy(), current_oscores.cpu().numpy(), current_preds.cpu().numpy(), \
        [mean_bentropy.cpu().numpy(), min_bentropy.cpu().numpy()], \
        [mean_oentropy.cpu().numpy(), min_oentropy.cpu().numpy()], \
        [min_bentropy_idx.cpu().numpy(), min_oentropy_idx.cpu().numpy()]


def _get_scores(logits, model='cascade', T=1000.):
    """Calculate softmax/bce probability scores for a given model's logits

    Parameters
    ----------
    logits : torch.tensor
        The logit scores to calculate the probabilities for. Tensor is
        expected to be 2D with shape=(batch size, num_classes).
    model : string
        Model name to determine method for calculating probabilities
    T : float
        The temperature scaling parameter

    Return
    -------
    (torch.tensor, torch.tensor, torch.tensor, torch.tensor) :
        A tuple of tensors in the following order:
            1. standard scores
            2. temperature scaled scores
            3. standard entropy
            4. temperature scaled entropy
    """
    score_fn = F.softmax if 'cascade' in model else torch.sigmoid
    bscores = score_fn(logits.data.clone().detach(), dim=1)
    tsscores = score_fn(logits.data.clone().detach()/T, dim=1)
    bentropy = -(bscores * torch.log(bscores)).sum(dim=1)
    tsentropy = -(tsscores * torch.log(tsscores)).sum(dim=1)
    return bscores, tsscores, bentropy, tsentropy


def gather_soft_scores(hlogits, hierarchy, model='cascade', T=1000.):
    """Gather the soft prediction scores for given hierarchical logits.

    Calculates soft prediction scores using product rule for proabaility.
    Returns all of soft prediction scores using the scheme for hlogits (i.e. a
    list of torch.tensors of length number or synsets.

    Parameters
    ----------
    hlogits : list(torch.tensor)
        hierarchical logits to convert to soft probs
    hierarchy : hierarchy_util.Hierarchy
    model : string
        Model name to determine method for calculating probabilities
    T : float
        The temperature scaling parameter

    Return
    -------
    (torch.tensor, torch.tensor, torch.tensor, torch.tensor) :
        A tuple of tensors in the following order:
            1. All synset standard scores
            2. All synset temperature scaled scores
            3. All synset standard entropy
            4. All synset temperature scaled entropy
    """
    batch_size = hlogits[0].size(0)
    synset_bscores = []
    synset_oscores = []
    synset_bentropy = torch.zeros((batch_size, len(hierarchy.synset_bounds))).to(device)
    synset_oentropy = torch.zeros((batch_size, len(hierarchy.synset_bounds))).to(device)
    for i, hl in enumerate(hlogits):
        curr_scores = _get_scores(hl, model, T)
        synset_bentropy[:, i] = curr_scores[2]
        synset_oentropy[:, i] = curr_scores[3]
        curr_par = hierarchy.get_synsetid_parent(i)
        if curr_par is not None:
            s, o = hierarchy.classid2synsetid_offset(curr_par)
            par_bprob = synset_bscores[s][:, o]
            par_oprob = synset_oscores[s][:, o]
            synset_bscores.append(curr_scores[0] * par_bprob.view(-1, 1))
            synset_oscores.append(curr_scores[1] * par_oprob.view(-1, 1))
        else:
            # par_prob = torch.tensor([1., ])
            synset_bscores.append(curr_scores[0])
            synset_oscores.append(curr_scores[1])
    return synset_bscores, synset_oscores, synset_bentropy, synset_oentropy


def soft_predict(hlogits, hierarchy, model='cascade', T=1000., as_tensor=False,
                 return_pathentropy=False):
    """Generate prediction from hlogits by following decision tree"""
    batch_size = hlogits[0].size(0)
    # Get soft prob scores
    synset_scores = gather_soft_scores(hlogits, hierarchy, model, T)
    synset_bscores, synset_oscores = synset_scores[0], synset_scores[1]
    synset_bentropy, synset_oentropy = synset_scores[2], synset_scores[3]

    # Get leaf node probs
    leaf_bprobs = hierarchy.to_leafs(synset_bscores)
    leaf_oprobs = hierarchy.to_leafs(synset_oscores)

    # Get max value
    bscores, preds = torch.max(leaf_bprobs, dim=1)
    oscores = torch.max(leaf_oprobs, dim=1)[0]
    preds = [hierarchy.class_list.index(hierarchy.train_classes[p]) for p in preds]

    # Get path entropy
    mean_bentropy = torch.zeros((batch_size,)).to(device)
    mean_oentropy = torch.zeros((batch_size,)).to(device)
    min_bentropy = torch.zeros((batch_size,)).to(device)
    min_oentropy = torch.zeros((batch_size,)).to(device)
    max_bentropy = torch.zeros((batch_size,)).to(device)
    max_oentropy = torch.zeros((batch_size,)).to(device)
    min_bentropy_idx = torch.ones((batch_size,)).to(device)*-1
    min_oentropy_idx = torch.ones((batch_size,)).to(device)*-1
    if return_pathentropy:
        allbentropys = []
    for i, p in enumerate(preds):
        bentropys = []
        oentropys = []
        plist = list(hierarchy.class_parents[hierarchy.class_list[p]])
        plist.append(p)
        for curr in plist:
            syn_id, _ = hierarchy.classid2synsetid_offset(curr)
            bentropys.append(synset_bentropy[i, syn_id])
            oentropys.append(synset_oentropy[i, syn_id])
        bentropys = torch.tensor(bentropys).to(device)
        oentropys = torch.tensor(oentropys).to(device)
        mean_bentropy[i] = bentropys.mean()
        mean_oentropy[i] = oentropys.mean()
        min_bentropy[i] = bentropys.min()
        min_oentropy[i] = oentropys.min()
        max_bentropy[i] = bentropys.max()
        max_oentropy[i] = oentropys.max()
        min_bentropy_idx[i] = bentropys.argmin()
        min_oentropy_idx[i] = oentropys.argmin()
        if return_pathentropy:
            allbentropys.append(bentropys.cpu().numpy())

    preds = torch.tensor(preds).to(device)
    if as_tensor:
        return bscores, oscores, preds, \
               [mean_bentropy, min_bentropy], [mean_oentropy, min_oentropy], \
               [min_bentropy_idx, min_oentropy_idx]
    if return_pathentropy:
        return bscores.cpu().numpy(), oscores.cpu().numpy(), preds.cpu().numpy(), \
            allbentropys, \
            [mean_bentropy.cpu().numpy(), min_bentropy.cpu().numpy(), \
             max_bentropy.cpu().numpy()], \
            [mean_oentropy.cpu().numpy(), min_oentropy.cpu().numpy(), \
             max_oentropy.cpu().numpy()], \
            [min_bentropy_idx.cpu().numpy(), min_oentropy_idx.cpu().numpy()]
    return bscores.cpu().numpy(), oscores.cpu().numpy(), preds.cpu().numpy(), \
        [mean_bentropy.cpu().numpy(), min_bentropy.cpu().numpy(), max_bentropy.cpu().numpy()], \
        [mean_oentropy.cpu().numpy(), min_oentropy.cpu().numpy(), max_oentropy.cpu().numpy()], \
        [min_bentropy_idx.cpu().numpy(), min_oentropy_idx.cpu().numpy()]


def calc_hdists(gt_ml, gt_act, preds, hierarchy, is_pred_ml=False):
    """Calculate the hierarchical distance between groundtruth and preds.

    Parameters
    ----------
    gt_ml :
        Groundtruth multilabels
    gt_act :
        Groundtruth activations
    preds :
        Hierarchical path predictions
    hierarchy : hierarchy_util.Hierarchy
    is_pred_ml : bool
        Flag indicating if preds is a multilabel and activation

    Returns
    -------
    (torch.IntTensor torch.IntTensor)
        The first tensor is the gt distance to the first shared node.
        The second tensor is the prediction distance to the shared node.
    """
    # Convert preds to ml and activations
    if is_pred_ml:
        pred_ml, pred_act = preds
    else:
        pred_ml, pred_act = hierarchy.to_multilabel(preds, is_leaf_idx=False)
        # Remove incorrectly indexed non-predictions values
        pred_ml[preds == -1] = torch.zeros(((preds == -1).sum(),
                                            len(hierarchy.synset_bounds)),
                                           dtype=int, device=device)
        pred_act[preds == -1] = torch.zeros(((preds == -1).sum(),
                                             len(hierarchy.synset_bounds)),
                                            device=device)
    # Initialize trackers
    gt_dist = torch.ones(pred_ml.size(0), dtype=int, device=device) * -1
    pred_dist = torch.ones(pred_ml.size(0), dtype=int, device=device) * -1

    # Loop over synsets
    for i, _ in enumerate(hierarchy.synset_bounds):
        # Groundtruth and prediction active idxs
        gt_idxs2set = (gt_act[:, i] == 1)
        pred_idxs2set = (pred_act[:, i] == 1)

        # Handle no prediction case
        # if i == 0 and pred_act == 0
        if i == 0:
            no_pred = (pred_act[:, i] == 0.)
            if no_pred.sum() > 0:
                print(f'No Prediction Found: {no_pred.sum()}')
            gt_dist[no_pred] = 0
            pred_dist[no_pred] = 0
        # Track common node where both are active but they mismatch
        both_active = torch.logical_and(gt_idxs2set, pred_idxs2set)
        common_node_found = ~(gt_ml[:, i]).eq(pred_ml[:, i])
        common_node_found = torch.logical_and(both_active, common_node_found)

        # If the the current set is a set of leafs then set dist to 1, 1
        hyponym_synsetid = hierarchy.get_hyponym_synsetid(
            hierarchy.synset_offsets[i].view(-1).long())
        if hyponym_synsetid[0] == -1:
            gt_dist[common_node_found] = 1
            pred_dist[common_node_found] = 1
        else:
            gt_dist[common_node_found] = 0
            pred_dist[common_node_found] = 0

        # Track gt distance
        # If gt is active and pred is inactive
        # And the common ancestor has been found
        # Add 1 to the gt distance
        gt_idxs2set = torch.logical_and(gt_idxs2set, ~both_active)
        gt_idxs2set = torch.logical_and(gt_idxs2set, gt_dist >= 0)
        gt_dist[gt_idxs2set] += 1

        # Track pred distance relative
        pred_idxs2set = torch.logical_and(pred_idxs2set, ~both_active)
        pred_idxs2set = torch.logical_and(pred_idxs2set, pred_dist >= 0)
        pred_dist[pred_idxs2set] += 1

        # Track active mismatch
        # If the pred is active and the gt is inactive
        # And the common node has not been set
        # Then the prediction is deeper than it should be but its on the
        # correct path -> Overprediction
        # The converse is Underprection
        pred_act_gt_inact = torch.logical_and(gt_act[:, i] == 0,
                                              pred_act[:, i] == 1)
        pred_act_gt_inact = torch.logical_and(pred_act_gt_inact,
                                              pred_dist == -1)
        pred_act_gt_inact = torch.logical_and(pred_act_gt_inact, gt_dist == -1)
        gt_dist[pred_act_gt_inact] = 0
        pred_dist[pred_act_gt_inact] = 1

        gt_act_pred_inact = torch.logical_and(gt_act[:, i] == 1,
                                              pred_act[:, i] == 0)
        gt_act_pred_inact = torch.logical_and(gt_act_pred_inact,
                                              pred_dist == -1)
        gt_act_pred_inact = torch.logical_and(gt_act_pred_inact, gt_dist == -1)
        gt_dist[gt_act_pred_inact] = 1
        pred_dist[gt_act_pred_inact] = 0

    # Set Accurate predictions to distance = 0
    assert((gt_dist == -1).eq(pred_dist == -1).all())
    gt_dist[gt_dist == -1] = 0
    pred_dist[pred_dist == -1] = 0
    return gt_dist, pred_dist
