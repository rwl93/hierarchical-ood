"""Hierarchy Inference Classes and Stopping Criterion"""

# import matplotlib.pyplot as plt
import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import logging
# from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve  # , auc, confusion_matrix
import hierarchy_metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
module_logger = logging.getLogger('__main__.hierarchy_inference')


class StoppingCriterion:
    """Base class for inference stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95, pred_method='topdown'):
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.StoppingCriterion')
        self.H = hierarchy
        self.tnr = tnr
        self._thresh = []
        self._pred_method = pred_method

    def update(self,):
        raise NotImplementedError(
            "Update method must be overriden by subclasses")

    def reset(self,):
        for i in range(len(self._scores)):
            self._scores[i] = np.zeros_like(self._scores[i])
            self._gtroc[i] = np.zeros_like(self._gtroc[i])
        self._thresh = []

    def gen_threshold(self,):
        for scores, gtroc in zip(self._scores, self._gtroc):
            fpr, tpr, thresh = roc_curve(gtroc, scores)
            idx = np.argmin(np.abs((1-fpr) - self.tnr))
            targ = thresh[idx]
            self._thresh.append(targ)

    def log_threshold(self):
        self.logger.info('Threshold values:\n')
        self.logger.info(self._thresh)

    def print_threshold(self):
        print('Threshold values:\n')
        print(self._thresh)

    def predict(self, scores, leaf_preds):
        if self._pred_method.lower() == 'topdown':
            return self.top_down_predict(scores, leaf_preds)
        elif self._pred_method.lower() == 'bottomup':
            raise NotImplementedError("Bottom up pred is not implemented")
            # return self.bottom_up_predict(scores, leaf_preds)
        elif self._pred_method.lower() == 'flat':
            return self.flat_predict(scores, leaf_preds)
        else:
            raise ValueError("Unknown prediction method: " + self._pred_method)

    def top_down_predict(self, scores, leaf_preds):
        # Get multilabel and active_synsets
        pred_ml, pred_act = self.H.to_multilabel(leaf_preds)

        # -1 will indicate root
        thresh_preds = np.ones((pred_ml.size(0))) * -1
        thresh_pred_found = np.zeros((pred_ml.size(0)), dtype=bool)

        for i, end_idx in enumerate(self.H.synset_bounds):
            # Get synset parent
            curr_par = self.H.child2parent[self.H.class_list[end_idx]]
            # Get active preds
            pred_idxs = (pred_act[:, i] == 1).cpu()
            # Remove already found preds
            pred_idxs = np.logical_and(np.logical_not(thresh_pred_found),
                                       pred_idxs).bool()
            # Get predicted score for this synset for each example
            if len(scores[i].shape) > 1:
                curr_scores = scores[i].gather(
                    1, pred_ml[:, i].cpu().view(-1, 1)).squeeze()
            else:
                curr_scores = scores[i]
            # Check if threshold is not met
            if len(self._thresh) > 1:
                thresh = self._thresh[i]
            else:
                thresh = self._thresh[0]
            unconfidxs = (curr_scores < thresh).cpu()
            confidxs = (curr_scores >= thresh).cpu()
            # Combine threshold masks with pred_idxs mask
            unconfidxs = np.logical_and(pred_idxs, unconfidxs).bool()
            confidxs = np.logical_and(pred_idxs, confidxs).bool()
            # Set output for preds that did not meet the confidence threshold
            if curr_par is not None:
                thresh_preds[unconfidxs] = self.H.class_list.index(curr_par)
            thresh_pred_found[unconfidxs] = True
            # Update predictions that meet the threshold
            thresh_preds[confidxs] = (pred_ml[:, i][confidxs] +
                                      self.H.synset_offsets[i]
                                      ).cpu().numpy()
        return thresh_preds

    def flat_predict(self, scores, leaf_preds):
        # -1 will indicate root
        thresh_preds = np.ones((leaf_preds.size(0))) * -1
        # Check if threshold is not met
        thresh = self._thresh[0]
        confidxs = (scores[range(scores.size(0)), leaf_preds] >= thresh).cpu()
        # Update predictions that meet the threshold
        thresh_preds[confidxs] = leaf_preds[confidxs]
        return thresh_preds


class FlatStoppingCriterion(StoppingCriterion):
    """Synset entropy based stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr, pred_method='flat')
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.SynsetPathProbStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)]
        self._gtroc = [np.empty((0,), dtype=float)]

    def update(self, scores, labels, inp_scores=True):
        self._scores[0] = np.concatenate(
                (self._scores[0], scores.cpu().numpy().ravel()), axis=0)
        self._gtroc[0] = np.concatenate(
                (self._gtroc[0],
                 torch.nn.functional.one_hot(
                     labels, scores.size(1)).numpy().ravel()),
                axis=0)

    def log_threshold(self):
        self.logger.info("Threshold: {}".format(self._thresh[0]))

    def print_threshold(self):
        print("Threshold: {}".format(self._thresh[0]))

    def predict(self, scores):
        leaf_preds = scores.max(dim=1)[1]
        preds = super().predict(scores, leaf_preds)
        nonroot_preds = (preds >= 0)
        preds[nonroot_preds] = self.H.leaf_ids[preds[nonroot_preds]]
        return preds


class SynsetPathProbStoppingCriterion(StoppingCriterion):
    """Synset entropy based stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr)
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.SynsetPathProbStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)
                        for _ in hierarchy.synset_offsets]
        self._gtroc = [np.empty((0,), dtype=float)
                       for _ in hierarchy.synset_offsets]

    def update(self, inputs, labels, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, _, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            scores = inputs

        ml, act = self.H.to_multilabel(labels)

        # Add synset scores to tracker
        for i, end_idx in enumerate(self.H.synset_bounds):
            # Gather active synsets
            active_synset_idxs = (act[:, i] == 1)
            synset_gt = (ml[active_synset_idxs])[:, i]
            synset_scores = scores[i][active_synset_idxs]
            synset_gtroc = np.zeros(
                (synset_gt.size(0),
                 end_idx - int(self.H.synset_offsets[i]) + 1), dtype=int)
            for j in range(synset_gtroc.shape[1]):
                synset_gtroc[:, j] = (synset_gt[:] == j).cpu().numpy()

            self._scores[i] = np.concatenate(
                (self._scores[i], synset_scores.cpu().numpy().ravel()), axis=0)
            self._gtroc[i] = np.concatenate(
                (self._gtroc[i], synset_gtroc.ravel()), axis=0)

    def log_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            self.logger.info("{} Synset: Path Pred threshold {}".format(
                             synset_desc, self._thresh[i]))

    def print_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            print("{} Synset: Path Pred threshold {}".format(
                  synset_desc, self._thresh[i]))

    def predict(self, inputs, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, _, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            scores = inputs

        # Get soft preds and path through hierarchy
        leaf_scores = self.H.to_leafs(scores)
        # Max of leaf node scores
        leaf_preds = leaf_scores.max(dim=1)[1]
        return super().predict(scores, leaf_preds)


class SynsetSoftmaxStoppingCriterion(StoppingCriterion):
    """Synset entropy based stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr)
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.SynsetSoftmaxStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)
                        for _ in hierarchy.synset_offsets]
        self._gtroc = [np.empty((0,), dtype=float)
                       for _ in hierarchy.synset_offsets]

    def update(self, inputs, labels, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores = []
            for logit in hlogits:
                scores.append(torch.softmax(logit, dim=1))
        else:
            scores = inputs

        ml, act = self.H.to_multilabel(labels)

        # Add synset scores to tracker
        for i, end_idx in enumerate(self.H.synset_bounds):
            # Gather active synsets
            active_synset_idxs = (act[:, i] == 1)
            synset_gt = (ml[active_synset_idxs])[:, i]
            synset_scores = scores[i][active_synset_idxs]
            synset_gtroc = np.zeros(
                (synset_gt.size(0),
                 end_idx - int(self.H.synset_offsets[i]) + 1), dtype=int)
            for j in range(synset_gtroc.shape[1]):
                synset_gtroc[:, j] = (synset_gt[:] == j).cpu().numpy()

            self._scores[i] = np.concatenate(
                (self._scores[i], synset_scores.cpu().numpy().ravel()), axis=0)
            self._gtroc[i] = np.concatenate(
                (self._gtroc[i], synset_gtroc.ravel()), axis=0)

    def log_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            self.logger.info("{} Synset: Softmax threshold {}".format(
                             synset_desc, self._thresh[i]))

    def print_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            print("{} Synset: Softmax Pred threshold {}".format(
                  synset_desc, self._thresh[i]))

    def predict(self, inputs):
        # Calculate soft scores from logits
        hlogits = self.H.split_logits_by_synset(inputs)
        softscores, _, _, _ = hierarchy_metrics.gather_soft_scores(
            hlogits, self.H)
        # Get soft preds and path through hierarchy
        leaf_scores = self.H.to_leafs(softscores)
        # Max of leaf node scores
        leaf_preds = leaf_scores.max(dim=1)[1]

        # Calculate softmax scores
        scores = []
        for logit in hlogits:
            scores.append(torch.softmax(logit, dim=1))
        return super().predict(scores, leaf_preds)


class SynsetEntropyStoppingCriterion(StoppingCriterion):
    """Synset entropy based stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr)
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.SynsetSoftmaxStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)
                        for _ in hierarchy.synset_offsets]
        self._gtroc = [np.empty((0,), dtype=float)
                       for _ in hierarchy.synset_offsets]

    def update(self, inputs, labels, inp_scores=False):
        # Calculate soft scores from logits
        inputs = inputs.to(device)
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores = []
            softmaxpreds = []
            for logit in hlogits:
                softmaxscores = torch.softmax(logit, dim=1)
                softmaxpreds.append(torch.argmax(softmaxscores, dim=1))
                scores.append(
                    (softmaxscores * torch.log(softmaxscores)).sum(dim=1))
        else:
            scores = inputs

        ml, act = self.H.to_multilabel(labels)

        # Add synset scores to tracker
        for i, end_idx in enumerate(self.H.synset_bounds):
            # Gather active synsets
            active_synset_idxs = (act[:, i] == 1)
            synset_gt = (ml[active_synset_idxs])[:, i]
            synset_scores = scores[i][active_synset_idxs]
            synset_preds = softmaxpreds[i][active_synset_idxs]
            synset_gtroc = (synset_gt.eq(synset_preds)).cpu().numpy()

            self._scores[i] = np.concatenate(
                (self._scores[i], synset_scores.cpu().numpy().ravel()), axis=0)
            self._gtroc[i] = np.concatenate(
                (self._gtroc[i], synset_gtroc.ravel()), axis=0)

    def log_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            self.logger.info("{} Synset: Softmax threshold {}".format(
                             synset_desc, self._thresh[i]))

    def print_threshold(self):
        for i, end_idx in enumerate(self.H.synset_bounds):
            synset_parent = self.H.child2parent[self.H.class_list[end_idx]]
            synset_desc = self.H.class_description[synset_parent] \
                if synset_parent is not None else "ROOT"
            print("{} Synset: Softmax Pred threshold {}".format(
                  synset_desc, self._thresh[i]))

    def predict(self, inputs):
        # Calculate soft scores from logits
        hlogits = self.H.split_logits_by_synset(inputs)
        softscores, _, _, _ = hierarchy_metrics.gather_soft_scores(
            hlogits, self.H)
        # Get soft preds and path through hierarchy
        leaf_scores = self.H.to_leafs(softscores)
        # Max of leaf node scores
        leaf_preds = leaf_scores.max(dim=1)[1]

        # Calculate softmax scores
        scores = []
        for logit in hlogits:
            softmaxscores = torch.softmax(logit, dim=1)
            scores.append(
                (softmaxscores * torch.log(softmaxscores)).sum(dim=1))
        return super().predict(scores, leaf_preds)


class PathProbStoppingCriterion(StoppingCriterion):
    """Path probability stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr, pred_method='topdown')
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.PathProbStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)]
        self._gtroc = [np.empty((0,), dtype=float)]

    def update(self, inputs, labels, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, _, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            scores = inputs

        leaf_scores = self.H.to_leafs(scores)

        pred_scores, preds = torch.max(leaf_scores.cpu(), dim=1)
        ml, act = self.H.to_multilabel(preds)

        gtroc = labels.eq(preds)

        self._scores[0] = np.concatenate(
            (self._scores[0], pred_scores.cpu().numpy()), axis=0)
        self._gtroc[0] = np.concatenate(
            (self._gtroc[0], gtroc), axis=0)

    def print_threshold(self):
        print("Path Probability threshold: {}".format(self._thresh[0]))

    def log_threshold(self):
        self.logger.info("Path Probability threshold: {}".format(
                         self._thresh[0]))

    def predict(self, inputs, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, _, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            scores = inputs

        # Get soft preds and path through hierarchy
        leaf_scores = self.H.to_leafs(scores)
        # Max of leaf node scores
        leaf_preds = leaf_scores.max(dim=1)[1]
        return super().predict(scores, leaf_preds)


class PathEntropyStoppingCriterion(StoppingCriterion):
    """Path entropy stopping criterion"""
    def __init__(self, hierarchy, tnr=0.95):
        super().__init__(hierarchy, tnr, pred_method='topdown')
        self.logger = logging.getLogger(
            '__main__.hierarchy_inference.PathEntropyStoppingCriterion')
        self.H = hierarchy
        self._scores = [np.empty((0,), dtype=float)]
        self._gtroc = [np.empty((0,), dtype=float)]

    def update(self, inputs, labels, inp_scores=False):
        # Calculate soft scores from logits
        inputs = inputs.to(device)

        # Calculate path prediciton
        # Calculate synset entropys
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, entscores, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            raise NotImplementedError

        # Set GT based on path prediction (1 per image)
        leaf_scores = self.H.to_leafs(scores)
        _, preds = torch.max(leaf_scores.cpu(), dim=1)
        ml, act = self.H.to_multilabel(preds)

        gtroc = labels.eq(preds)

        # NOTE: Not a true micro-ROC
        # Cannot use all predictions as FP/TPs because all leafs that are
        # siblings have the same path entropy
        entrunning = torch.zeros(entscores.size(0))
        entcounts = torch.zeros(entscores.size(0))
        # Calculate mean pathwise entropy
        for i, end_idx in enumerate(self.H.synset_bounds):
            # Gather active pred path synsets
            active_synset_idxs = (act[:, i] == 1)
            entrunning += entscores[:, i] * active_synset_idxs
            entcounts += active_synset_idxs
        mean_predpath_ent = -entrunning / entcounts
        self._scores[0] = np.concatenate(
            (self._scores[0], mean_predpath_ent.cpu().numpy()), axis=0)
        self._gtroc[0] = np.concatenate(
            (self._gtroc[0], gtroc), axis=0)

    def print_threshold(self):
        print("Path Entropy threshold: {}".format(self._thresh[0]))

    def log_threshold(self):
        self.logger.info("Path Entropy threshold: {}".format(
                         self._thresh[0]))

    def predict(self, inputs, inp_scores=False):
        # Calculate soft scores from logits
        if not inp_scores:
            hlogits = self.H.split_logits_by_synset(inputs)
            scores, _, entscores, _ = hierarchy_metrics.gather_soft_scores(
                hlogits, self.H)
        else:
            scores = inputs

        # Set GT based on path prediction (1 per image)
        leaf_scores = self.H.to_leafs(scores)
        _, preds = torch.max(leaf_scores, dim=1)
        ml, act = self.H.to_multilabel(preds)

        entrunning = torch.zeros(entscores.size(0))
        entcounts = torch.zeros(entscores.size(0))
        # Calculate mean pathwise entropy
        for i, end_idx in enumerate(self.H.synset_bounds):
            # Gather active pred path synsets
            active_synset_idxs = (act[:, i] == 1)
            entrunning += entscores[:, i] * active_synset_idxs
            entcounts += active_synset_idxs
        mean_predpath_ent = -entrunning / entcounts
        return super().predict(mean_predpath_ent, preds)
