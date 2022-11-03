import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'
module_logger = logging.getLogger('__main__.hierarchy_loss')


class FocalLoss(nn.Module):
    """Focal loss from Lin et al. ICCV 2017 & TPAMI 2020

    Implementation from pytorch forums: https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    """
    def __init__(self, weight=None, gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )
        return loss


class HierarchicalOutlierExposureLoss:
    """Hierarchical outlier exposure criteria"""
    def __init__(self, hierarchy, normalize=False):
        self.logger = logging.getLogger('__main__.hierarchy_loss.HierarchicalOutlierExposureLoss')
        self.H = hierarchy
        # Move synset offsets and bounds to gpu
        self.syn_ob = torch.empty(
            (len(self.H.synset_bounds), 2), dtype=torch.long, device=device)
        self.targets = torch.zeros((len(self.H.synset_bounds),),
                dtype=torch.float32, device=device)
        for i in range(len(self.H.synset_bounds)):
            # Set offset and bounds
            off = self.H.synset_offsets[i]
            bound = self.H.synset_bounds[i]+1
            self.syn_ob[i, :] = torch.tensor(
                [off, bound], device=device).long()
            # Target is the entropy of a uniform distribution
            if normalize:
                self.targets[i] = torch.log(bound - off)
            else:
                self.targets[i] = 0.
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, logits, labels):
        # CPU Loop GPU Calc
        loss = torch.tensor(0., device=device)
        _, act = self.H.to_multilabel(labels)
        for i, (off, bnd) in enumerate(self.syn_ob):
            if (1. - act[:, i]).sum() > 0:
                # Calculate synset softmax probs
                temp = -logits[:, off:bnd].mean(1) + \
                        torch.logsumexp(logits[:, off:bnd], dim=1)
                temp = torch.abs(temp - self.targets[i])
                temp = temp * (1. - act[:, i])
                loss += temp.sum() / (1. - act[:, i]).sum()
        return loss


class HierarchicalSoftProbCELoss:
    """Hierarchical soft probability Cross Entropy criteria

    Parameters
    ----------
    H : hierarchy_util.Hierarchy

    Notes
    -----
    Calculates the cross entropy loss for each synset and multiplies by the
    boolean active synset indicator to remove "inactive" synsets:
    .. math::
        - \sum^{N}_{i=1} \sum^{# Synsets}_{s=1} \sum^{# Classes \in s}_{k=1}
        \mathbb{I}(k==y_s) \textsc{LogSoftmax}(\text{logit}^{(i)}_s)_k
    This is equivalent to the cross entropy of the leaf nodes:
    .. math::
        - \sum^{N}_{i=1} \sum^{# Classes \in leafs}_{k=1}
        \mathbb{I}(k==y_{leafs}) \log(\Pr(k|x_i))
    where :math:`\Pr(k|x_i)` is the path probability of leaf class k.

    However, due to the computational overhead of slicing the tensors, it is
    more efficient to calculate synset wise CE and sum.
    """
    def __init__(self, H, depth_weight=False, focal=False):
        self.H = H
        self.synset_parents = torch.empty(
            (len(self.H.synset_bounds),), dtype=torch.long, device=device)
        # Move synset offsets and bounds to gpu
        self.syn_ob = torch.empty(
            (len(self.H.synset_bounds), 2), dtype=torch.long, device=device)

        self.leaf_idxs = self.H.leaf_ids.clone().detach().to(device)
        # self.leaf_idxs.requires_grad = True
        for i in range(len(self.H.synset_bounds)):
            # Set parents
            p = self.H.get_synsetid_parent(i)
            self.synset_parents[i] = p if p is not None else -1
            # Set offset and bounds
            off = self.H.synset_offsets[i]
            bound = self.H.synset_bounds[i]+1
            self.syn_ob[i, :] = torch.tensor(
                [off, bound], device=device).long()
        self.weighted_ce = True if H.ce_weights is not None else False
        self.focal = focal
        if self.focal:
            crit = FocalLoss
        else:
            crit = torch.nn.CrossEntropyLoss
        if H.ce_weights is not None:
            self.criterion = []
            for off, bnd in self.syn_ob:
                l = crit(weight=H.ce_weights[off:bnd], reduction='none')
                self.criterion.append(l)
        else:
            self.criterion = [crit(reduction='none')] * len(self.syn_ob)
        if depth_weight:
            self.synset_weights = H.norm_factors.to(device)
            self.synset_weights.requires_grad = False
        else:
            self.synset_weights = torch.ones((len(H.synset_bounds),),
                                             device=device, requires_grad=False)

    def __call__(self, logits, labels):
        # CPU Loop GPU Calc
        loss = torch.zeros(logits.shape[0], device=device)
        ml, act = self.H.to_multilabel(labels)
        for i, (off, bnd) in enumerate(self.syn_ob):
            # Calculate synset softmax probs
            loss += (self.synset_weights[i] *
                     self.criterion[i](logits[:, off:bnd], ml[:, i]) *
                     act[:, i])  # Remove non_active loss contributions
        loss = loss.mean()
        return loss


class HierarchicalSoftProbAndSynsetCELoss:
    """Hierarchical soft probability and synset cross entropy criteria"""
    def __init__(self, hierarchy, epochs,
                 synset_weight_range=[1., 0.],
                 softprob_weight_range=[0., 5.],
                 outlier_weight_range=[0., 0.],
                 focal_loss=False,
                 depth_weight=True,
                 ):
        self.logger = logging.getLogger('__main__.hierarchy_loss.HierarchicalSoftProbAndSynsetCELoss')
        self.hierarchy = hierarchy
        self.softprob = HierarchicalSoftProbCELoss(hierarchy,
                                                   depth_weight=depth_weight,
                                                   focal=focal_loss)
        self.outlier = HierarchicalOutlierExposureLoss(hierarchy)
        self.softprob_wt_gen = self.weight_gen(softprob_weight_range, epochs)
        self.outlier_wt_gen = self.weight_gen(outlier_weight_range, epochs)
        if softprob_weight_range[0] == 0.0 and softprob_weight_range[1] == 0.0:
            self.softprob = lambda *args, **kwargs: 0.
        if outlier_weight_range[0] == 0.0 and outlier_weight_range[1] == 0.0:
            self.outlier = lambda *args, **kwargs: 0.
        self.step_weights()

    def weight_gen(self, wrange, num_epochs):
        i = 0
        w_spacing = torch.linspace(*wrange, num_epochs+1).to(device)
        while True:
            yield w_spacing[i]
            i = i+1 if i < num_epochs else i

    def step_weights(self):
        self._softprob_weight = next(self.softprob_wt_gen)
        self._outlier_weight = next(self.outlier_wt_gen)

    @property
    def softprob_weight(self):
        """Current weight for soft prediction cross-entropy loss"""
        return self._softprob_weight

    @property
    def outlier_weight(self):
        """Current weight for outlier exposure loss"""
        return self._outlier_weight

    def print_weights(self):
        self.logger.info(
            "Soft prediction CE Weight: {}\nOutlier Exposure Weight: {}".format(
                self.softprob_weight, self.outlier_weight))

    def __call__(self, outputs, labels,
                 return_list=False):
        sploss = self.softprob(outputs, labels)
        sploss *= self.softprob_weight
        outloss = self.outlier(outputs, labels)
        outloss *= self.outlier_weight
        if return_list:
            return [sploss, outloss]
        return sploss + outloss


class MOSLoss:
    """MOS Loss"""
    def __init__(self, hierarchy):
        self.logger = logging.getLogger('__main__.hierarchy_loss.MOSLoss')
        if hierarchy.max_depth != 2:
            if hierarchy.max_depth < 2:
                raise RuntimeError("Hierarchy must be depth 2 for MOS Head")
            warnings.warn("Hierarchy depth > 2 for MOS Head", RuntimeWarning)
        self.hierarchy = hierarchy
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        # Move synset offsets and bounds to gpu
        self.syn_ob = self.hierarchy.get_MOS_offsets_bounds()

    def __call__(self, outputs, labels):
        """Calculate soft prob cross entropy loss"""
        lbls = self.hierarchy.to_MOSlabel(labels)
        loss = 0.
        for i in range(self.syn_ob.size(0)):
            curr_logits = outputs[:, self.syn_ob[i, 0]:self.syn_ob[i, 1]+1]
            curr_labels = lbls[:, i]
            loss += self.criterion(curr_logits, curr_labels)
        return loss
