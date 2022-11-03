""" Utilities for manipulating hierarchy

Utilities were influenced by the ImageNet21K implementation.
"""
import numpy as np
import torch
import logging

_DEFAULT_HIERARCHY = "./pruned-wn.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
module_logger = logging.getLogger('__main__.hierarchy_util')
_MIN_NORM_FACTOR = 1./20.


class Hierarchy:
    """Label hierarchy

    Parameters
    ----------
    train_classes : list
        Tensor of the classes included in the training set
    hierarchy_fn : string
        Hierarchy filename

    Raises
    ------
    KeyError if the dataset contains a key not found in the hierarchy file

    Attributes
    ----------
    num_classes : int
        The number of classes in the hierarchy. Initialized to 0 until
        hierarchy is initialized with the current dataset.

    Methods
    -------
    to_multilabel(label: torch.Tensor): -> torch.Tensor, torch.tensor
    split_logits_by_synset(logits: torch.Tensor): -> list(torch.Tensor)
    """

    def __init__(self,
                 train_classes,
                 hierarchy_fn=_DEFAULT_HIERARCHY,
                 min_norm_factor=_MIN_NORM_FACTOR,
                 ):
        self.hierarchy = torch.load(hierarchy_fn)
        self.class_description = self.hierarchy['class_description']
        self.child2parent = self.hierarchy['child2parent']
        self.train_classes = train_classes
        res = self._trim_hierarchy(train_classes)
        self.class_list, self.class_parents = res
        self._num_classes = len(self.class_list)
        self._max_depth = self._gen_maxdepth()
        self.synset_bounds = self._gen_synset_bounds(self.class_list)
        self.full_synset_bounds = self._gen_synset_bounds(
            self.hierarchy['class_list'])
        self.synset_offsets = self._gen_synset_offsets(self.synset_bounds)
        self.full_synset_offsets = self._gen_synset_offsets(
            self.full_synset_bounds)
        self.min_norm_factor = min_norm_factor
        self.norm_factors = self._gen_norm_factors()
        self.leaf_ids = self._gen_leaf_ids()  # In order of train_list
        self.num_leafs = len(self.leaf_ids)
        self._multilabel_map, self._active_synset_map = self._to_multilabel(
            list(range(len(self.train_classes))),
            self.synset_bounds, self.class_parents,
            self.class_list, is_leaf_idx=True)
        # full_leafs = list(range(len(self.hierarchy['class_list'])))
        # for k, v in self.hierarchy['class_parents'].items():
        #     for idx in v:
        #         try:
        #             full_leafs.remove(idx)
        #         except ValueError:
        #             pass
        self._full_multilabel_map, self._full_active_synset_map = \
            self._to_multilabel(
                list(range(len(self.train_classes))),
                self.full_synset_bounds,
                self.hierarchy['class_parents'],
                self.hierarchy['class_list'],
                is_leaf_idx=True)
        self._MOSlabel_map = self._gen_MOSlabels()
        self._MOSlabelidxs_map = self._gen_MOSlabelidxs()
        self._synset_parent_ids = [self._get_synsetid_parent(i)
                                   for i in range(len(self.synset_bounds))]
        self._class_synset_offsets = [self._classid2synsetid_offset(i)
                                      for i in range(len(self.class_list))]
        self._leaf_synset_offsets = [self._classid2synsetid_offset(i)
                                      for i in self.leaf_ids]
        self.ce_weights = None

    def _trim_hierarchy(self, train_classes):
        """Remove classes that are not in the train_classes hierarchy

        Parameters
        ----------
        train_classes : list
            The WordNet ids for the classes to be used for training

        Returns
        -------
        class_list, class_parents
            class_list : list(string)
                The WordNet ids of the training classes ordered to preserve the
                hierarchy synsets
            class_parents : dict(string -> list(int))
                The parent list for the training classes keyed by class WordNet
                id and the parent list ints corresponding to trimmed class_list
                indexes
        """
        # Avoid duplicates and disassociate from train_classes
        classes_to_keep = set(train_classes)
        # Add all parent classes
        # FIXME: This is not efficient but not an issue for now
        for c in train_classes:
            for p in self.hierarchy['class_parents'][c]:
                classes_to_keep.add(self.hierarchy['class_list'][p])
        # Duplicate tree and remove unnecessary classes
        # NOTE: the hierarchy's class list indexing groups classes with the
        # same parent and must therefore be used
        # FIXME: This is not efficient but not an issue for now
        class_list = list(self.hierarchy['class_list'])
        class_parents = dict(self.hierarchy['class_parents'])
        i = 0
        while i < len(class_list):
            if class_list[i] not in classes_to_keep:
                del class_parents[class_list[i]]
                del class_list[i]
                continue
            i += 1
        # Catch unexpected logic errors
        if set(class_list) != classes_to_keep:
            raise RuntimeError("Class list is inconsistent with hierarchy")
        # Recreate class parents to reflect updated class_list
        # FIXME: This is not efficient but not an issue for now
        for k, v in class_parents.items():
            updated_parent_idxs = []
            for idx in v:
                updated_parent_idxs.append(
                    class_list.index(self.hierarchy['class_list'][idx]))
            class_parents[k] = updated_parent_idxs
        return class_list, class_parents

    def _gen_maxdepth(self,):
        """Get the maximum depth of tree"""
        max_depth = 0
        for pars in self.class_parents.values():
            if len(pars) > max_depth:
                max_depth = len(pars)
        return max_depth + 1

    def _gen_synset_bounds(self, class_list):
        """Set the bounds of the hierarchy synsets

        Parameters
        ----------
        class_list : list(string)
            the class list to get the synset bounds for

        Return
        ------
        list(int) :
            indices of the last member of each synset
        """
        bounds = []
        i = 0
        curr_parent = self.child2parent[class_list[i]]
        # Loop over all parents and append last member to list
        while i < len(class_list):
            if self.child2parent[class_list[i]] != curr_parent:
                bounds.append(i-1)
                curr_parent = self.child2parent[class_list[i]]
            i += 1
        bounds.append(len(class_list)-1)
        return bounds

    def _gen_synset_offsets(self, synset_bounds):
        """Create a synset offset tensor

        Parameters
        ----------
        synset_bounds : list(int)

        Return
        ------
        torch.tensor :
            offsets for each synset
        """
        offsets = torch.zeros(len(synset_bounds)).to(device)
        offsets[1:] += torch.tensor(synset_bounds[:-1]).to(device) + 1
        return offsets

    def _gen_norm_factors(self):
        """Initialize the normalization factors for a given set of classes

        Note that the method of calculating norm factors is different from the
        ImageNet21K method. We calculate the number of subclasses contained
        within each softmax which is proportional to the number of times that
        softmax is activated during training since the class occurences are
        balanced.

        In contrast, ImageNet21K uses the number of classes at each depth of
        the hierarchy.
        """
        norms = np.zeros(len(self.synset_bounds))
        class_counts = np.ones(len(self.class_list))
        for c in self.class_list:
            for p in self.class_parents[c]:
                class_counts[p] += 1
        cumsum_class_counts = np.cumsum(class_counts)
        start_class_count = 0
        for i, end_idx in enumerate(self.synset_bounds):
            norms[i] = cumsum_class_counts[end_idx] - start_class_count
            start_class_count = cumsum_class_counts[end_idx]
        norms = norms/norms[0]
        norms = np.clip(norms, self.min_norm_factor, None)
        return torch.tensor(norms)

    def gen_CEweights(self, trainloader):
        """Generate the weights used for CE loss for a given dataset

        Note that the method of calculating norm factors is different from the
        ImageNet21K method. We calculate the number of subclasses contained
        within each softmax which is proportional to the number of times that
        softmax is activated during training since the class occurences are
        balanced.

        In contrast, ImageNet21K uses the number of classes at each depth of
        the hierarchy.
        """
        weights = torch.zeros((self._num_classes,), dtype=int, device=device)
        for _, targets in trainloader:
            ml, act = self.to_multilabel(targets) 
            act_idxs = torch.where(act)    
            class_idxs = ml[act_idxs] + self.synset_offsets[act_idxs[1]]
            idxs, counts = torch.unique(class_idxs, return_counts=True)
            weights[idxs.long()] += counts
        self.ce_weights = 1./weights

    def _gen_leaf_ids(self):
        leaf_ids = torch.zeros((len(self.train_classes)),).long() 
        for i, c in enumerate(self.class_list):
            if self.get_hyponym_synsetid(torch.tensor([i, ])) == -1:
                leaf_ids[self.train_classes.index(c)] = i
        return leaf_ids

    def _gen_MOSlabels(self):
        syn_ob = self.get_MOS_offsets_bounds()
        # Generate labels for all leaf nodes
        labels = torch.empty((0, syn_ob.size(0)))
        for lbl_wnid in self.train_classes:
            lbl = self.class_list.index(lbl_wnid)
            lbl_tensor = torch.zeros(len(syn_ob))
            for i, (off, bound, orig_syn_id) in enumerate(syn_ob):
                # Check if label in current group by comparing class_list index
                # to original synset offsets and bounds
                # Bounds are inclusive
                if (lbl >= self.synset_offsets[orig_syn_id] and
                        lbl <= self.synset_bounds[orig_syn_id]):
                    lbl_tensor[i] = lbl - self.synset_offsets[orig_syn_id]
                else:
                    lbl_tensor[i] = bound - off
            labels = torch.cat((labels, lbl_tensor.view(1, -1)))
        labels = labels.long().to(device)
        return labels

    def _gen_MOSlabelidxs(self):
        syn_ob = self.get_MOS_offsets_bounds()
        # Generate labels for all leaf nodes
        labels = []
        for lbl_wnid in self.train_classes:
            lbl = self.class_list.index(lbl_wnid)
            for i, (off, bound, orig_syn_id) in enumerate(syn_ob):
                # Check if label in current group by comparing class_list index
                # to original synset offsets and bounds
                # Bounds are inclusive
                if (lbl >= self.synset_offsets[orig_syn_id] and
                        lbl <= self.synset_bounds[orig_syn_id]):
                    labels.append(lbl - self.synset_offsets[orig_syn_id] + off)
                    break
        labels = torch.tensor(labels).long().to(device)
        return labels

    @property
    def num_classes(self):
        """Get the number of classes in the hierarchy"""
        return self._num_classes

    @property
    def max_depth(self):
        """Get the maximum depth of the hierarchy"""
        return self._max_depth

    def _to_multilabel(self, label, synset_bounds, class_parents, class_list,
                       is_leaf_idx=True):
        multilabels = torch.empty((0, len(synset_bounds)))
        active_tensors = torch.empty((0, len(synset_bounds)))
        for lbl in label:
            if is_leaf_idx:
                wnid = self.train_classes[lbl]
            else:
                wnid = class_list[lbl]
            ml = list(class_parents[wnid])
            ml.append(class_list.index(wnid))

            active_synsets = torch.searchsorted(
                    torch.tensor(synset_bounds), torch.tensor(ml))
            ml_tensor = torch.zeros(len(synset_bounds))
            active_tensor = torch.zeros(len(synset_bounds))
            active_tensor[active_synsets] = 1
            for i, s in enumerate(active_synsets):
                if s == 0:
                    ml_tensor[s] = ml[i]
                else:
                    ml_tensor[s] = ml[i]-synset_bounds[s-1]-1
            multilabels = torch.cat((multilabels, ml_tensor.view(1, -1)))
            active_tensors = torch.cat((active_tensors,
                                        active_tensor.view(1, -1)))
        multilabels = multilabels.long().to(device)
        active_tensors = active_tensors.to(device)
        return multilabels, active_tensors

    def _fast_to_multilabel(self, label, toFull=False):
        """Convert to multilabel by tensor slicing

        Parameters
        ----------
        label : torch.tensor (on device)
        toFull : bool
            Whether to return the full or regular label
        """
        if toFull:
            return self._full_multilabel_map[label], \
                   self._full_active_synset_map[label]
        return self._multilabel_map[label], self._active_synset_map[label]

    def to_multilabel(self, label, is_leaf_idx=True):
        """Convert batch of single labels to multilabel according to hierarchy

        Parameters
        ----------
        label : torch.Tensor (on device)
            Batch of labels  to convert to multilabels

        Return
        ------
        torch.tensor, torch.tensor :
            multilabels : torch.tensor
                The multilabel for each image in the batch. Each image's
                multilabel is encoded as a zero vector where the active synsets
                have the appropriate label assigned (all else are  0)
                Shape -> [Batch Size, Num Synsets]
            active_tensor : torch.tensor
                Indicator for which synsets are active for each image
                Shape -> [Batch Size, Num Synsets]
        """
        if is_leaf_idx:
            ml, act = self._fast_to_multilabel(label, toFull=False)
        else:
            label = label.cpu().tolist()
            ml, act = self._to_multilabel(label, self.synset_bounds,
                                      self.class_parents, self.class_list,
                                      is_leaf_idx)
        return ml, act

    def to_full_multilabel(self, label, is_leaf_idx=True):
        """Convert batch of single labels to multilabel according to full hierarchy.

        Parameters
        ----------
        label : torch.Tensor
            The label to convert to full hierarchy multilabel

        Returns
        -------
        torch.Tensor :
            The full hierarchy multilabel
        """
        if is_leaf_idx:
            ml, act = self._fast_to_multilabel(label, toFull=True)
        else:
            label = label.cpu().tolist()
            ml, act = self._to_multilabel(label, self.full_synset_bounds,
                                      self.hierarchy['class_parents'],
                                      self.hierarchy['class_list'],
                                      is_leaf_idx)
        return ml, act

    def trim_full_multilabel(self, multilabel, active_synsets):
        """Remove synsets that are not present in the current hierarchy

        Parameters
        ----------
        multilabel : torch.tensor
            the full hierarhcy multilabel to trim
        active_synsets : torch.tensor
            the active synsets to trim

        Returns
        -------
        torch.tensor, torch.tensor
            The trimmed multilabel, the trimmed active synsets
        """
        # Convert multilabels to idxs
        ml_idxs = multilabel + self.full_synset_offsets
        # Get active labels
        active_idxs = torch.where(active_synsets == 1)
        active_ml_idxs = ml_idxs[active_idxs]
        # import pdb; pdb.set_trace()
        active_ml_wnids = np.array([self.hierarchy['class_list'][int(i)]
                                    for i in active_ml_idxs])
        # Trim labels not in train_classes
        labels_to_keep = np.isin(active_ml_wnids, self.class_list)
        wnids = active_ml_wnids[labels_to_keep]
        lbl_idxs = torch.tensor([self.class_list.index(w) for w in wnids])
        img_idxs = active_idxs[0][labels_to_keep]
        trimmed_active_synsets = torch.searchsorted(
            torch.tensor(self.synset_bounds), lbl_idxs)
        # Assign multilabels and active synsets
        trimmed_ml = torch.zeros(multilabel.size(0), len(self.synset_bounds))
        trimmed_at = torch.zeros(multilabel.size(0), len(self.synset_bounds))
        trimmed_at[img_idxs, trimmed_active_synsets] = 1

        trimmed_ml[img_idxs, trimmed_active_synsets] = \
            lbl_idxs - self.synset_offsets[trimmed_active_synsets].cpu()
        return trimmed_ml.to(device), trimmed_at.to(device)

    def get_MOS_offsets_bounds(self):
        # Move synset offsets and bounds to gpu
        syn_ob = list()
        start_idx = 0
        for i in range(len(self.synset_bounds)):
            # Set offset and bounds
            off = int(self.synset_offsets[i])
            bound = self.synset_bounds[i]
            if off in self.leaf_ids:
                # class range : [offset, bound] (inclusive)
                # num valid classes = bound - offset + 1
                # num total classes = num valid classes+1 = bound - offset + 2
                # since bound is inclusive remove 1 -> bound - offset + 1
                syn_ob.append([start_idx, start_idx + bound - off + 1, i])
                start_idx += bound-off+1+1
        return torch.tensor(syn_ob, dtype=torch.long, device=device)

    def get_num_MOSclasses(self):
        """Return the number of classes for a MOS classifier

        Number of MOS classes = num leaf classes + num leaf groups.
        """
        return len(self.train_classes) + self._MOSlabel_map.size(1)

    def to_MOSlabel(self, label):
        """Convert batch of single labels to MOS label according to hierarchy

        Parameters
        ----------
        label : torch.Tensor (on device)
            Batch of labels  to convert to multilabels

        Return
        ------
        torch.tensor:
            The label for each image in the batch for each MOS group. Each
            image's label is encoded as a vector whose elements correspond to
            the label for each MOS group
            Shape -> [Batch Size, Num Groups]
        """
        return self._MOSlabel_map[label]

    def to_MOSlabelidx(self, label):
        """Convert batch of labels to MOS labels according to hierarchy

        Parameters
        ----------
        label : torch.Tensor (on device)

        Return
        ------
        torch.tensor:
            The label for each image in the batch that corresponds to the
            non-other true label index. 
            Shape -> [Batch Size,]
        """
        return self._MOSlabelidxs_map[label]

    def split_logits_by_synset(self, logits):
        """Splits the flattened logits into synsets

        Parameter
        ---------
        logits : torch.tensor (on device)
            The flat logits to be split
            Shape -> [batch size, number of classes in hierarchy]

        Return
        ------
        list(torch.tensor (on device)) :
            The split logits for calculating the loss. Each synset is an entry
            in the list.
            len(list) = # of synsets
            Shape of list element -> [batch size, number of classes in synset]
        """
        split_logits = []
        start_idx = 0
        for end_idx in self.synset_bounds:
            split_logits.append(logits[..., start_idx:end_idx+1])
            start_idx = end_idx+1
        return split_logits

    def get_hyponym_synsetid(self, parents):
        """Return the synset id of the hyponym set for a tensor of parents

        Given the tensor of parent class ids, find the synset containing
        hyponyms for each parent. Return -1 for leaf nodes.

        Parameter
        ---------
        parents : torch.LongTensor (on device)

        Return
        ------
        torch.LongTensor :
            The synset id of the hyponyms for each parent
        """
        # get unique list of parent classes
        unique_parents = torch.unique(parents).cpu()
        result = torch.ones(parents.size(0)).long()*-1

        # Get synset of parents
        parent_synset = torch.searchsorted(
            torch.tensor(self.synset_bounds), unique_parents)

        # Loop over parents
        for p,curr_s in zip(unique_parents, parent_synset):
            curr_par = self.class_list[p]
            # Loop over first class in each synset and check if parent in
            # parent list
            s = curr_s + 1
            while s < len(self.synset_bounds):
                child_p = self.child2parent[self.class_list[self.synset_bounds[s]]]
                if child_p == curr_par:
                    isFound = True
                    result[parents.eq(p)] = s
                    break
                s += 1
        return result.long().to(device)

    def get_synsetid_depth(self, synset_id):
        """Return the depth of the synset id.

        Parameter
        ---------
        synset_id : torch.long

        Return
        ------
        int :
            The depth of the synset
        """
        # get list of parent classes
        parent_list  = self.class_parents[self.class_list[
            self.synset_bounds[synset_id]]]
        return len(parent_list)

    def _get_synsetid_parent(self, synset_id):
        """Return the parent of the synset id.
        
        For internal use to generate the synset id to parent id mapping.

        Parameter
        ---------
        synset_id : torch.long

        Return
        ------
        int :
            The parent id of the synset
        """
        class_id = self.class_list[self.synset_bounds[synset_id]]
        parent = self.child2parent[class_id]
        parent_id = None if parent is None else self.class_list.index(parent)
        return parent_id

    def get_synsetid_parent(self, synset_id):
        """Return the parent of the synset id.

        Parameter
        ---------
        synset_id : torch.long

        Return
        ------
        int :
            The parent id of the synset
        """
        return self._synset_parent_ids[synset_id]

    def _classid2synsetid_offset(self, class_id):
        """Return the synset id and offset for the class_id.

        For internal use to generate the class id to synset, offset mapping.

        Parameter
        ---------
        class_id : torch.long

        Return
        ------
        int, int :
            The synset id and offset for the class_id
        """
        synset_id = torch.searchsorted(
                torch.tensor(self.synset_bounds), class_id)
        offset = class_id - self.synset_offsets[synset_id]
        return synset_id, offset.long()

    def classid2synsetid_offset(self, class_id):
        """Return the synset id and offset for the class_id.

        Parameter
        ---------
        class_id : torch.long

        Return
        ------
        int, int :
            The synset id and offset for the class_id
        """
        return self._class_synset_offsets[class_id]

    def to_leafs(self, inputs):
        """Convert inputs to only leaf node values

        Parameters
        ----------
        inputs : list(torch.tensor) or torch.tensor
            Inputs to process either in hlogits format or in a flat tensor
            input corresponding to class_list.

        Returns
        -------
        torch.tensor :
            the values corresponding to the leaf nodes ordered like
            self.train_classes
        """
        outputs = torch.zeros(
                (inputs[0].size(0), self.num_leafs),
                dtype=inputs[0].dtype, device=inputs[0].device)
        for i, (s, o) in enumerate(self._leaf_synset_offsets):
            outputs[:, i] = inputs[s][:, o]
        return outputs
