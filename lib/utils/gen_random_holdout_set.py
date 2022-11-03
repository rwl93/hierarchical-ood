"""Module provides functions for generating WordNet hierarchies

Note
----
Module is heavily based on NBDT's hierarchy implementation.
"""

import torch
from nltk.corpus import wordnet as wn
import networkx as nx
from networkx.readwrite.json_graph import node_link_graph
import json
import os
import argparse
import pandas as pd
import hierarchy_util
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--leafs', metavar="FILENAME",
                        default='data/wnids1k.csv',
                        help='File containing leaf node ids')
    parser.add_argument('--hierarchy', metavar="FILENAME",
                        default='imagenet1000-wn.pth',
                        help='Hierarchy pth filename')
    parser.add_argument('--prune-prob', metavar="FLOAT",
                        default=0.1, type=float,
                        help="Probability of holding out classes")
    parser.add_argument('--outfile-base', metavar="FILENAME",
                        default='imagenet1000',
                        help='Filename base to write id and ood labels ' +
                             '(will be overwritten)')
    parser.add_argument('--seed', metavar="RANDOMSEED",
                        default=1234567, type=int,
                        help='Random seed for holding out classes')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    # Load hierarchy
    leaf_wnids = pd.read_csv(args.leafs, header=None)
    leaf_wnids = leaf_wnids[0].to_list()
    h = hierarchy_util.Hierarchy(leaf_wnids, args.hierarchy)

    # Random Pruning
    id_classes = list(range(len(h.class_list)))
    max_depth = h.max_depth
    depth_range = (max_depth-3) // 3
    ood_classes = {}
    start_depth = 3
    lvls = ['coarse', 'medium', 'fine']
    holdout_levels = np.ones((3,))*args.prune_prob / [20, 4, 1]
    holdout_levels = dict(zip(lvls, holdout_levels))
    print(holdout_levels)
    for lvl in lvls:
        in_range_classids = []
        ood_classes[lvl] = set() 
        if lvl == 'fine':
            smaxdepth = max_depth
        else:
            smaxdepth = start_depth + depth_range
        print("{} depth range = [{},{})".format(lvl,
                                                start_depth, smaxdepth))
        # Get all id classes at lvl
        for i, (offset, bound) in enumerate(zip(h.synset_offsets,
                                                h.synset_bounds)):
            offset, bound = int(offset), int(bound)
            sdepth = h.get_synsetid_depth(i)
            if sdepth >= start_depth and sdepth < smaxdepth:
                in_range_classids.extend(list(range(offset, bound+1)))

        # Remove nodes missing from id_classes
        for idx in in_range_classids:
            if idx not in id_classes:
                in_range_classids = [i for i in in_range_classids if i != idx]

        # Randomly select nodes to holdout
        in_range_classids = np.array(in_range_classids) 
        hldout_probs = np.random.uniform(size=(len(in_range_classids),)) 
        hldouts = in_range_classids[hldout_probs < holdout_levels[lvl]]
        
        # Remove holdouts and children from id_classes
        id_classes = [i for i in id_classes if not (i in hldouts)]
        for hld in hldouts:
            ood_classes[lvl].add(hld)
            for k, v in h.class_parents.items():
                if (hld in v) and (h.class_list.index(k) in id_classes):
                    id_classes.remove(h.class_list.index(k))
                    ood_classes[lvl].add(h.class_list.index(k))
                    
        start_depth += depth_range

    # Get leaf nodes
    id_leafs = [i for i in id_classes if i in h.leaf_ids]
    ood_leafs = {lvl: [i for i in ood_classes[lvl] if i in h.leaf_ids]
                 for lvl in lvls}

    # Convert to wnids
    id_wnids = [h.class_list[i] for i in id_leafs]
    ood_wnids = {lvl: [h.class_list[i] for i in ood_leafs[lvl]]
                 for lvl in lvls}

    # Write output
    id_filename = args.outfile_base + "_id_labels.csv"
    ood_filename = args.outfile_base + "_ood_labels.csv"

    id_df = pd.DataFrame(id_wnids)
    id_df.to_csv(id_filename, header=False, index=False)

    oodidxs = []
    oodlvls = []
    for k, v in ood_wnids.items():
        oodidxs.extend(v)
        oodlvls.extend(len(v) * [k])
    ood_df = pd.DataFrame({'idx': oodidxs, 'lvl': oodlvls})
    ood_df.to_csv(ood_filename, header=False, index=False)


if __name__ == "__main__":
    main()
