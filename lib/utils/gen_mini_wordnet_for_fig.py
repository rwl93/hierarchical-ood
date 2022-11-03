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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hierarchy', metavar="FILENAME",
                        default='pruned-wn.pth',
                        help='Hierarchy pth filename')
    parser.add_argument('--wnids-to-keep', metavar="FILENAME",
                        default='data/mini-wn-labels.csv',
                        help='Classes to keep in the hierarchy')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='mini-wn.pth',
                        help='Hierarchy pth output file ' +
                             '(will be overwritten)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    wnids2keep = pd.read_csv(args.wnids_to_keep, header=None)[0].to_list()

    hierarchy = torch.load(args.hierarchy) 
    class_list = [wnid for wnid in hierarchy['class_list'] if wnid in wnids2keep] 
    class_description = {}
    child2parent = {}
    class_parents = {}
    root = None
    for wnid in wnids2keep:
        class_description[wnid] = hierarchy['class_description'][wnid]
        if hierarchy['child2parent'][wnid] in class_list:
            child2parent[wnid] = hierarchy['child2parent'][wnid] 
        else:
            if root is not None:
                raise ValueError("Class list must have a single parent")
            root = wnid
            child2parent[wnid] = None
        class_parents[wnid] = []
        for p in hierarchy['class_parents'][wnid]:
            p_wnid = hierarchy['class_list'][p]
            if p_wnid in class_list:
                class_parents[wnid].append(class_list.index(p_wnid))

    mini_hierarchy = {
        'class_list': class_list,
        'class_description': class_description,
        'child2parent': child2parent,
        'class_parents': class_parents
    }
    torch.save(mini_hierarchy, args.outfile)


if __name__ == "__main__":
    main()
