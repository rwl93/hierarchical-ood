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


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbdt-graph', metavar="FILENAME",
                        default='nbdt-imagenet1000-wordnet.json',
                        help='Hierarchy pth filename')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='imagenet1000-wn.pth',
                        help='Hierarchy pth output file ' +
                             '(will be overwritten)')
    return parser


# CODE FROM NBDT
def is_leaf(G, node):
    return len(G.succ[node]) == 0


# CODE FROM NBDT
def get_leaves(G, root=None):
    nodes = G.nodes if root is None else nx.descendants(G, root) | {root}
    for node in nodes:
        if is_leaf(G, node):
            yield node


# CODE FROM NBDT
def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node


# CODE FROM NBDT
def get_root(G):
    roots = list(get_roots(G))
    assert len(roots) == 1, f"Multiple ({len(roots)}) found"
    return roots[0]


# CODE FROM NBDT
def read_graph(path):
    if not os.path.isfile(path):
        raise ValueError("No such file: {path}.")
    with open(path) as f:
        return node_link_graph(json.load(f))


def wnid_to_description(wnid):
    try:
        synset = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
        return synset.name().split(".")[0]
    except:
        return 'generated'


class FakeSynset:
    def __init__(self, wnid):
        self.wnid = wnid

        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset("f{:08d}".format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return "f"

    def name(self):
        return "(generated)"

    def definition(self):
        return "(generated)"

def gen_hierarchy(G, wnid, class_list, child2parent):
    children = G.succ[wnid]
    for c in children:
        class_list.append(c)
        child2parent[c] = wnid
    for c in children:
        gen_hierarchy(G, c, class_list, child2parent)


def gen_parents(class_list, child2parent, class_parents):
    for i, c in enumerate(class_list):
        pars = list()
        p = child2parent[c]
        if p is not None:
            pars.extend(list(class_parents[p]))
            pars.append(class_list.index(p))
        class_parents[c] = pars


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read NBDT graph
    G = read_graph(args.nbdt_graph)

    # Get root node
    root = get_root(G)

    # Get leaves
    # leaves = list(get_leaves(G))

    # Convert to hierarchy format
    class_list = []
    class_description = {}
    child2parent = {}
    class_parents = {}

    gen_hierarchy(G, root, class_list, child2parent)

    # Remove references to root node
    for c in G.succ[root]:
        child2parent[c] = None

    gen_parents(class_list, child2parent, class_parents)

    for c in class_list:
        class_description[c] = wnid_to_description(c)

    torch.save({"class_list": class_list,
                "class_description": class_description,
                "child2parent": child2parent,
                "class_parents": class_parents}, args.outfile)


if __name__ == "__main__":
    main()
