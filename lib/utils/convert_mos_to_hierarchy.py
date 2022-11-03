"""Script converts the MOS ImageNet 2012 groups to a hierarchy"""

import torch
from nltk.corpus import wordnet as wn
import argparse
import pandas as pd


def wnid_to_description(wnid):
    synset = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
    return synset.name().split(".")[0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mos-groups', metavar="FILENAME",
                        default='mos-groups-imagenet2012.txt',
                        help='MOS Groups filename')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='imagenet1000-mos.pth',
                        help='Hierarchy pth output file ' +
                             '(will be overwritten)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read MOS
    mos_groups = pd.read_csv('mos-groups-imagenet2012.txt', sep=' ', header=None,
                             names=['path', 'group id', 'class offset'])
    wnids = []
    for pth in mos_groups.loc[:, 'path']:
        wnids.append(pth.split('/')[1])
    mos_groups['wnid'] = wnids

    # Convert to hierarchy format
    child2parent = {}
    class_parents = {}
    class_list = []
    class_description = {}

    # Add groups to class list
    num_groups = len(mos_groups['group id'].unique())
    group_desc = ['group' + str(i) for i in range(num_groups)]
    class_list.extend(group_desc)
    for desc in group_desc:
        class_description[desc] = desc
        child2parent[desc] = None
        class_parents[desc] = []

    # Loop over wnids sorted by group
    for idx in range(num_groups):
        group_wnids = mos_groups['wnid'][mos_groups['group id'] == idx]
        for wnid in group_wnids.unique():
            class_list.append(wnid)
            class_parents[wnid] = [idx]
            child2parent[wnid] = class_list[idx]
            class_description[wnid] = wnid_to_description(wnid)

    torch.save({"class_list": class_list,
                "class_description": class_description,
                "child2parent": child2parent,
                "class_parents": class_parents}, args.outfile)


if __name__ == "__main__":
    main()
