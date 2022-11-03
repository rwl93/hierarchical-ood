"""Convert JSON to hierarchy"""

import torch
import json
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', metavar="FILENAME",
                        default='nbdt-imagenet1000-wordnet.json',
                        help='Hierarchy pth filename')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='JSONHIERARCHY.pth',
                        help='Hierarchy pth output file ' +
                             '(will be overwritten)')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load json
    with open(args.json, 'r') as f:
        json_h = json.load(f)

    # Get info from json
    class_list = []
    class_parents = {}
    class_description = {}
    child2parent = {}

    def recurse_json(hdict, parent):
        if len(hdict) == 0:
            return

        for k, v in hdict.items():
            class_list.append(k)
            class_description[k] = v['description']
            child2parent[k] = parent
            if parent is not None:
                parent_id = class_list.index(parent)
                class_parents[k] = class_parents[parent].copy()
                class_parents[k].append(parent_id)
            else:
                class_parents[k] = []
        for k, v in hdict.items():
            recurse_json(v['children'], k)
    recurse_json(json_h, None)

    torch.save({"class_list": class_list,
                "class_description": class_description,
                "child2parent": child2parent,
                "class_parents": class_parents}, args.outfile)


if __name__ == "__main__":
    main()
