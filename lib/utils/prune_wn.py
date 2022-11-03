import argparse
import copy
import hierarchy_util
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
from tqdm import tqdm


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Add stdout handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelfile', metavar="FILENAME",
                        default='data/wnids1k.csv',
                        help='Leaf node Wordnet ID label file')
    parser.add_argument('--datadir', metavar="PATH",
                        default='/home/public/ImageNet/train/',
                        help='Data directory with images sorted by Wordnet ID')
    parser.add_argument('--hierarchy', metavar="FILENAME",
                        default='imagenet1000-wn.pth',
                        help='Hierarchy to prune')
    parser.add_argument('--numsynsets', metavar="INT", type=int,
                        default=100,
                        help='Number of synsets to prune the hierarchy to')
    parser.add_argument('--outfile', metavar="FILENAME",
                        default='PRUNEDWNOUTPUT.pth',
                        help='Hierarchy pth output file ' +
                             '(will be overwritten)')
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="increase output verbosity")
    parser.add_argument("-lf", "--logfile",
                        default=None,
                        help="File to write log to")
    parser.add_argument("--gather-stats", action='store_true',
                        help="Gather hierarchy train stats and output json")
    parser.add_argument("--gather-stats-only", action='store_true',
                        help="Gather stats, output json, and exit")
    parser.add_argument("--stats-outfile", metavar="FILENAME",
                        default="GATHERSTATS.json",
                        help="File to output hierarchy stats")
    return parser


## Top down gather parents recursive helper
def gen_hierarchy(classes, class_info, H,
                  class_list, class_description,
                  child2parent, class_parents):
    for wnid in classes:
        class_list.append(wnid)
        class_description[wnid] = H.class_description[wnid]
    for wnid in classes:
        mask = class_info['wnid'] == wnid
        parent = class_info['parent'][mask].item()
        child2parent[wnid] = parent
        if parent is None:
            class_parents[wnid] = list()
        else:
            pars = []
            pars.extend(copy.copy(class_parents[parent]))
            pars.append(class_list.index(parent))
            class_parents[wnid] = pars
        cwnids = class_info['wnid'][class_info['parent'] == wnid]
        gen_hierarchy(cwnids, class_info, H, class_list, class_description,
                      child2parent, class_parents)


def main():
##
    parser = get_parser()
    args = parser.parse_args()

    if args.verbose:
        ch.setLevel(logging.DEBUG)

    # Add file handler
    if args.logfile is not None:
        fh = logging.FileHandler(args.logfile, 'w')
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    logger.info("Reading classes...")
    # Get class list
    class_info = pd.read_csv(args.labelfile, header=None)
    class_info.columns = ['wnid']
    # Remove tab character and get class counts
    prefix = args.datadir
    counts = []
    for i, wnid in enumerate(class_info['wnid']):
        if wnid[-1] == '\t':
            wnid = wnid[:-1]
        class_info['wnid'].iloc[i] = wnid

        wnid_imgs = os.listdir(os.path.join(prefix, wnid))
        count = 0
        for img in wnid_imgs:
            if 'JPEG' in img.upper():
                count += 1
        counts.append(count)
    class_info['counts'] = counts
    N_total = sum(counts)

    # Load WN hierarchy
    logger.info("Loading original hierarchy...")
    H = hierarchy_util.Hierarchy(class_info['wnid'].to_list(), args.hierarchy)
    # Add internal nodes to class_info
    logger.info("Populating class list stats...")
    # Get children for all classes (O(N^2))
    class_children = {}
    for i, wnid in enumerate(H.class_list):
        children = []
        for k, v in H.child2parent.items():
            if wnid == v:
                children.append(k)
        class_children[wnid] = children
    # Calc avg children len
    counts = 0.
    N = 0.
    min_children = 100000
    max_children = -1
    for k, c in class_children.items():
        max_children = len(c) if len(c) > max_children else max_children
        if len(c) > 0:
            counts += len(c)
            N += 1
            min_children = len(c) if len(c) < min_children else min_children

    if args.verbose:
        logger.info('Original hierarchy stats')
        logger.info(f'Avg # Children {counts/N}')
        logger.info(f'Min # Children {min_children}')
        logger.info(f'Max # Children {max_children}')

    parent_todo = copy.copy(H.class_list)
    while len(parent_todo) > 0:
        curr_parent = parent_todo.pop(-1)
        if len(class_children[curr_parent]) == 0:
            if args.verbose:
                logger.info(f'Removing {curr_parent}')
            continue
        all_children_done = True
        example_count = 0
        for child in class_children[curr_parent]:
            if child not in class_info['wnid'].to_list():
                all_children_done = False
                break
            example_count += class_info.loc[class_info['wnid'] == child,
                                            'counts'].item()
        if all_children_done:
            if args.verbose:
                logger.info(f'adding to class_info wnid: {curr_parent},'
                            + 'counts: {example_count}')
            cinfolen = len(class_info)
            class_info = class_info.append(
                {'wnid': curr_parent,
                 'counts': example_count},
                ignore_index=True)
            if cinfolen == len(class_info):
                break
        else:
            if args.verbose:
                logger.debug(f'len todolist {len(parent_todo)}')
                logger.debug(f'readding to todo {curr_parent}')
            parent_todo.insert(0, curr_parent)
    class_info['probability'] = class_info['counts'] / N_total
    class_info['info content'] = np.log2(1./class_info['probability'])
    # Add parent info to each class and Order into synsets
    synsetids = []
    class_parents = []
    for wnid in class_info['wnid']:
        idx = H.class_list.index(wnid)
        synset_id, _ = H.classid2synsetid_offset(idx)
        synsetids.append(int(synset_id))
        class_parents.append(H.child2parent[wnid])
    class_info['synset id'] = synsetids
    class_info['parent'] = class_parents

    if args.gather_stats or args.gather_stats_only:
        entropys = []
        def recurse_hierarchy(wnid, depth):
            info = class_info.loc[class_info.wnid == wnid]
            entropys.append(info.probability * info['info content'])
            temp = {
                'description': H.hierarchy['class_description'][wnid],
                'parent': H.hierarchy['child2parent'][wnid],
                'ancestors': [H.hierarchy['class_list'][i]
                              for i in H.hierarchy['class_parents'][wnid]],
                'children': {i: recurse_hierarchy(i, depth+1)
                             for i in class_children[wnid]},
                'counts': int(info.counts),
                'probability': float(info.probability),
                'info content': float(info['info content']),
                'synset id': int(info['synset id']),
                'depth': depth,
            }
            return temp
        for_json = {wnid: recurse_hierarchy(wnid, 0)
                    for wnid in class_info[class_info.parent.values == None].wnid}
        entropys = np.array(entropys)
        for_json['Entropy (sum)'] = entropys.sum()
        for_json['Entropy (mean)'] = entropys.mean()
        logger.info(f'Writing hierarchy training dataset stats to {args.stats_outfile}')
        with open(args.stats_outfile, 'w') as f:
            json.dump(for_json, f, indent=4)
        if args.gather_stats_only:
            sys.exit('Completed gathering stats and exiting')

    # Remove min entropy idx
    logger.info("Pruning Synsets")
    pbar = tqdm(total=len(class_info['synset id'].unique())-args.numsynsets)
    while len(class_info['synset id'].unique()) > args.numsynsets:
        # Remove top level nodes
        class_info_indexed = class_info.set_index(['synset id', 'wnid'])
        has_parent = class_info['parent'].values != None
        nonroot_synsets = class_info[has_parent]['synset id'].unique()
        synset_entropy = []
        synset_hasparent = []
        for grp in class_info['synset id'].unique():
            synset_hasparent.append(grp in nonroot_synsets)
            synset_entropy.append(
                (class_info_indexed.loc[grp]['probability'] *
                class_info_indexed.loc[grp]['info content']).sum())
        synset_entropy = pd.DataFrame({'entropy': synset_entropy,
                                       'has parent':synset_hasparent},
                                       index=class_info['synset id'].unique(),)
        mask = synset_entropy['has parent']
        minent_synset = synset_entropy[mask]['entropy'].argmin()
        minent_synset = synset_entropy[mask].index[minent_synset]
        # Remove parent from list
        minent_info = class_info_indexed.loc[minent_synset]
        class_info_indexed = class_info_indexed.drop(minent_synset)
        parent_wnid = minent_info['parent'][0]
        parent_synsetid = class_info[class_info['wnid'] == parent_wnid
                                    ]['synset id'].item()
        grandparent_wnid = class_info[class_info['wnid'] == parent_wnid
                                     ]['parent'].item()
        # Drop parent
        class_info_indexed = class_info_indexed.drop(
            (parent_synsetid, parent_wnid))
        # Add back in children
        minent_info.loc[:, 'parent'] = grandparent_wnid
        minent_info['synset id'] = parent_synsetid
        minent_info = minent_info.reset_index()
        minent_info = minent_info.set_index(['synset id', 'wnid'])
        class_info_indexed = class_info_indexed.append(minent_info)
        # Update class info
        class_info = class_info_indexed.reset_index()
        pbar.update()
    pbar.close()

    # Create hierarchy from class_info
    class_list = list()
    class_description = {}
    class_parents = {}
    child2parent = {}
    toplvl_wnids = class_info['wnid'][class_info['parent'].values == None]
    gen_hierarchy(toplvl_wnids, class_info, H, class_list, class_description,
                      child2parent, class_parents)

    # save hiearchy
    torch.save({'class_list': class_list,
                'class_description': class_description,
                'class_parents': class_parents,
                'child2parent': child2parent},
               args.outfile)


if __name__ == '__main__':
    main()
