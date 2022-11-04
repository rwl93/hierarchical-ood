import argparse
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utilsdata
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models
import train_util
import ood_helpers
import calculate_log as callog
import hierarchy_util
import hierarchy_loss
import hierarchy_metrics as hm
from utils import config_util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Gather OOD metrics for checkpoint')
parser.add_argument('-c', '--config_fn', metavar='FILENAME', default=None,
                    help='Protobuf config filename')
parser.add_argument('-d', '--data', metavar='DIR', default='./data/fine/',
                    help='path to the dataset')
parser.add_argument('-m', '--model', metavar='MODEL', default='softmax',
                    choices=['softmax', 'ilr', 'cascade', 'hilr',
                             'cascadefchead'],
                    help='Model to train')
parser.add_argument('-ckpt', '--checkpoint', metavar='FILENAME',
                    help='Checkpoint filename')
parser.add_argument('-hfn', '--hierarchy', metavar='FILENAME',
                    default='pruned-wn.pth',
                    help='Hierarchy to use for training')
parser.add_argument('-spl', '--softpredceloss', action="store_true",
                    help='Use soft prediction based predictions')
parser.add_argument('-o', '--output', metavar='FILENAME',
                    default='output.pth',
                    help='The output file to store the results')
parser.add_argument('-gpu', '--gpu-devices', metavar='INT',
                    default=None,
                    help='GPU Device IDs')
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add stdout handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


def gen_datasets(datadir):
    """Generate datasets for experiment.

    Preprocessing from pytorch Imagenet example code

    Parameters
    ----------
    datadir : string
        path to directory of data

    Returns
    -------
    Tuple of torchvision.datasets.Dataset objects:
        val_dataset, ood_dataset
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        os.path.join(datadir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = datasets.ImageFolder(os.path.join(datadir, 'val'),
                                       eval_transform)
    ood_dataset = datasets.ImageFolder(os.path.join(datadir, 'ood'),
                                       eval_transform)
    return train_dataset, val_dataset, ood_dataset


def gen_far_ood_datasets(dset: str = "iNaturalist"):
    if dset not in ['iNaturalist', 'SUN', 'Places', 'Textures',
                    'coarseid-fineood', 'coarseid-coarseood',
                    'imagenet1000-fineood', 'imagenet1000-mediumood',
                    'imagenet1000-coarseood',
                    ]:
        raise ValueError("Unknown far ood dataset: " + dset)
    datadir = "data/" + dset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.ImageFolder(datadir, transform)
    return ds


def print_stats_of_list(prefix,dat):
    # Helper to print min/max/avg/std/len of values in a list
    dat = np.array(dat)
    logger.info("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
            prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
    )


def main(args):
    # if args.gpu_devices is not None:
    #     # Set GPU
    #     os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    # If config file specified, read from it
    if args.config_fn is not None:
        config = config_util.read_config(args.config_fn, for_metrics=True)
    else:
        logger.warning(
            "Passing hyperparameters via argparse is deprecated and " +
            "may not function properly. Prefer to use protobuf configs.")
        config = config_util.build_config_from_args(args, for_metrics=True)

    # Training Params
    batch_size = config.train_params.batch_size
    hierarchy_fn = config.hierarchy_fn
    checkpoint_fn = config.train_params.checkpoint_fn

    logger.info('==> Preparing data..')
    train_ds, val_ds, ood_ds = gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    valloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    oodloader = torch.utils.data.DataLoader(
        ood_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info("# ID Test: {}".format(len(val_ds.imgs)))
    logger.info("# OOD: {}".format(len(ood_ds.imgs)))

    ood_dsets = [['ID', valloader],
                 ['OOD', oodloader],]
    for dset in config.far_ood_dsets:
        ds = gen_far_ood_datasets(dset)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=16)
        ood_dsets.append([dset, loader])
        logger.info("# {}: {}".format(dset, len(ds.imgs)))

    kwargs = {}
    if config.HasField('model_config'):
        mc = getattr(config, config.WhichOneof("model_config"))
        if hasattr(mc, 'fc_head_sizes'):
            kwargs['head_layer_sizes'] = mc.fc_head_sizes
        if hasattr(mc, 'split_fchead_layers'):
            kwargs['split_fchead_layers'] = mc.split_fchead_layers
    backbone = getattr(models, config.backbone)
    # Load Model
    if config.model in [config.SOFTMAX, config.ILR, config.SOFTMAXFCHEAD]:
        if config.model == config.SOFTMAXFCHEAD:
            net = models.build_softmax_fchead(num_id_classes,
                                              backbone=config.backbone,
                                              **kwargs,
                                              )
        else:
            net = backbone(num_classes=num_id_classes)
        id_hierarchy = None
    elif config.model in [config.CASCADE, config.HILR, config.CASCADEFCHEAD]:
        id_hierarchy = hierarchy_util.Hierarchy(train_ds.classes, hierarchy_fn)
        print(kwargs)
        if config.model == config.CASCADEFCHEAD:
            net = models.build_softmax_cascade(
                id_hierarchy, backbone=config.backbone, **kwargs)
            print(net.head)
        else:
            net = backbone(num_classes=id_hierarchy.num_classes)
    elif config.model == config.MOS:
        id_hierarchy = hierarchy_util.Hierarchy(train_ds.classes, hierarchy_fn)
        net = models.build_MOS(
            id_hierarchy, backbone=config.backbone, **kwargs)

    # Distribution strategy
    if config.distribution_strategy.lower() == 'dataparallel':
        logger.info("Using DataParallel")
        net = torch.nn.DataParallel(net)
    elif config.distribution_strategy.lower() == 'distributeddataparallel':
        logger.info("Using DistributedDataParallel")
        net = torch.nn.parallel.DistributedDataParallel(net)
    net = net.to(device)

    # Load checkpoint
    net.load_state_dict(torch.load(checkpoint_fn))
    net.eval()

    # import pdb; pdb.set_trace()
    dataset_results = {'config_fn': args.config_fn}
    logger.info("Generating Results")
    with torch.no_grad():
        for dset, loader in [['train', trainloader],
                             ['val', valloader],
                             ['ood', oodloader]]:
        # for dset, loader in ood_dsets:
            logger.info("Working on " + dset + "...")
            logits = torch.empty((0,), device='cpu')
            targets = torch.empty((0,), dtype=torch.long, device='cpu')
            # pp = np.empty((0,))
            # Hmean = np.empty((0,))
            # Hmin = np.empty((0,))
            # Hpath = []
            for batch_idx, (inputs, targs) in enumerate(loader):
                #inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.to(device)
                outputs = net(inputs)
                logits = torch.cat((logits, outputs.detach().cpu()), 0)
                targets = torch.cat((targets, targs.long()), 0)
                # hlogits = id_hierarchy.split_logits_by_synset(logits)
                # scores = hm.soft_predict(
                #      hlogits, id_hierarchy, model='cascade',  #FIXME: Don't hardcode this
                #      return_pathentropy=True)
                # pathprob, _, _, entropys, entropy_stats, _, _ = scores
                # # Get Pred Path Prob
                # pp = np.concatenate((pp, pathprob), axis=0)
                # # Get Mean Path Ent
                # Hmean = np.concatenate((Hmean, entropy_stats[0]), axis=0)
                # # Get Min Path Ent
                # Hmin = np.concatenate((Hmin, entropy_stats[1]), axis=0)
                # # Get all path ent
                # Hpath.extend(entropys)
            #dataset_results[dset] = {
            #    'pathprob': pp,
            #    'meanent': Hmean,
            #    'minent': Hmin,
            #    'entlist': Hpath,
            #}
            torch.save({'logits':logits, 'targets': targets},
                       args.output+dset+'logits.out')

    # Save metrics
    """
    Save:
        {
            Config FN: ""?,
            DATASET: {
                - Predicted Path Probability
                - Mean Path Entropy
                - Min Path Entropy
                - [Path Entropies]
            }
        }
    """
    #torch.save(
    #    dataset_results,
    #    os.path.join(os.path.dirname(config.train_params.log_fn),
    #                 'dset.scores')
    #)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
