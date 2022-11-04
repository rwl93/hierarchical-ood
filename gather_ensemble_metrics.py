import argparse
from collections import defaultdict
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
from utils import dataset_util
from utils import model_util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Gather OOD metrics for ensemble')
parser.add_argument('-c', '--config_fn', metavar='FILENAME', default=None,
                    help='Protobuf config filename')
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


def main(args):
    if args.config_fn is not None:
        config = config_util.read_config(args.config_fn, ensemble=True, for_metrics=True)
    else:
        raise ValueError("No config filename provided")

    # Add file handler
    fh = logging.FileHandler(config.train_params.log_fn, 'w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if config.model == config.MOS:
        model = 'mos'
    elif config.model in [config.CASCADE, config.CASCADEFCHEAD]:
        model = 'cascade'
    elif config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
        model = 'softmax'
    else:
        raise ValueError("HILR and ILR are not supported")

    # Training Params
    batch_size = config.train_params.batch_size

    logger.info('==> Preparing data..')
    train_ds, val_ds, ood_ds = dataset_util.gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    valloader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    oodloader = torch.utils.data.DataLoader(
        ood_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    logger.info("# ID Test: {}".format(len(val_ds.imgs)))
    logger.info("# OOD: {}".format(len(ood_ds.imgs)))

    far_ood_dsets = []
    for dset in config.far_ood_dsets:
        ds = dataset_util.gen_far_ood_datasets(dset)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=16)
        far_ood_dsets.append([dset, loader])
        logger.info("# {}: {}".format(dset, len(ds.imgs)))

    hierarchy_fn = config.hierarchy_fn
    if hierarchy_fn.upper() not in ['', 'NONE']:
        hierarchy = hierarchy_util.Hierarchy(train_ds.classes, hierarchy_fn)
    else:
        hierarchy = None

    # Distribution strategy
    if config.distribution_strategy.lower() == 'dataparallel':
        logger.info("Using DataParallel")
        dist_strat = torch.nn.DataParallel
    elif config.distribution_strategy.lower() == 'distributeddataparallel':
        logger.info("Using DistributedDataParallel")
        dist_strat = torch.nn.parallel.DistributedDataParallel
    else:
        logger.info("No distribution strategy")
        dist_strat = lambda inp : inp

    ckpts = config.train_params.checkpoint_fn.split(';')
    assert(len(ckpts) == config.num_models)
    # Loop over each model in ensemble
    net, acc, ood, ood_std = model_util.build_model(
        config, num_id_classes, train_ds.classes)
    _, ens_acc, ens_ood, ens_ood_std = model_util.build_model(
        config, num_id_classes, train_ds.classes)
    net = dist_strat(net)
    net = net.to(device)

    # Setup output storage
    val_logits = torch.empty((0,), device='cpu')
    val_targets = torch.empty((0,), device='cpu')
    ood_logits = torch.empty((0,), device='cpu')
    far_logits = defaultdict(lambda : torch.empty((0,), device='cpu'))

    def reset_state():
        acc.reset_state()
        ood.reset_state()
        if ood_std:
            ood_std.reset_state()

    for model_idx in range(config.num_models):
        reset_state()
        logger.info(f'Model R{model_idx} Results')
        # Load checkpoint
        checkpoint_fn = ckpts[model_idx]
        net.load_state_dict(torch.load(checkpoint_fn))
        net.eval()

        # Gather validation accuracy
        vlogits_tmp = torch.empty((0,), device='cpu')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                vlogits_tmp = torch.cat((vlogits_tmp, outputs.cpu()), 0)
                if model_idx == 0:
                    val_targets = torch.cat((val_targets, targets.cpu()), 0)
                acc.update_state(outputs, targets)
        val_logits = torch.cat((val_logits, vlogits_tmp.expand(1,-1,-1)),0)
        logger.info(f'Model R{model_idx} Accuracy Results')

        ood_res = None
        acc_res = None
        top1_res = None
        pred_res = None
        gw_res = None
        if config.model == config.MOS:
            acc_res = top1_res = acc.result()
            gw_res = acc.result_groupwise()
            logger.info("Accuracy: {}".format(top1_res))
            logger.info("Groupwise Accuracy: {}".format(gw_res))
        elif config.model in [config.CASCADE, config.CASCADEFCHEAD]:
            acc_res = acc.result()
            top1_res = acc.result_top1()
            pred_res = acc.result_pred()
            error_depth_res = acc.result_error_depth()
            logger.info("Mean Synset Accuracy: {}".format(acc_res))
            logger.info("Top-1 Accuracy: {}".format(top1_res))
            logger.info("Pred Accuracy: {}".format(pred_res))
            logger.info("Synset Accuracies:")
            logger.info("{}".format(acc.result_full()))
            logger.info("Error Depth:")
            for i, ed in enumerate(error_depth_res):
                logger.info("\tNum. Errors at Depth {} = {}".format(
                    i if i != len(error_depth_res)-1 else -1, ed))
        elif config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
            acc_res = acc.result()
            top1_res = acc_res[0]
            pred_res = acc_res[0]
            logger.info("Top-1 Accuracy: {}".format(acc_res[0]))
            logger.info("Top-5 Accuracy: {}".format(acc_res[1]))
        else:
            raise ValueError("HILR and ILR are not supported")

        # Gather OOD metrics
        logger.info("OOD Results")
        with torch.no_grad():
            logits_tmp = torch.empty((0,), device='cpu')
            for batch_idx, (inputs, _) in enumerate(oodloader):
                inputs = inputs.to(device)
                outputs = net(inputs)
                logits_tmp = torch.cat((logits_tmp, outputs.cpu()), 0)
            ood_logits = torch.cat((ood_logits, logits_tmp.expand(1,-1,-1)),0)

            for dset, loader in far_ood_dsets:
                logits_tmp = torch.empty((0,), device='cpu')
                for batch_idx, (inputs, _) in enumerate(loader):
                    inputs = inputs.to(device)
                    outputs = net(inputs)
                    logits_tmp = torch.cat((logits_tmp, outputs.cpu()), 0)
                far_logits[dset] = torch.cat((far_logits[dset], logits_tmp.expand(1,-1,-1)),0)

        # Skipping to reduce runtime
        # ood.update_state(net, valloader, oodloader)
        # for dset, loader in far_ood_dsets:
        #     ood.update_state(net, None, loader, dset)
        # if config.model in [config.CASCADE, config.HILR, config.CASCADEFCHEAD]:
        #     ood.print_result_full()
        #     ood_std.update_state(net, valloader, oodloader)
        #     for dset, loader in far_ood_dsets:
        #         ood_std.update_state(net, None, loader, dset)
        #     ood_std.print_result()
        # else:
        #     ood.print_result()

    ens_acc = hm.calc_ensemble_accuracy(
        val_logits, val_targets, model=model, hierarchy=hierarchy)
    logger.info(f'Ensemble: Model: {model}, M: {config.num_models}')
    logger.info(f'Ensemble Accuracy: {ens_acc}')
    ens_ood = hm.calc_ensemble_ood(
        val_logits, ood_logits, model=model, hierarchy=hierarchy)
    for k, v in ens_ood.items():
        for met in ['AUROC', 'TNR', 'AUOUT']:
            logger.info(f'Ensemble {k} {met}: {v["TMP"][met]}')

    far_ens_ood_all = []
    far_dsets = []
    for dset,_ in far_ood_dsets:
        logger.info(50*'-')
        logger.info(f'Ensemble: {dset}')
        logger.info(50*'-')
        far_ens_ood = hm.calc_ensemble_ood(
            val_logits, far_logits[dset], model=model, hierarchy=hierarchy)
        for k, v in far_ens_ood.items():
            for met in ['AUROC', 'TNR', 'AUOUT']:
                logger.info(f'Ensemble {k} {met}: {v["TMP"][met]}')
        far_ens_ood_all.append(far_ens_ood)
        far_dsets.append(dset)

    # Save metrics
    torch.save({
        'ood': ens_ood,
        'acc': ens_acc,
        'far ood': far_ens_ood_all,
        'far dsets': far_dsets,
        },
        os.path.join(os.path.dirname(config.train_params.log_fn), 'exp.result')
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
