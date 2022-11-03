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
import sklearn

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Gather Mahalanobis OOD metrics')
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
        config = config_util.read_config(args.config_fn, ensemble=False, for_metrics=True)
    else:
        raise ValueError("No config filename provided")

    # Add file handler
    fh = logging.FileHandler(config.train_params.log_fn+'mahala', 'w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
        model = 'softmax'
    else:
        raise ValueError("Mahalanobis only supports softmax")

    # Training Params
    batch_size = config.train_params.batch_size

    logger.info('==> Preparing data..')
    train_ds, val_ds, ood_ds = dataset_util.gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    trainloader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=16)
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


    logger.info("Build Model")
    net, _, _, _ = model_util.build_model(config, num_id_classes, train_ds.classes)
    net = dist_strat(net)
    net = net.to(device)
    logger.info("Model moved to device")

    # Load checkpoint
    checkpoint_fn = config.train_params.checkpoint_fn
    net.load_state_dict(torch.load(checkpoint_fn))
    net.eval()

    # Only use penultimate layer features
    logger.info("Getting penultimate layer feats")
    fe = torch.nn.Sequential(*list(net.children())[:-1])

    ### Calculate means and covariance stats for Mahalanobis detector
    logger.info("Computing means and precisions")
    class_means, class_precisions, bkgd_mean, bkgd_precision = \
        compute_empirical_means_and_precision(fe, trainloader, num_id_classes)

    logger.info("Computing val and ood scores")
    ### Measure OOD scores with state-of-the-art methods
    mahala_scores_dict = {}
    generalized_mahala_scores_dict = {}

    # Calculate scores
    val_md, val_rmd = generate_mahalanobis_scores(
            fe, valloader, class_means, class_precisions, bkgd_mean, bkgd_precision)
    ood_md, ood_rmd = generate_mahalanobis_scores(
            fe, oodloader, class_means, class_precisions, bkgd_mean, bkgd_precision)
    logger.info("OOD Results")

    metric_results = {}
    metric_results["Mahalanobis"] = callog.metric(val_md, ood_md)
    metric_results["Relative Mahalanobis"] = callog.metric(val_rmd, ood_rmd)

    for k, v in metric_results.items():
        for met in ['AUROC', 'TNR', 'AUOUT']:
            logger.info(f'{k} {met}: {v["TMP"][met]}')

    far_ood_md = []
    far_ood_rmd = []
    for dset, loader in far_ood_dsets:
        md, rmd = generate_mahalanobis_scores(
                fe, loader, class_means, class_precisions, bkgd_mean, bkgd_precision)
        far_ood_md.append(md)
        far_ood_rmd.append(rmd)

    far_metric_results = {"Mahalanobis": [], "Relative Mahalanobis": []}
    far_dsets = []
    for i, (dset, _) in enumerate(far_ood_dsets):
        far_dsets.append(dset)
        far_metric_results["Mahalanobis"].append(callog.metric(val_md, far_ood_md[i]))
        far_metric_results["Relative Mahalanobis"].append(callog.metric(val_rmd, far_ood_rmd[i]))

        logger.info(f'Dataset name: {dset}')
        for k, v in far_metric_results.items():
            for met in ['AUROC', 'TNR', 'AUOUT']:
                logger.info(f'{k} {met}: {v[i]["TMP"][met]}')

    # Save metrics
    torch.save({
        'metric_results': metric_results,
        'far_metric_results': far_metric_results,
        'far_dsets': far_dsets,
        },
        os.path.join(os.path.dirname(config.train_params.log_fn), 'exp.resultmahala')
    )


def compute_empirical_means_and_precision(fe, loader, num_classes):
    # Calculate means and covariance stats for Mahalanobis detector
    with torch.no_grad():
        print("Computing sample matrix...")
        act = torch.empty((0,), dtype=torch.float, device='cpu')
        lbls = torch.empty((0,), dtype=torch.float, device='cpu')
        for x,y in tqdm(loader, desc='Batches'):
            x = x.to(device); y = y.to(device)
            # Forward pass data through model to prime the activations dictionary
            feats = fe(x).squeeze()
            # Append this representation onto layer_reps
            act = torch.cat((act, feats.data.cpu()), 0)
            lbls = torch.cat((lbls, y.cpu()), 0)
        torch.save({'act':act, 'lbls':lbls, 'config_fn': args.config_fn},
            'mahala_act.tempout')

        print("Done with feats")

        # Calculate background stats
        print("Computing background stats...")
        background_mean = act.mean(0)
        # background_diff = act - background_mean # replaced by inplace op
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit((act-background_mean).cpu().numpy().astype(np.float32))
        background_precision = torch.from_numpy(group_lasso.precision_).float().to(device)

        # Calculate class-wise stats
        print("Computing class-wise stats...")
        class_means = []
        class_precisions = []
        centered_data = torch.empty((0,), device='cpu')
        for cls in range(num_classes):
            cls_idxs = torch.where(lbls == cls)
            mu = act[cls_idxs].mean(0)
            class_means.append(mu)
            # mudiff = act[cls_idxs] - mu # replaced by inplace op
            centered_data = torch.cat((centered_data, (act[cls_idxs] - mu).data), 0)
        print("Remove act and lbls")
        del act # no longer needed
        del lbls # no longer needed
        print("Group Lasso")
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
        del centered_data # no longer needed
        precision_matrix = torch.from_numpy(group_lasso.precision_).float().to(device)

    return class_means, precision_matrix, background_mean, background_precision


def generate_mahalanobis_scores(
        base_model, loader, means, precision,
        bkgd_mean, bkgd_precision,
        ):
    mahalanobis_scores = np.empty((0,))
    M0_scores = np.empty((0,))
    with torch.no_grad():
        for pkg in loader:
            dat = pkg[0]
            dat = dat.to(device) #; lbl = lbl.to(device)
            # Initial forward pass & populate activation_dictionary
            feats = base_model(dat).squeeze()
            # Compute mahalanobis scores over each layer and sum
            scores = torch.zeros((feats.shape[0])).to(device)

            num_classes = len(means)
            class_scores = torch.zeros((feats.shape[0],num_classes)).float().to(device)
            for cls in range(num_classes):
                zc_tensor = feats - means[cls].to(device)
                Mx_PT = -0.5*torch.matmul(torch.matmul(zc_tensor,
                    precision.to(device)), zc_tensor.t()).diag()
                class_scores[:,cls] = Mx_PT
            class_scores = torch.max(class_scores,dim=1)[0]
            mahalanobis_scores = np.concatenate((mahalanobis_scores, class_scores.cpu().numpy()), 0)
            back_mudiff = feats - bkgd_mean.to(device)
            M0 = -0.5*torch.matmul(torch.matmul(back_mudiff,
                bkgd_precision.to(device)), back_mudiff.t()).diag()
            M0_scores = np.concatenate((M0_scores, M0.cpu().numpy()), 0)
    return mahalanobis_scores, mahalanobis_scores-M0_scores


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
