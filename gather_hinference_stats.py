import argparse
import numpy as np
import os
import logging

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
# from tqdm.contrib.logging import logging_redirect_tqdm

from lib import models
from lib.hierarchy import Hierarchy
from lib import hierarchy_metrics as hm
from lib import hierarchy_inference as hi
from lib.utils import config_util
from lib.utils import dataset_util

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Gather OOD metrics for checkpoint')
parser.add_argument('-c', '--config_fn', metavar='FILENAME', default=None,
                    help='Protobuf config filename')
parser.add_argument('-tnr', '--tnr-range', nargs=3, type=float,
                    default=[0., 1., 100.],
                    help="TNR Range: start stop steps")
parser.add_argument('-hi', '--hierarchy',
                    default='pruned-wn.pth',
                    help="Hierarchy to compare to with Flat classifiers")

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add stdout handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


def print_stats_of_list(prefix,dat):
    # Helper to print min/max/avg/std/len of values in a list
    dat = np.array(dat)
    logger.info("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
            prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
    )


def get_avg_hdist(hdist_mat):
    hdist_total = 0.
    count = 0.
    for i in range(hdist_mat.shape[0]):
        for j in range(hdist_mat.shape[1]):
            hdist_total += (i+j)*hdist_mat[i,j]
            count += hdist_mat[i,j]
    return hdist_total/count


def calc_tnr_threshstats(stopcriterion, hierarchy,
                         train_logits, train_targets,
                         val_logits, val_ml, val_act,
                         ood_logits, ood_ml, ood_act,
                         tnr_range=[0., 1., 1000]):
    val_acc = []
    val_hdist = []
    val_avg_hdist = []
    ood_acc = []
    ood_hdist = []
    ood_avg_hdist = []
    tnr_range = np.linspace(tnr_range[0], tnr_range[1], tnr_range[2])
    # with logging_redirect_tqdm():
        # for i in trange(len(tnr_range)):
    for i in tqdm(range(len(tnr_range))):
        tnr = tnr_range[i]
        sc = stopcriterion(hierarchy, tnr)
        sc.update(train_logits, train_targets, inp_scores=False)
        sc.gen_threshold()
        val_preds = sc.predict(val_logits)
        ood_preds = sc.predict(ood_logits)
        val_hmet = hm.HierarchicalPredAccuracy(hierarchy, track_hdist=True, is_gt_multilabel=True)
        val_hmet.update_state(torch.tensor(val_preds).long().to(device), (val_ml.to(device), val_act.to(device)))
        ood_hmet = hm.HierarchicalPredAccuracy(hierarchy, track_hdist=True, is_gt_multilabel=True)
        ood_hmet.update_state(torch.tensor(ood_preds).long().to(device), (ood_ml.to(device), ood_act.to(device)))
        val_acc.append(val_hmet.result())
        vhd = val_hmet.result_hierarchy_distances()
        val_hdist.append(vhd)
        val_avg_hdist.append(get_avg_hdist(vhd))
        ood_acc.append(ood_hmet.result())
        ohd = ood_hmet.result_hierarchy_distances()
        ood_hdist.append(ohd)
        ood_avg_hdist.append(get_avg_hdist(ohd))
    return {'val_acc': val_acc,
            'val_hdist': val_hdist,
            'val_avg_hdist': val_avg_hdist,
            'ood_acc': ood_acc,
            'ood_hdist': ood_hdist,
            'ood_avg_hdist': ood_avg_hdist,
            'tnr_range': tnr_range,
           }


def main(args):
    config = config_util.read_config(args.config_fn, for_metrics=True)

    # Add file handler
    fh = logging.FileHandler(config.train_params.log_fn+'hinference', 'w')
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Training Params
    batch_size = config.train_params.batch_size
    hierarchy_fn = config.hierarchy_fn
    checkpoint_fn = config.train_params.checkpoint_fn

    logger.info('==> Preparing data..')
    train_ds, val_ds, ood_ds = dataset_util.gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    # trainloader = torch.utils.data.DataLoader(
    #     val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    oodloader = DataLoader(
        ood_ds, batch_size=batch_size, shuffle=False, num_workers=16)

    backbone = getattr(models, config.backbone)
    # Load Model
    if config.model in [config.CASCADE, config.CASCADEFCHEAD]:
        id_hierarchy = Hierarchy(train_ds.classes, hierarchy_fn)
        ood_hierarchy = Hierarchy(ood_ds.classes, hierarchy_fn)
        if config.model == config.CASCADEFCHEAD:
            net = models.build_softmax_cascade(
                id_hierarchy, backbone=config.backbone)
        else:
            net = backbone(num_classes=id_hierarchy.num_classes)
        # Setup stopping criterion
        stopping_criterions = [
            [hi.PathProbStoppingCriterion, 'Path Prob SC'],
            [hi.SynsetPathProbStoppingCriterion, 'Synset Path Prob SC'],
            # hi.SynsetPathProbStoppingCriterion(id_hierarchy, tnr=0.95)
            # hi.SynsetSoftmaxStoppingCriterion(id_hierarchy,  tnr=0.95)
            # hi.SynsetEntropyStoppingCriterion(id_hierarchy,  tnr=0.95)
        ]
    elif config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
        id_hierarchy = Hierarchy(train_ds.classes, args.hierarchy)
        ood_hierarchy = Hierarchy(ood_ds.classes, args.hierarchy)
        if config.model == config.SOFTMAXFCHEAD:
            net = models.build_softmax_fchead(
                num_id_classes,
                backbone=config.backbone,
                embed_layer=config.embed_layer,
            )
        else:
            net = backbone(num_classes=num_id_classes)
        stopping_criterions = [
            [hi.FlatStoppingCriterion, 'FlatSC'],
        ]
    else:
        raise ValueError("Unsupported model type")

    # Distribution strategy
    if config.distribution_strategy.lower() == 'dataparallel':
        logger.info("Using DataParallel")
        net = torch.nn.parallel.DataParallel(net)
    elif config.distribution_strategy.lower() == 'distributeddataparallel':
        logger.info("Using DistributedDataParallel")
        net = torch.nn.parallel.DistributedDataParallel(net)
    net = net.to(device)

    # Load checkpoint
    net.load_state_dict(torch.load(checkpoint_fn))
    net.eval()


    logger.info("Gather scores...")
    with torch.no_grad():
        train_logits = {
            'logits': torch.empty((0,), device='cpu'),
            'targets': torch.empty((0,), device='cpu', dtype=torch.long),
        }
        val_logits = {
            'logits': torch.empty((0,), device='cpu'),
            'targets': torch.empty((0,), device='cpu', dtype=torch.long),
        }
        ood_logits = {
            'logits': torch.empty((0,), device='cpu'),
            'targets': torch.empty((0,), device='cpu', dtype=torch.long),
        }
        for loader, logits in zip([trainloader, valloader, oodloader],
                                  [train_logits, val_logits, ood_logits]):
            for inputs, targets in tqdm(loader):
                inputs = inputs.to(device)
                outputs = net(inputs)
                logits['logits'] = torch.cat(
                    (logits['logits'], outputs.cpu()), 0)
                logits['targets'] = torch.cat(
                    (logits['targets'], targets.cpu()), 0)


    val_ml, val_act = id_hierarchy.to_multilabel(val_logits['targets'].long())
    full_ood_ml, full_ood_act = ood_hierarchy.to_full_multilabel(ood_logits['targets'].long())
    ood_ml, ood_act = id_hierarchy.trim_full_multilabel(full_ood_ml, full_ood_act)

    tnr_range = args.tnr_range
    tnr_range[-1] = int(tnr_range[-1])
    logger.info("Gather inference stats...")
    allres = {}
    for sc in stopping_criterions:
        res = calc_tnr_threshstats(
            sc[0], id_hierarchy,
            train_logits['logits'], train_logits['targets'],
            val_logits['logits'], val_ml, val_act,
            ood_logits['logits'], ood_ml, ood_act,
            tnr_range=args.tnr_range)
        allres[sc[1]] = res

    if config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
        T=1000.
        for sc in stopping_criterions:
            res = calc_tnr_threshstats(
                sc[0], id_hierarchy,
                train_logits['logits']/T, train_logits['targets'],
                val_logits['logits']/T, val_ml, val_act,
                ood_logits['logits']/T, ood_ml, ood_act,
                tnr_range=args.tnr_range)
            allres[sc[1]+' ODIN'] = res

    # Save metrics
    torch.save(allres,
               os.path.join(os.path.dirname(config.train_params.log_fn),
               'hinference.result')
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
