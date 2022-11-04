import argparse
import logging

import torch
from torch.utils.data import DataLoader

from lib import models
from lib.hierarchy import Hierarchy
from lib.utils import config_util
from lib.utils.dataset_util import gen_datasets, gen_far_ood_datasets

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


def main(args):
    config = config_util.read_config(args.config_fn, for_metrics=True)

    # Training Params
    batch_size = config.train_params.batch_size
    hierarchy_fn = config.hierarchy_fn
    checkpoint_fn = config.train_params.checkpoint_fn

    logger.info('==> Preparing data..')
    train_ds, val_ds, ood_ds = gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=16)
    oodloader = DataLoader(
        ood_ds, batch_size=batch_size, shuffle=False, num_workers=16)

    ood_dsets = [['ID', valloader],
                 ['OOD', oodloader],]
    for dset in config.far_ood_dsets:
        ds = gen_far_ood_datasets(dset)
        loader = DataLoader(
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
        id_hierarchy = Hierarchy(train_ds.classes, 'hierarchies/' + hierarchy_fn)
        print(kwargs)
        if config.model == config.CASCADEFCHEAD:
            net = models.build_softmax_cascade(
                id_hierarchy, backbone=config.backbone, **kwargs)
            print(net.head)
        else:
            net = backbone(num_classes=id_hierarchy.num_classes)
    elif config.model == config.MOS:
        id_hierarchy = Hierarchy(train_ds.classes, 'hierarchies/' + hierarchy_fn)
        net = models.build_MOS(
            id_hierarchy, backbone=config.backbone, **kwargs)
    else:
        raise ValueError(f'Unsupported model: {config.model}')

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

    logger.info("Generating Results")
    with torch.no_grad():
        for dset, loader in [['train', trainloader],
                             ['val', valloader],
                             ['ood', oodloader]]:
        # for dset, loader in ood_dsets:
            logger.info("Working on " + dset + "...")
            logits = torch.empty((0,), device='cpu')
            targets = torch.empty((0,), dtype=torch.long, device='cpu')
            for inputs, targs in loader:
                inputs = inputs.to(device)
                outputs = net(inputs)
                logits = torch.cat((logits, outputs.detach().cpu()), 0)
                targets = torch.cat((targets, targs.long()), 0)
            torch.save({'logits':logits, 'targets': targets},
                       args.output+dset+'logits.out')

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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
