import argparse
import logging
import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Custom
from lib import models
from lib import train_util
from lib.hierarchy import Hierarchy
from lib import hierarchy_loss
from lib.utils import config_util
from lib.utils.dataset_util import gen_datasets


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser(description='Hierarchical OOD experiment \
                                 runner harness')
parser.add_argument('-c', '--config_fn', metavar='FILENAME',
                    help='Protobuf config filename')


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add stdout handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


def setup_filehandler(config):
    """Create filehandler for logger."""
    # Add file handler
    file_handler = logging.FileHandler(config.train_params.log_fn, 'w')
    file_handler.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(fh_formatter)
    logger.addHandler(file_handler)


def main(args):
    """Train network."""
    config = config_util.read_config(args.config_fn)
    setup_filehandler(config)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    feat_extractor_config = None
    if config.finetune_from_ckpt:
        logger.info("Reading Feature Extractor Config")
        feat_extractor_config = config_util.read_config(
            config.finetune_from_ckpt)
        # Get backbone
        if feat_extractor_config.backbone != config.backbone:
            raise ValueError("Feature extractor and model must have the same" +
                             " backbone")

    backbone = getattr(models, config.backbone)

    logger.info('==> Preparing data..')
    train_ds, val_ds, _ = gen_datasets(config.data_dir)
    num_id_classes = len(train_ds.classes)

    trainloader = DataLoader(
        train_ds, batch_size=config.train_params.batch_size,
        shuffle=True, num_workers=16)
    valloader = DataLoader(
        val_ds, batch_size=config.train_params.batch_size,
        shuffle=False, num_workers=16)

    if config.no_save:
        config.train_params.checkpoint_fn = None
    logger.info('checkpoint filename: %s', config.train_params.checkpoint_fn)
    logger.info('log filename: %s', config.train_params.log_fn)

    # Load Model
    kwargs = {}
    if config.HasField('model_config'):
        mc = getattr(config, config.WhichOneof("model_config"))
        if hasattr(mc, 'fc_head_sizes'):
            kwargs['head_layer_sizes'] = mc.fc_head_sizes
        if hasattr(mc, 'split_fchead_layers'):
            kwargs['split_fchead_layers'] = mc.split_fchead_layers
    kwargs['freeze_bb'] = config.train_params.freeze_bb
    kwargs['freeze_bb_bn'] = config.train_params.freeze_bb_bn
    kwargs['bb_pretrained'] = config.train_params.bb_pretrained
    if config.model in [config.SOFTMAX, config.SOFTMAXFCHEAD]:
        if config.model == config.SOFTMAXFCHEAD:
            net = models.build_softmax_fchead(
                num_classes=num_id_classes,
                backbone=config.backbone,
                embed_layer=config.embed_layer,
                **kwargs,
            )
            model_type = 'SOFTMAXFCHEAD'
        else:
            net = backbone(num_classes=num_id_classes)
            model_type = 'SOFTMAX'
        criterion = nn.CrossEntropyLoss()
        hierarchy = None
    elif config.model in [config.CASCADE, config.CASCADEFCHEAD]:
        hierarchy = Hierarchy(train_ds.classes,
                              './hierarchies/' + config.hierarchy_fn,
                              config.min_norm_factor,
                              )
        if config.model == config.CASCADE:
            net = backbone(num_classes=hierarchy.num_classes)
            model_type = 'CASCADE'
        elif config.model == config.CASCADEFCHEAD:
            net = models.build_softmax_cascade(
                hierarchy, backbone=config.backbone,
                embed_layer=config.embed_layer,
                **kwargs,)
            model_type = 'CASCADEFCHEAD'
            print(net.head)
        else:
            raise ValueError(f'Unsupported model type: {config.model}')
        if config.hl.weight_ce:
            logger.info("Generating CE Weights...")
            hierarchy.gen_CEweights(trainloader)
            logger.info("Finished generating weights")
        criterion = hierarchy_loss.HierarchicalSoftProbAndSynsetCELoss(
            hierarchy,
            config.train_params.epochs,
            synset_weight_range=[config.hl.synsetce_range.start,
                                 config.hl.synsetce_range.end],
            softprob_weight_range=[config.hl.softpred_range.start,
                                   config.hl.softpred_range.end],
            outlier_weight_range=[config.hl.outlier_exposure_range.start,
                                  config.hl.outlier_exposure_range.end],
            focal_loss=config.hl.focal_loss,
            depth_weight=config.hl.depth_weight,
        )
    elif config.model == config.MOS:
        hierarchy = Hierarchy(train_ds.classes,
                              './hierarchies/' + config.hierarchy_fn,
                              config.min_norm_factor,
                              )
        net = models.build_MOS(hierarchy, config.backbone,
                               **kwargs)
        criterion = hierarchy_loss.MOSLoss(hierarchy)
        model_type = 'MOS'
    else:
        raise ValueError(f'Unsupported model type: {config.model}')

    if config.resume_from_ckpt:
        logger.info("Loading checkpoint to continue training")
        checkpoint_fn = config.train_params.checkpoint_fn
        state_dict = torch.load(checkpoint_fn)
        prefix = 'module.'
        prefix_len = len(prefix)
        updated_state_dict = {}
        for k, v in state_dict.items():
            if prefix in k:
                updated_state_dict[k[prefix_len:]] = v
            else:
                updated_state_dict[k] = v
        net.load_state_dict(updated_state_dict)
    elif config.finetune_from_ckpt:
        if feat_extractor_config is None:
            raise ValueError('Feature extractor config not found')
        logger.info("Loading weights from feature extractor")
        checkpoint_fn = feat_extractor_config.train_params.checkpoint_fn
        state_dict = torch.load(checkpoint_fn)
        # Remove non-backbone keys
        if feat_extractor_config.model in [config.SOFTMAX, config.CASCADE]:
            prefix = 'module.'
            prefix_len = len(prefix)
            updated_state_dict = {}
            for k, v in state_dict.items():
                if prefix in k:
                    updated_state_dict[k[prefix_len:]] = v
                else:
                    updated_state_dict[k] = v
            state_dict = {k: v
                          for k, v in updated_state_dict.items()
                          if 'fc' not in k}
        else:
            prefix = 'backbone.'
            prefix_len = len(prefix)
            state_dict = {k[prefix_len:]: v
                          for k, v in state_dict.items()
                          if prefix in k}
        # Add in necessary keys
        if config.model in [config.SOFTMAX, config.CASCADE]:
            state_dict['fc.weight'] = net.fc._parameters['weight']
            state_dict['fc.bias'] = net.fc._parameters['bias']
            # remove extras from resnet backbone keys
            _ = state_dict.pop('freeze_bb', None)
            _ = state_dict.pop('freeze_bb_bn', None)
            _ = state_dict.pop('bb_pretrained', None)
            net.load_state_dict(state_dict)
            if kwargs['freeze_bb']:
                for p in net.parameters():
                    p.requires_grad = False
                net.fc._parameters['weight'].requires_grad = True
                net.fc._parameters['bias'].requires_grad = True
        else:
            state_dict['freeze_bb'] = net.backbone.freeze_bb
            state_dict['freeze_bb_bn'] = net.backbone.freeze_bb_bn
            state_dict['bb_pretrained'] = net.backbone.bb_pretrained
            net.backbone.load_state_dict(state_dict)
            if kwargs['freeze_bb']:
                for p in net.backbone.parameters():
                    p.requires_grad = False

    if config.distribution_strategy.lower() == 'dataparallel':
        logger.info("DataParallel on %d devices", torch.cuda.device_count())
        net = torch.nn.DataParallel(net)
    elif config.distribution_strategy.lower() == 'distributeddataparallel':
        logger.info("DistributedDataParallel on %d devices", torch.cuda.device_count())
        net = torch.nn.parallel.DistributedDataParallel(net)
    print(net)
    logger.info(DEVICE)
    net = net.to(DEVICE)

    if config.WhichOneof('optimizer') == 'sgd':
        optimizer = optim.SGD(net.parameters(),
                              lr=config.sgd.learning_rate,
                              momentum=config.sgd.momentum,
                              weight_decay=config.sgd.weight_decay,
                              nesterov=config.sgd.nesterov)
        warmup_iters = config.sgd.warmup_iters
        warmup_factor = config.sgd.warmup_factor
        lr_decay_factor = config.sgd.lr_decay_factor
        lr_steps = config.sgd.lr_step
        if len(lr_steps) == 0:
            lr_steps = None
        print(config.sgd.learning_rate)
    elif config.WhichOneof('optimizer') == 'adam':
        optimizer = optim.Adam(net.parameters(),
                               lr=config.adam.learning_rate,
                               weight_decay=config.adam.weight_decay)
        warmup_iters = config.adam.warmup_iters
        warmup_factor = config.adam.warmup_factor
        lr_decay_factor = config.adam.lr_decay_factor
        lr_steps = config.adam.lr_step
        if len(lr_steps) == 0:
            lr_steps = None
    else:
        raise ValueError(f'Unsupported optimizer {config.WhichOneof("optimizer")}')

    # TODO: Add this to proto
    log_every_n = 250
    top1_acc, top5_acc = train_util.train(
        net, trainloader, valloader, criterion, optimizer,
        config.train_params.epochs, config.train_params.batch_size,
        checkpoint=config.train_params.checkpoint_fn, hierarchy=hierarchy,
        model_type=model_type, freeze_bb_bn=kwargs['freeze_bb_bn'],
        warmup_iters=warmup_iters, warmup_factor=warmup_factor,
        lr_decay_factor=lr_decay_factor,
        lr_steps=lr_steps, log_every_n=log_every_n,
    )

    logger.info("Printing Final Accuracy + OOD Detection stats")
    logger.info("Top 1 Accuracy: %f", top1_acc)
    logger.info("Top 5 Accuracy: %f", top5_acc)


if __name__=="__main__":
    args = parser.parse_args()
    main(args)
