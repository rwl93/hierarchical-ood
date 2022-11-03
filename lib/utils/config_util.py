"""Config file reader functions."""
import datetime
from google.protobuf import text_format
import os
from protos import main_pb2, ensemble_pb2

# Read config file
def read_config(config_fn, ensemble=False, for_metrics=False):
    if ensemble:
        pb_config = ensemble_pb2.Ensemble()
    else:
        pb_config = main_pb2.Main()

    with open(config_fn, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pb_config)

    # Generate checkpoint filename
    if not pb_config.train_params.checkpoint_fn:
        dirname = os.path.dirname(config_fn)
        dataset_name = os.path.split(pb_config.data_dir)[-1]
        print(dataset_name)
        if pb_config.savedir:
            dirname = pb_config.savedir
            # basename = 'exp.result'  if for_metrics else 'checkpoint.pt'
            basename = 'checkpoint.pt'
            logbasename = 'result.log' if for_metrics else 'train.log'
        elif 'experiments' in dirname:
            # basename = 'exp.result'  if for_metrics else 'checkpoint.pt'
            basename = 'checkpoint.pt'
            logbasename = 'result.log' if for_metrics else 'train.log'
        elif pb_config.model == pb_config.SOFTMAX:
            basename = "ckpts/softmax_{}_SEED{}_ITER{}_EPOCHS{}.pt".format(
                dataset_name, pb_config.seed, pb_config.repeat_iters-1,
                pb_config.train_params.epochs)
            if for_metrics:
                # basename = str(basename).replace(
                #     'ckpts/','results/').replace('.pt','.result')
                logbasename = str(basename).replace(
                    'ckpts/','results/').replace('.pt','.log')
            else:
                logbasename = str(basename).replace(
                    'ckpts/','logs/').replace('.pt','.log')
        elif pb_config.model == pb_config.ILR:
            basename = "ckpts/ilr_{}_SEED{}_ITER{}_EPOCHS{}.pt".format(
                dataset_name, pb_config.seed, pb_config.repeat_iters-1,
                pb_config.train_params.epochs)
            if for_metrics:
                # basename = str(basename).replace(
                #     'ckpts/','results/').replace('.pt','.result')
                logbasename = str(basename).replace(
                    'ckpts/','results/').replace('.pt','.log')
            else:
                logbasename = str(basename).replace(
                    'ckpts/','logs/').replace('.pt','.log')
        elif pb_config.model == pb_config.HILR:
            basename = "ckpts/hilr_{}_{}_SEED{}_ITER{}_EPOCHS{}.pt".format(
                dataset_name,
                os.path.splitext(os.path.basename(pb_config.hierarchy_fn))[0],
                pb_config.seed, pb_config.repeat_iters-1,
                pb_config.train_params.epochs)
            if for_metrics:
                # basename = str(basename).replace(
                #     'ckpts/','results/').replace('.pt','.result')
                logbasename = str(basename).replace(
                    'ckpts/','results/').replace('.pt','.log')
            else:
                logbasename = str(basename).replace(
                    'ckpts/','logs/').replace('.pt','.log')
        elif pb_config.model in [pb_config.CASCADE, pb_config.CASCADEFCHEAD]:
            model_str = 'cascade'
            if pb_config.model == pb_config.CASCADEFCHEAD:
                model_str += 'fchead'
            # NOTE: Assumes that cascade models use hierarchical loss 'hl'
            loss_str = ''
            if pb_config.loss.softpred_loss:
                loss_str += 'softpred-'
                loss_str += str(config.hl.softpred_range.start).replace('.','p')
                loss_str += '-'
                loss_str += str(config.hl.softpred_range.end).replace('.','p')
            if pb_config.loss.synsetce_loss:
                if loss_str:
                    loss_str += '_'
                loss_str += 'synce-'
                loss_str += str(pb_config.loss.synsetce_range.start).replace('.','p')
                loss_str += '-'
                loss_str += str(pb_config.loss.synsetce_range.end).replace('.','p')
            if pb_config.loss.outlier_exposure:
                loss_str += '_oe-'
                loss_str += str(pb_config.loss.outlier_exposure_range.start).replace('.','p')
                loss_str += '-'
                loss_str += str(pb_config.loss.outlier_exposure_range.end).replace('.','p')
            basename = "ckpts/{}_{}_{}_{}_SEED{}_ITER{}_EPOCHS{}.pt".format(
                model_str, dataset_name, loss_str,
                os.path.splitext(os.path.basename(pb_config.hierarchy_fn))[0],
                pb_config.seed, pb_config.repeat_iters-1,
                pb_config.train_params.epochs)
            if for_metrics:
                # basename = str(basename).replace(
                #     'ckpts/','results/').replace('.pt','.result')
                logbasename = str(basename).replace(
                    'ckpts/','results/').replace('.pt','.log')
            else:
                logbasename = str(basename).replace(
                    'ckpts/','logs/').replace('.pt','.log')
        if ensemble:
            pb_config.train_params.checkpoint_fn += \
                os.path.join(dirname, 'R0', basename)
            for model_idx in range(pb_config.num_models-1):
                pb_config.train_params.checkpoint_fn += \
                    ';' + os.path.join(dirname, f'R{model_idx+1}', basename)
        else:
            pb_config.train_params.checkpoint_fn = os.path.join(dirname, basename)
        pb_config.train_params.log_fn = os.path.join(dirname, logbasename)
    # # Check model_config
    # if pb_config.WhichOneof('model_config') != None:
    #     pb_config['model_config']['']
    check_config(pb_config)
    return pb_config


def build_config_from_args(args, for_metrics=False):
    config = main_pb2.Main()
    config.model = eval('config.' + args.model.upper())
    config.data_dir = args.data
    config.hierarchy_fn = args.hierarchy
    config.train_params.batch_size = 64
    config.train_params.checkpoint_fn = str(args.checkpoint)
    config.verbose = args.verbose
    if for_metrics:
        config.train_params.checkpoint_fn = str(args.checkpoint).replace(
            'ckpts/', 'results/').replace('.pt','.result')
        config.train_params.log_fn = str(args.checkpoint).replace(
            'ckpts/', 'results/').replace('.pt','.log')
        config.hl.CopyFrom(main_pb2.HierarchicalLoss())
        config.hl.softpred_loss = args.softpredceloss
    else:
        config.train_params.log_fn = 'train.log'
        config.train_params.epochs = args.epochs
        config.sgd.CopyFrom(main_pb2.SGDParams())
        config.sgd.learning_rate = args.learningrate
        config.seed = args.seed
        config.repeat_iters = args.iters
        config.no_save = args.nosave
        config.backbone = args.backbone
        if args.model == 'softmax':
            config.ce.CopyFrom(main_pb2.CrossEntropy())
        elif args.model == 'ilr':
            config.bce.CopyFrom(main_pb2.BinaryCrossEntropy())
        elif args.model == 'hilr':
            config.hbce.CopyFrom(main_pb2.HierarchicalBinaryCrossEntropy())
        elif 'cascade' in args.model:
            config.hl.CopyFrom(main_pb2.HierarchicalLoss())
            if args.softpredceloss:
                config.hl.synsetce_loss = False
                config.hl.softpred_loss = True
                config.hl.outlier_exposure = args.outlierexposure
                config.hl.synsetce_range.start = 0.
                config.hl.synsetce_range.end = 0.
                if args.outlierexposure:
                    config.hl.softpred_range.start = 0.5
                    config.hl.softpred_range.end = 0.5
                    config.hl.outlier_exposure_range.start = 0.5
                    config.hl.outlier_exposure_range.end = 0.5
                else:
                    config.hl.softpred_range.start = 1.
                    config.hl.softpred_range.end = 1.
                    config.hl.outlier_exposure_range.start = 0.
                    config.hl.outlier_exposure_range.end = 0.
            elif args.splandce:
                config.hl.synsetce_loss = True
                config.hl.softpred_loss = True
                if args.dynamicweight:
                    config.hl.outlier_exposure = args.outlierexposure
                    if args.outlierexposure:
                        config.hl.outlier_exposure_range.start = \
                            args.outlierexposurerange[0]
                        config.hl.outlier_exposure_range.end = \
                            args.outlierexposurerange[1]
                    else:
                        config.hl.outlier_exposure_range.start = 0.
                        config.hl.outlier_exposure_range.end = 0.
                    config.hl.synsetce_range.start = args.softlossrange[0]
                    config.hl.synsetce_range.end = args.softlossrange[1]
                    config.hl.softpred_range.start = args.synsetcerange[0]
                    config.hl.softpred_range.end = args.synsetcerange[1]
                else:
                    config.hl.outlier_exposure = False
                    config.hl.synsetce_range.start = 1.
                    config.hl.synsetce_range.end = 1.
                    config.hl.softpred_range.start = 1.
                    config.hl.softpred_range.end = 1.
                    config.hl.outlier_exposure_range.start = 0.
                    config.hl.outlier_exposure_range.end = 0.
            else:
                config.hl.synsetce_loss = True
                config.hl.softpred_loss = False
                config.hl.outlier_exposure = False
                config.hl.synsetce_range.start = 1.
                config.hl.synsetce_range.end = 1.
                config.hl.softpred_range.start = 0.
                config.hl.softpred_range.end = 0.
                config.hl.outlier_exposure_range.start = 0.
                config.hl.outlier_exposure_range.end = 0.
        check_config(pb_config)
    return config


def check_config(config):
    if config.model == config.SOFTMAX:
        if config.WhichOneof('loss') != 'ce':
            raise ValueError(
                'Invalid loss type {} for softmax type model.'.format(
                    config.WhichOneof('loss')))
    elif config.model in [config.CASCADE, config.CASCADEFCHEAD]:
        if config.WhichOneof('loss') != 'hl':
            raise ValueError(
                'Invalid loss type {} for cascade type model.'.format(
                    config.WhichOneof('loss')))
    elif config.model == config.HILR:
        if config.WhichOneof('loss') != 'hbce':
            raise ValueError(
                'Invalid loss type {} for hilr type model.'.format(
                    config.WhichOneof('loss')))
    elif config.model == config.ILR:
        if config.WhichOneof('loss') != 'bce':
            raise ValueError(
                'Invalid loss type {} for ilr type model.'.format(
                    config.WhichOneof('loss')))
    if config.WhichOneof('loss') == 'hl':
        if not config.hl.softpred_loss:
            if ((config.hl.softpred_range.start != 0.) or
                (config.hl.softpred_range.end != 0.)):
                raise ValueError('Non-zero soft prediction loss range ' +
                                 '[{},{}] w/ softpred_loss = False'.format(
                                 config.hl.softpred_range.start,
                                 config.hl.softpred_range.end
                                 ))
        if not config.hl.synsetce_loss: 
            if ((config.hl.synsetce_range.start != 0.) or
                (config.hl.synsetce_range.end != 0.)):
                raise ValueError('Non-zero synset CE loss range ' +
                                 '[{},{}] w/ synsetce_loss = False'.format(
                                 config.hl.synsetce_range.start,
                                 config.hl.synsetce_range.end
                                 ))
        if not config.hl.outlier_exposure:
            if ((config.hl.outlier_exposure_range.start != 0.) or
                (config.hl.outlier_exposure_range.end != 0.)):
                raise ValueError('Non-zero outlier exposure loss range ' +
                                 '[{},{}] w/ outlier_exposure_loss = False'.format(
                                 config.hl.outlier_exposure_range.start,
                                 config.hl.outlier_exposure_range.end
                                 ))
