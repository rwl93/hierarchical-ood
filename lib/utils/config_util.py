"""Config file reader functions."""
from google.protobuf import text_format
import os
from ..protos import main_pb2, ensemble_pb2

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
            basename = 'checkpoint.pt'
            logbasename = 'result.log' if for_metrics else 'train.log'
        elif 'experiments' in dirname:
            basename = 'checkpoint.pt'
            logbasename = 'result.log' if for_metrics else 'train.log'
        else:
            raise ValueError("Command line model specification no longer " +
                             "accepted. Use protocol buffer configs instead.")

        if ensemble:
            pb_config.train_params.checkpoint_fn += \
                os.path.join(dirname, 'R0', basename)
            for model_idx in range(pb_config.num_models-1):
                pb_config.train_params.checkpoint_fn += \
                    ';' + os.path.join(dirname, f'R{model_idx+1}', basename)
        else:
            pb_config.train_params.checkpoint_fn = os.path.join(dirname, basename)
        pb_config.train_params.log_fn = os.path.join(dirname, logbasename)
    check_config(pb_config)
    return pb_config


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
    if config.WhichOneof('loss') == 'hl':
        if not config.hl.softpred_loss:
            if ((config.hl.softpred_range.start != 0.) or
                (config.hl.softpred_range.end != 0.)):
                raise ValueError('Non-zero soft prediction loss range ' +
                                 '[{},{}] w/ softpred_loss = False'.format(
                                 config.hl.softpred_range.start,
                                 config.hl.softpred_range.end
                                 ))
        if not config.hl.outlier_exposure:
            if ((config.hl.outlier_exposure_range.start != 0.) or
                (config.hl.outlier_exposure_range.end != 0.)):
                raise ValueError('Non-zero outlier exposure loss range ' +
                                 '[{},{}] w/ outlier_exposure_loss = False'.format(
                                 config.hl.outlier_exposure_range.start,
                                 config.hl.outlier_exposure_range.end
                                 ))
