import time

from collections import defaultdict
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import attacks
import calculate_log as callog
import hierarchy_metrics as hmetrics
from protos import main_pb2
import models

device = 'cuda' if torch.cuda.is_available() else 'cpu'
module_logger = logging.getLogger('__main__.train_util')


class AverageMetric:
    """Average metric

    Methods
    -------
    update_state(value, counts)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    """
    def __init__(self,):
        self.logger = logging.getLogger('__main__.train_util.AverageMetric')
        self._running_scores = torch.zeros(1)
        self._count = 0.

    def update_state(self, value, counts):
        self._count += counts
        self._running_scores += value

    def reset_state(self,):
        self._running_scores = torch.zeros_like(self._running_scores)
        self._count = 0

    def result(self,):
        return self._running_scores/self._count


# adapted from pytorch ImageNet example code
class Accuracy(AverageMetric):
    """Topk accuracy metric

    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    """
    def __init__(self, topk=(1,)):
        super().__init__()
        self.logger = logging.getLogger('__main__.train_util.Accuracy')
        self._maxk = max(topk)
        self._running_scores = torch.zeros(len(topk))
        self._topk = topk

    def update_state(self, outputs, targets):
        with torch.no_grad():
            self._count += targets.size(0)
            _, pred = outputs.topk(self._maxk, 1, True, True)

            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            for i, k in enumerate(self._topk):
                self._running_scores[i] += \
                    correct[:k].reshape(-1).float().sum(0).to('cpu')


class BCELoss:
    """Binary Cross Entropy criteria for use with ILR classifiers"""
    def __init__(self, weights=None):
        self.logger = logging.getLogger('__main__.train_util.BCELoss')
        self.weights = weights

    def to_one_hot(self, inp, num_classes):
        out = torch.zeros((inp.size()[0], num_classes), dtype=float,
                          requires_grad=False, device=device)
        out[torch.arange(inp.size(0)), inp.long()] = 1
        return out

    def __call__(self, outputs, labels):
        lbls = self.to_one_hot(labels, outputs.size(1))
        loss = F.binary_cross_entropy_with_logits(outputs, lbls,
                                                  pos_weight=self.weights)
        return loss


class OOD:
    """OOD metric

    Parameters
    ----------
    topk : tuple
        A set of topk accuracies to compute

    Methods
    -------
    update_state(outputs, targets)
        Update the state with the current batch outputs
    reset_state()
        Reset state to empty
    result()
        Return the result at the current state
    result_full()
        Return the result at the current state
    """
    def __init__(self, model='softmax', ood_methods=['MSP', 'ODIN'],):
        super().__init__()
        self.logger = logging.getLogger('__main__.train_util.OOD')
        self._ood_methods = ood_methods
        if type(model) == int:
            ref_config = main_pb2.Main()
            if model == ref_config.SOFTMAX:
                model = 'softmax'
            elif model == ref_config.ILR:
                model = 'ilr'
            elif model == ref_config.CASCADE:
                model = 'cascade'
            elif model == ref_config.CASCADEFCHEAD:
                model = 'cascadefchead'
            elif model == ref_config.SOFTMAXFCHEAD:
                model = 'softmaxfchead'
            elif model == ref_config.HILR:
                model = 'hilr'
            elif model == ref_config.AMSOFTMAX:
                model = 'amsoftmax'
            elif model == ref_config.AMCASCADE:
                model = 'amcascade'
            else:
                raise ValueError('Invalid model for OOD metrics')
        self._model = model
        self.reset_state()

    def reset_state(self):
        self._metric_results = {k: {} for k in self._ood_methods}
        self._id_scores = None
        self._ood_dsets = []

    def update_state(self, net, id_loader, ood_loader, dset="OOD"):
        if self._id_scores is None:
            msp_scores, odin_scores, _ = self.gen_msp_odin_scores(
                    net, id_loader, 1000.0, True, IPP=False, eps=0.01)
            self._id_scores = [msp_scores, odin_scores]
        mspood, odinood, _ = self.gen_msp_odin_scores(
            net, ood_loader, 1000.0, False, IPP=False, eps=0.01)

        self.logger.info("Computing OOD Statistics...")
        mspid = self._id_scores[0]
        if mspood.size > 0:
            msp_res = callog.metric(mspid, mspood)
            odin_res = callog.metric(self._id_scores[1], odinood)
        else:
            msp_res = {'TMP': {}}
            odin_res = {'TMP': {}}
            for met in ['AUROC', 'TNR', 'AUOUT']:
                msp_res['TMP'][met] = -1
                odin_res['TMP'][met] = -1
        self._metric_results["MSP"][dset] = defaultdict(list)
        self._metric_results["ODIN"][dset] = defaultdict(list)
        for met in ['AUROC', 'TNR', 'AUOUT']:
            self._metric_results["MSP"][dset][met].append(msp_res['TMP'][met])
            self._metric_results["ODIN"][dset][met].append(
                odin_res['TMP'][met])
        self._ood_dsets.append(dset)

    def gen_msp_odin_scores(self, net, loader, T, isID, IPP=False, eps=0.):
        """
        Generate OOD scores with the ODIN detector and the msp detector.

        Parameters
        ----------
        model : nn.Module
            base classifier network to test
        loader : torch.utils.data.DataLoader
            dataloader to compute ood scores over
        T : int
            temperature level to divide logits by
        isID : bool
            if the dataloader is for an in distribution set
        IPP : bool
            whether or not to do the input preprocessing step
        eps : float
            epsilon of perturbation if IPP is set
        """
        net.eval()
        baseline_scores = []
        odin_scores = []
        odin_ipp_scores = []
        baseline_scores = np.empty((0,))
        odin_scores = np.empty((0,))
        odin_ipp_scores = np.empty((0,))

        if IPP:
            raise NotImplementedError()
        else:
            with torch.no_grad():
                for dat, _ in loader:
                    dat = dat.to(device)
                    logits = net(dat)

                    # initialize tracker for max score across all synsets
                    if self._model in ['softmax', 'softmaxfchead',
                                       'cascade', 'cascadefchead',
                                       'amsoftmax',]:
                        bscores = torch.max(
                            F.softmax(logits.clone().detach(), dim=1),
                            dim=1)[0].cpu().numpy()
                        oscores = torch.max(
                            F.softmax(logits.clone().detach()/T, dim=1),
                            dim=1)[0].cpu().numpy()
                    elif self._model in ['ilr', 'hilr']:
                        bscores = torch.max(
                            torch.sigmoid(logits.clone().detach()),
                            dim=1)[0].cpu().numpy()
                        oscores = torch.max(
                            torch.sigmoid(logits.clone().detach()/T),
                            dim=1)[0].cpu().numpy()
                    # import pdb; pdb.set_trace()
                    baseline_scores = np.concatenate((
                        baseline_scores, bscores), axis=0)
                    odin_scores = np.concatenate((
                        odin_scores, oscores), axis=0)
        return baseline_scores, odin_scores, odin_ipp_scores

    def print_stats_of_list(self, prefix, dat):
        # Helper to print min/max/avg/std/len of values in a list
        dat = np.array(dat)
        self.logger.info(
            "{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}"
            .format(prefix, dat.min(), dat.max(),
                    dat.mean(), dat.std(), len(dat))
        )

    def print_result(self):
        for dset in self._ood_dsets:
            self.logger.info("OOD Dataset: " + dset)
            for ood in self._ood_methods:
                for met in ['AUROC', 'TNR', 'AUOUT']:
                    self.logger.info(ood + ' ' + met + ' ' +
                                     str(self._metric_results[ood][dset][met]))

    def print_result_full(self):
        self.print_result()

    def result(self):
        return dict(self._metric_results)


def update_lipschitz(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, models.SpectralNormConv2d) or \
               isinstance(m, models.SpectralNormLinear) or \
               isinstance(m, models.InducedNormConv2d) or \
               isinstance(m, models.InducedNormLinear):
                m.compute_weight(update=True)


def get_lipschitz_constants(model):
    lipschitz_constants = []
    for m in model.modules():
        if isinstance(m, models.SpectralNormConv2d) or \
           isinstance(m, models.SpectralNormLinear) or \
           isinstance(m, models.InducedNormConv2d) or \
           isinstance(m, models.InducedNormLinear) or \
           isinstance(m, models.LopConv2d) or isinstance(m, models.LopLinear):
            lipschitz_constants.append(m.scale.item())
    return lipschitz_constants


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'

def train(
        net, trainloader, testloader, criterion, optimizer, epochs, batch_size,
        log_every_n=250, checkpoint=None, attack=None, eps=0.5, iters=7,
        alpha=0.5/5, rand_start=True, attack_norm='inf',
        hierarchy=None,
        profile=False,
        warmup_iters=5,
        warmup_factor=0.1,
        model_type='SOFTMAX',
        freeze_bb_bn=False,
        lr_decay_factor=0.1,
        lr_steps=None,
        ):
    """
    Training a network

    Parameters
    ----------
    net : nn.Module
        Network for training
    trainloader : torch.utils.data.DataLoader
    testloader : torch.utils.data.DataLoader
    epochs : int
        Number of epochs in total.
    batch_size : int
        Batch size for training.
    log_every_n : int
        Intra-epoch logging interval in number of steps
    checkpoint : string
        Checkpoint path
    attack : string
        Type of attack to train against
    eps : float
        Attack strength
    iters : int
        Number of iterations for PGD attack
    alpha : float
        PGD attack step size
    rand_start : bool
        Whether to choose random starting point for attack
    attack_norm : string
        Indicates which norm to attack with (either 'inf' or 'l2')

    Returns
    -------
    tuple of floats:
        best top-1 validation accuracy achieved,
        best top-5 validation accuracy achieved
    """
    print(criterion)
    if (('AMCASCADE'==model_type) or ('CASCADE' in model_type)
            or ('MOS' in model_type)):
        if 'MOS' in model_type:
            accuracy = hmetrics.MOSAccuracy(hierarchy)
        else:
            accuracy = hmetrics.HierarchicalAccuracy(hierarchy,
                                                     soft_preds=True)
    else:
        accuracy = Accuracy((1, 5))
    train_loss = AverageMetric()
    test_loss = AverageMetric()
    best_acc = (0., 0.)  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if warmup_iters > 0:
        linear_warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
        if lr_steps is None:
            multistep_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(epochs//3)-warmup_iters,
                            2*int(epochs//3)-warmup_iters],
                gamma=lr_decay_factor)
        else:
            multistep_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(epochs * step)-warmup_iters
                            for step in lr_steps if step > 0.],
                gamma=lr_decay_factor)
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[linear_warmup_scheduler, multistep_scheduler],
            milestones=[4])
    else:
        if lr_steps is None:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(epochs//3), 2*int(epochs//3)],
                gamma=lr_decay_factor)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(epochs * step)
                            for step in lr_steps if step > 0.],
                gamma=lr_decay_factor)

    global_steps = 0
    import contextlib

    @contextlib.contextmanager
    def NoneCM():
        yield

    def get_profiler():
        if profile:
            return torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=2,
                                                 active=6, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    './profiler_output/'),
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
            )
        return NoneCM()

    def set_bn_to_eval(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        epochstart = time.time()
        module_logger.info('\nEpoch: %d' % epoch)
        module_logger.info('\nLearning Rate: %.4f' %
                           optimizer.param_groups[0]['lr'])
        if hasattr(criterion, 'step_weights'):
            criterion.print_weights()

        net.train()
        if isinstance(net, models.resnet_pytorch.ResNet) and freeze_bb_bn:
            set_bn_to_eval(net)
        # elif freeze_bb_bn:
        #     set_bn_to_eval(net.backbone)

        train_loss.reset_state()
        start = time.time()
        with get_profiler() as profiler:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                if attack == 'PGD':
                    adv_data = attacks.PGD_attack(
                        net, device, inputs.clone().detach(), targets, eps=eps,
                        alpha=alpha, iters=iters, rand_start=rand_start,
                        norm=attack_norm)
                elif attack == 'FGSM':
                    adv_data = attacks.FGSM_attack(
                        net, device, inputs.clone().detach(), targets, eps=eps)
                elif attack == 'MI-FGSM':
                    adv_data = attacks.MomentumIterative_attack(
                        net, device, inputs.clone().detach(), targets, eps=eps,
                        alpha=alpha, iters=iters, mu=1.0)
                else:
                    adv_data = inputs
                outputs = net(adv_data)
                net.zero_grad()
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                update_lipschitz(net)
                train_loss.update_state(loss.item(), 1)
                global_steps += 1

                if global_steps % log_every_n == 0:
                    end = time.time()
                    num_examples_per_second = log_every_n * batch_size / (end - start)
                    if hierarchy:
                        # top1 = accuracy.result_top1()
                        # top5 = accuracy.result()
                        module_logger.info("[Step=%d]\tLoss=%.4f\t%.1f examples/second"
                              % (global_steps, train_loss.result(), num_examples_per_second))
                        # module_logger.info("[Step=%d]\tLoss=%.4f\ttop1 acc=%.4f\tmean synset acc=%.4f\t%.1f examples/second"
                        #       % (global_steps, train_loss.result(), top1, top5, num_examples_per_second))
                    else:
                        # top1, top5 = accuracy.result()
                        module_logger.info("[Step=%d]\tLoss=%.4f\t%.1f examples/second"
                              % (global_steps, train_loss.result(), num_examples_per_second))
                        # module_logger.info("[Step=%d]\tLoss=%.4f\ttop1 acc=%.4f\ttop5 acc=%.4f\t%.1f examples/second"
                        #       % (global_steps, train_loss.result(), top1, top5, num_examples_per_second))
                    train_loss.reset_state()
                    #accuracy.reset_state()
                    start = time.time()
                if profile:
                    profiler.step()
        scheduler.step()
        accuracy.reset_state()

        """
        Start the testing code.
        """
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss.update_state(loss.item(), 1)
                accuracy.update_state(outputs, targets)
            lipschitz_constants = get_lipschitz_constants(net)
        if isinstance(accuracy, hmetrics.HierarchicalAccuracy):
            val_top1 = accuracy.result_pred()
            val_top5 = accuracy.result()
            module_logger.info(
                "Test Loss=%.4f, Test pred acc=%.4f, Test mean synset acc=%.4f" % \
                (test_loss.result(), val_top1, val_top5))
            module_logger.info("Error Depth: {}".format(accuracy.result_error_depth()))
        elif isinstance(accuracy, hmetrics.MOSAccuracy):
            val_top1 = accuracy.result()
            val_groupacc = accuracy.result_groupwise()
            val_top5 = val_groupacc.mean()
            module_logger.info(
                "Test Loss=%.4f, Test top-1 acc=%.4f" %
                (test_loss.result(), val_top1))
            module_logger.info('Group Accuracy:\n')
            module_logger.info(val_groupacc)
        else:
            val_top1, val_top5 = accuracy.result()
            module_logger.info(
                "Test Loss=%.4f, Test top-1 acc=%.4f, Test top-5 acc=%.4f" %
                (test_loss.result(), val_top1, val_top5))
        if len(lipschitz_constants) > 0:
            module_logger.info('Lipsh: {}'.format(
                pretty_repr(lipschitz_constants)))
        accuracy.reset_state()
        test_loss.reset_state()

        if (checkpoint is not None) and (val_top1 > best_acc[0]):
            best_acc = (val_top1, val_top5)
            module_logger.info("Saving...")
            torch.save(net.state_dict(), checkpoint)
        module_logger.info("Epoch time: {}".format(time.time() - epochstart))
        if hasattr(criterion, 'step_weights'):
            criterion.step_weights()
    return best_acc
