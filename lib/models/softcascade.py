"""Softmax Cascade Models

Module provides head networks for softmax cascades
"""
import torch
import torch.nn as nn
import logging
import warnings
import torch.nn.functional as F
from functools import partial


__all__ = ['build_softmax_cascade', 'build_softmax_fchead', 'build_MOS',]
logging.captureWarnings(True)
module_logger = logging.getLogger('__main__.models.softcascade')
SOFT_CASCADE_FC_HEAD_SIZES = [256, 128]


class SoftCascade(nn.Module):
    """Softmax Cascade Model

    Parameters
    ----------

    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, embed=False):
        out = self.backbone(x)
        out = self.head(out, embed=embed)
        return out


class SoftCascadeFCHead(nn.Module):
    """Fully-connected softmax cascade head.

    Parameters
    ----------
    inplanes : list(int)
        input dimensions
    hierarchy : hierarchy_util.Hierarchy
        hierarchy to create head for
    """
    def __init__(self, inplanes, hierarchy,
                 embed_layer=False,
                 layer_sizes=SOFT_CASCADE_FC_HEAD_SIZES,
                 **kwargs
                 ):
        super().__init__()
        print(f'head_layer_sizes: {layer_sizes}')
        if embed_layer:
            if layer_sizes is not None:
                layer_sizes.append(3)
            else:
                layer_sizes = [3]
        self._gen_synset_modules(inplanes, hierarchy, layer_sizes)

    def _gen_synset_modules(self, inplanes, hierarchy, layer_sizes):
        """Generate the head FC layers"""
        linlayer = nn.Linear
        self.synset_mods = nn.ModuleList()
        for o, b in zip(hierarchy.synset_offsets, hierarchy.synset_bounds):
            layers = []
            curr_inplanes = inplanes
            for ls in layer_sizes:
                layers.append(linlayer(curr_inplanes, ls))
                # TODO: Add non-linear activation after each intermediate layer!
                # XXX: Commented because not yet retrained
                # layers.append(nn.ReLU())
                if ls != 3:
                    layers.append(nn.ReLU())
                curr_inplanes = ls
            layers.append(linlayer(curr_inplanes, int(b-o+1)))
            self.synset_mods.append(nn.Sequential(*layers))

    def forward(self, x, embed=False):
        out = []
        for smod in self.synset_mods:
            if embed:
                o = x
                for l in smod[:-1]:
                    o = l(o)
                out.append(o)
            else:
                out.append(smod(x))
        return torch.cat(out, dim=1)


class SoftmaxFCHead(nn.Module):
    """Fully-connected softmax cascade head.

    Parameters
    ----------
    inplanes : list(int)
        input dimensions
    outplanes : int
        output dimension
    """
    def __init__(self, inplanes, outplanes,
                 embed_layer=False,
                 layer_sizes=SOFT_CASCADE_FC_HEAD_SIZES,
                 **kwargs
                 ):
        super().__init__()
        if embed_layer:
            if layer_sizes is not None:
                layer_sizes.append(3)
            else:
                layer_sizes = [3]
        layers = []
        print("head_layer_sizes: {}".format(layer_sizes))
        curr_inplanes = inplanes
        linlayer = nn.Linear
        for ls in layer_sizes:
            layers.append(linlayer(curr_inplanes, ls))
            # TODO: Add non-linear activation after each intermediate layer!
            # XXX: Commented because not yet retrained
            # layers.append(nn.ReLU())
            if ls != 3:
                layers.append(nn.ReLU())
            curr_inplanes = ls
        self.head_layers = nn.Sequential(*layers)
        self.class_layer = linlayer(curr_inplanes, outplanes)

    def forward(self, x, embed=False):
        x = self.head_layers(x)
        if embed:
            return x
        return self.class_layer(x)



class MOSHead(nn.Module):
    """MOS Head Module

    Parameters
    ----------
    inplanes : list(int)
        input dimensions
    hierarchy : hierarchy_util.Hierarchy
        Expecting a 2 level head where the superclasses correspond to the
        "groups" in the MOS technique
    """
    def __init__(self, inplanes, hierarchy, layer_sizes=[], **kwargs):
        super().__init__()
        if hierarchy.max_depth != 2:
            if hierarchy.max_depth < 2:
                raise RuntimeError("Hierarchy must be depth 2 for MOS Head")
            warnings.warn("Hierarchy depth > 2 for MOS Head", RuntimeWarning)
        print("head_layer_sizes: {}".format(layer_sizes))
        layers = []
        curr_inplanes = inplanes
        linlayer = nn.Linear
        for ls in layer_sizes:
            layers.append(linlayer(curr_inplanes, ls))
            curr_inplanes = ls
            # TODO: Add non-linear activation after each intermediate layer!
            # XXX: Commented because not yet retrained
            # layers.append(nn.ReLU())
        # TODO: Make bias=False for classification layer
        # XXX: Not yet retrained with bias = False
        layers.append(linlayer(curr_inplanes, hierarchy.get_num_MOSclasses()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, embed=False):
        if embed:
            return x
        return self.layers(x)


def _backbone(backbone, **kwargs):
    """ Returns a backbone object for the given backbone.
    """
    if 'densenet' in backbone:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    elif 'seresnext' in backbone or 'seresnet' in backbone \
            or 'senet' in backbone:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    elif 'resnet' in backbone:
        from .resnet_pytorch import ResnetBackbone as b
    elif 'mobilenet' in backbone:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    elif 'vgg' in backbone:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    elif 'EfficientNet' in backbone:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    elif 'AdditiveMarginNet' in backbone:
        from .additive_margin_net import ConvNet as b
    else:
        raise NotImplementedError(
            'Backbone class for  \'{}\' not implemented.'.format(backbone))
    return b(backbone, **kwargs)


def build_softmax_cascade(hierarchy,
                          backbone='resnet50',
                          freeze_bb=False,
                          freeze_bb_bn=False,
                          bb_pretrained=False,
                          embed_layer=False,
                          head_layer_sizes=SOFT_CASCADE_FC_HEAD_SIZES,
                          split_fchead_layers=True,
                          **kwargs
                          ):
    """Construct a Softmax Cascade Model

    The network utilizes a classification backbone (e.g. ResNet50) as a feature
    extractor and builds a softmax cascade model from it based on the
    hierarchy.

    Parameters
    ----------
    hierarchy : hierarchy_util.Hierarchy
    backbone : string, optional
        Backbone model to use.
        Currently, supporting {'resnet50'}
    freeze_bb : bool
        Freeze the backbone weights if True
    freeze_bb_bn : bool
        Freeze the batch norm moving stats during training if True
    bb_pretrained : bool
        Pulls pretrained weights from pytorch if True
    head_layer_sizes : list(int)
        The head's intermediate synset layer sizes

    Returns
    -------
    torch model
    """
    # Get parameters
    b = _backbone(backbone,
                  freeze_bb=freeze_bb,
                  freeze_bb_bn=freeze_bb_bn,
                  bb_pretrained=bb_pretrained,
                  **kwargs,
                  )
    backbone_outplanes = b.outplanes
    if split_fchead_layers:
        head = SoftCascadeFCHead(backbone_outplanes,
                                 hierarchy,
                                 embed_layer=embed_layer,
                                 layer_sizes=head_layer_sizes,
                                 **kwargs)
    else:
        head = SoftmaxFCHead(backbone_outplanes,
                             hierarchy.num_classes,
                             embed_layer=embed_layer,
                             layer_sizes=head_layer_sizes,
                             **kwargs)
    return SoftCascade(b, head)


def build_softmax_fchead(num_classes,
                         backbone='resnet50',
                         freeze_bb=False,
                         freeze_bb_bn=False,
                         bb_pretrained=False,
                         embed_layer=False,
                         head_layer_sizes=SOFT_CASCADE_FC_HEAD_SIZES,
                         **kwargs
                         ):
    """Construct a Softmax FC Head

    The network utilizes a classification backbone (e.g. ResNet50) as a feature
    extractor and adds ``len(head_layer_sizes)+1`` fully-connected linear
    layers to the end of the model to perform classification.

    Parameters
    ----------
    num_classes : int
        Number of classes for final output size
    backbone : string, optional
        Backbone model to use.
        Currently, supporting {'resnet50'}
    freeze_bb : bool
        Freeze the backbone weights if True
    freeze_bb_bn : bool
        Freeze the batch norm moving stats during training if True
    bb_pretrained : bool
        Pulls pretrained weights from pytorch if True
    head_layer_sizes : list(int)
        The head's intermediate synset layer sizes

    Returns
    -------
    torch model
    """
    # Get parameters
    b = _backbone(backbone,
                  freeze_bb=freeze_bb,
                  freeze_bb_bn=freeze_bb_bn,
                  bb_pretrained=bb_pretrained,
                  **kwargs,
                  )
    backbone_outplanes = b.outplanes
    head = SoftmaxFCHead(backbone_outplanes, num_classes,
                         embed_layer=embed_layer,
                         layer_sizes=head_layer_sizes,
                         **kwargs)
    return SoftCascade(b, head)


def build_MOS(hierarchy,
              backbone='resnet50',
              freeze_bb=False,
              freeze_bb_bn=False,
              bb_pretrained=False,
              head_layer_sizes=[],
              **kwargs
              ):
    """Construct a MOS Model

    The network utilizes a classification backbone (e.g. ResNet50) as a feature
    extractor and builds a softmax for each MOS "group" based on the hierarchy.

    Parameters
    ----------
    hierarchy : hierarchy_util.Hierarchy
    backbone : string, optional
        Backbone model to use.
        Currently, supporting {'resnet50'}
    freeze_bb : bool
        Freeze the backbone weights if True
    freeze_bb_bn : bool
        Freeze the batch norm moving stats during training if True
    bb_pretrained : bool
        Pulls pretrained weights from pytorch if True
    head_layer_sizes : list(int)
        The head's intermediate synset layer sizes

    Returns
    -------
    torch model
    """
    # Get parameters
    b = _backbone(backbone,
                  freeze_bb=freeze_bb,
                  freeze_bb_bn=freeze_bb_bn,
                  bb_pretrained=bb_pretrained,
                  **kwargs,
                  )
    backbone_outplanes = b.outplanes
    head = MOSHead(backbone_outplanes, hierarchy, head_layer_sizes,
                   **kwargs)
    return SoftCascade(b, head)
