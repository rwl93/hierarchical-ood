import models
import train_util
import hierarchy_metrics as hm
import hierarchy_util

def build_model(config, num_id_classes, id_classes):
    kwargs = {}
    if config.HasField('model_config'):
        mc = getattr(config, config.WhichOneof("model_config"))
        if hasattr(mc, 'fc_head_sizes'):
            kwargs['head_layer_sizes'] = mc.fc_head_sizes
        if hasattr(mc, 'split_fchead_layers'):
            kwargs['split_fchead_layers'] = mc.split_fchead_layers
    backbone = getattr(models, config.backbone)

    # Load Model
    ood_std = None
    if config.model in [config.SOFTMAX, config.ILR, config.SOFTMAXFCHEAD]:
        if config.model == config.SOFTMAXFCHEAD:
            net = models.build_softmax_fchead(num_id_classes,
                                              backbone=config.backbone,
                                              **kwargs,
                                             )
        else:
            net = backbone(num_classes=num_id_classes)
        id_hierarchy = None
        ood_hierarchy = None
        acc = train_util.Accuracy((1, 5))
        ood = train_util.OOD(config.model)
    elif config.model in [config.CASCADE, config.HILR, config.CASCADEFCHEAD]:
        id_hierarchy = hierarchy_util.Hierarchy(id_classes, config.hierarchy_fn)
        # ood_hierarchy = hierarchy_util.Hierarchy(ood_ds.classes,
        #         hierarchy_fn)
        ood_hierarchy = id_hierarchy
        acc = hm.HierarchicalAccuracy(id_hierarchy,
                                      soft_preds=config.hl.softpred_loss)
        ood = hm.HierarchicalOOD(ood_hierarchy, id_hierarchy,
                                 model=config.model,
                                 soft_preds=config.hl.softpred_loss)
        ood_std = train_util.OOD(config.model)

        print(kwargs)
        if config.model == config.CASCADEFCHEAD:
            net = models.build_softmax_cascade(
                id_hierarchy, backbone=config.backbone, **kwargs)
            print(net.head)
        else:
            net = backbone(num_classes=id_hierarchy.num_classes)
    elif config.model == config.MOS:
        id_hierarchy = hierarchy_util.Hierarchy(id_classes, config.hierarchy_fn)
        net = models.build_MOS(id_hierarchy, backbone=config.backbone, **kwargs)
        acc = hm.MOSAccuracy(id_hierarchy)
        ood = hm.MOSOOD(id_hierarchy)
    return net, acc, ood, ood_std
