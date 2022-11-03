import logging
import numpy as np
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets


logger = logging.getLogger('__main__.utils.dataset_util')


def gen_datasets(datadir,
                 mean=None,
                 std=None,
                 resize=None,
                 cropsize=None,
                 ):
    """Generate datasets for experiment.

    Preprocessing from pytorch Imagenet example code

    Parameters
    ----------
    datadir : string
        path to directory of data
    mean=None,
    std=None,
    resize=None,
    cropsize=None,

    Returns
    -------
    Tuple of torchvision.datasets.Dataset objects:
        train_dataset, val_dataset, ood_dataset
    """
    if mean is None:
        if 'cifar' in datadir.lower():
            cropsize = 32
            resize = 32
            if 'cifar10' == datadir.lower():
                mean=[0.4914, 0.4822, 0.4465]
                std=[0.2023, 0.1994, 0.2010]
            elif 'cifar100' == datadir.lower():
                mean=[0.5071, 0.4867, 0.4408]
                std=[0.2675, 0.2565, 0.2761]
            else:
                raise ValueError("Invalid dataset")
        else:
            # Imagenet stats
            cropsize = 224
            resize = 256
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])

    if 'cifar10' == datadir.lower():
        train_dataset = datasets.CIFAR10('data', train=True, download=True,
                                         transform=train_transform)
        val_dataset = datasets.CIFAR10('data', train=False, download=True,
                                       transform=eval_transform)
        ood_dataset = datasets.CIFAR100('data', train=False, download=True,
                                        transform=eval_transform)
    elif 'cifar100' == datadir.lower():
        train_dataset = datasets.CIFAR100('data', train=True, download=True,
                                          transform=train_transform)
        val_dataset = datasets.CIFAR100('data', train=False, download=True,
                                        transform=eval_transform)
        ood_dataset = datasets.CIFAR10('data', train=False, download=True,
                                       transform=eval_transform)
    else:
        train_dataset = datasets.ImageFolder(os.path.join(datadir, 'train'),
                                             train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(datadir, 'val'),
                                           eval_transform)
        ood_dataset = datasets.ImageFolder(os.path.join(datadir, 'ood'),
                                           eval_transform)
    return train_dataset, val_dataset, ood_dataset


def gen_far_ood_datasets(dset: str = "iNaturalist"):
    if dset not in ['iNaturalist', 'SUN', 'Places', 'Textures',
                    'coarseid-fineood', 'coarseid-coarseood',
                    'imagenet1000-fineood', 'imagenet1000-mediumood',
                    'imagenet1000-coarseood', 'balanced100-coarseood',
                    'balanced100-mediumood', 'balanced100-fineood',
                    'balanced100-finemediumood',
                    ]:
        raise ValueError("Unknown far ood dataset: " + dset)
    datadir = "data/" + dset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.ImageFolder(datadir, transform)
    return ds


def print_stats_of_list(prefix,dat):
    # Helper to print min/max/avg/std/len of values in a list
    dat = np.array(dat)
    logger.info("{} Min: {:.4f}; Max: {:.4f}; Avg: {:.4f}; Std: {:.4f}; Len: {}".format(
            prefix, dat.min(), dat.max(), dat.mean(), dat.std(), len(dat))
    )

