# Hierarchical Out-of-Distribution Detection

## Overview

__Fine-grain Inference on Out-of-Distribution Data with Hierarchical Classification.__

Randolph Linderman, Jingyang Zhang, Nathan Inkawhich, Hai Li, Yiran Chen @ [Duke CEI Lab](https://cei.pratt.duke.edu/)


Paper (arxiv preprint): https://arxiv.org/abs/2209.04493

This work has been accepted to [NeurIPS 2022 MLSafety Workshop](https://neurips2022.mlsafety.org/)!

## Environment
Setup the conda environment:
```sh
conda create -n hierarchical-ood python=3.8
conda activate hierarchical-ood
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install numpy scikit-learn pandas tqdm protobuf nltk
```

## Config Files
We utilize [Protocol Buffers](https://developers.google.com/protocol-buffers)
to set experiment parameters, models, etc. through "config" files. The `protoc`
protobuf compiler is required to run our training scripts. The protocol buffer
binaries can be found at
https://github.com/protocolbuffers/protobuf/releases/tag/v3.17.3.
Choose the appropriate binary for your system.
Note are several newer versions which have not been tested with our code.

### Download `protoc`
To install on Linux:
```sh
mkdir -p local/protoc
cd local/protoc
PROTOC_ZIP=protoc-3.17.3-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/$PROTOC_ZIP
unzip protoc-3.17.3-linux-x86_64.zip
rm protoc-3.17.3-linux-x86_64.zip
cd ..
```

### Generate protos
Since protocol buffers are system dependent you will need to generate the
python files with `protoc`. We have provided a [script](scripts/make-protos.sh)
that will generate the protos by running the following command:
```sh
# From top level directory
sh scripts/make_protos.sh
```

### Setting up experiments with protos
The `.proto` files located [lib/protos](lib/protos) define all of the parameter
and model settings available. We provide all experimental configs for the
experiments in the paper under [experiments](experiments). We strongly
recommend storing all new experients in the [experiments/](experiments). We use
organize our experiments by dataset, model, hierarchy, etc.:
```
experiments
├── dataset
│   ├── model_type
│   │   ├── hierarchy
│   │   │   ├── experiment
├── imagenet100
│   ├── cascade
│   │   ├── pruned-wn
│   │   │   ├── softpred_R0
│   │   │   │   ├── exp.config
│   │   │   │   ├── exp.result
│   │   │   │   ├── checkpoint.pt
│   │   │   │   ├── train.log
│   │   │   │   ├── ...
│   │   │   ├── softpred_oe_R0
│   │   │   ├── ensemble_M3
│   ├── softmax
│   │   ├── R0
│   │   │   ├── exp.config
│   │   │   ├── exp.result
│   │   │   ├── checkpoint.pt
│   │   │   ├── train.log
│   │   │   ├── ...
│   │   ├── R1
│   ├── ...
├── ...
```

More detailed information on each of the fields are described in the comments
of the `.proto` files.

## Datasets
See [data/README](data/README.md).

## Training
To train a model pass your config file to the [main.py](main.py) script:
```sh
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config_fn experiments/imagenet100/softmax/R0/exp.config
```
The training log and checkpoint will be saved to the directory that contains
`exp.config`.

## Metrics
We provide several scripts for calculating each of our reported performance
metrics. The global OOD metrics are calculated with
[gather_metrics](gather_metrics.py):
```sh
CUDA_VISIBLE_DEVICES=0 python gather_metrics.py \
    --config_fn experiments/imagenet100/softmax/R0/exp.config
```
This will calculate AUROC, AUPR, TNR scores using MSP and temperature scaling
(ODIN w/out preprocessing) for the fine-grain OOD dataset and any far OOD
datasets specified in `exp.config`. If a hierarchical model is provided then it
will also calculate the path probability and entropy metrics. The metrics will
be saved to `exp.result`

Additional scripts are provided for calculating ensemble, Mahalanobis, and
hierarchy inference metrics.

## Notebooks
We provide 2 notebooks:
1. [Results](Results.ipynb): Tabulate results from `exp.result` files
2. [OODGamify](OODGamify.ipynb): Generate ROC plots and hierarchy inference
   plots
