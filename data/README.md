# Datasets
All datasets are contained in `./data`. There are two types of datasets: (1)
Standard training and validation datasets, and (2) far-OOD datasets. We assume
that the datasets follow the following structure:
```
data
├── balanced100-id-labels.csv
├── balanced100-ood-labels.csv
├── gen-symlinks.sh
├── imagenet100-id-labels.csv
├── imagenet100-ood-labels.csv
├── imagenet1000-id-labels.csv
├── imagenet1000-ood-labels.csv
├── imagenet100
│   ├── train
│   │   ├── n01530575
│   │   ├── n01531178
│   │   ├── ...
│   ├── val
│   │   ├── n01530575
│   │   ├── n01531178
│   │   ├── ...
│   ├── ood
│   │   ├── n01534433
│   │   ├── n02099429
│   │   ├── ...
├── balanced100
│   ├── train
│   │   ├── ...
│   ├── val
│   │   ├── ...
│   ├── ood
│   │   ├── ...
├── imagenet1000
│   ├── train
│   │   ├── ...
│   ├── val
│   │   ├── ...
│   ├── ood
│   │   ├── ...
├── iNaturalist
│   ├── ...
├── ...

```
We utilize symbolic links to the data to avoid wasteful memory usage. For each
standard set each class is symlinked to the appropriate image directory. For
example, `imagenet100>train>n01530575 -> IMAGENETDIR/train/n01530575`. We
provide a script `data/gen-symlinks.sh` to setup the symlinks. The following
commands will create the necessary datasets to recreate our experimental
results.

```sh
cd data
# Setup Imagenet100
mkdir -p imagenet100/{train,val,ood}
./gen-symlinks.sh imagenet100-id-labels.csv <IMAGENETDIR>/train imagenet100/train
./gen-symlinks.sh imagenet100-id-labels.csv <IMAGENETDIR>/val imagenet100/val
./gen-symlinks.sh imagenet100-ood-labels.csv <IMAGENETDIR>/val imagenet100/ood

# Setup Balanced100
mkdir -p balanced100/{train,val,ood}
./gen-symlinks.sh balanced100-id-labels.csv <IMAGENETDIR>/train balanced100/train
./gen-symlinks.sh balanced100-id-labels.csv <IMAGENETDIR>/val balanced100/val
./gen-symlinks.sh balanced100-ood-labels.csv <IMAGENETDIR>/val balanced100/ood

# Setup Imagenet1000
mkdir -p imagenet1000/{train,val,ood}
./gen-symlinks.sh imagenet1000-id-labels.csv <IMAGENETDIR>/train imagenet1000/train
./gen-symlinks.sh imagenet1000-id-labels.csv <IMAGENETDIR>/val imagenet1000/val
./gen-symlinks.sh imagenet1000-ood-labels.csv <IMAGENETDIR>/val imagenet1000/ood

# Setup Imagenet100 Granularity Sorted OOD Datasets
mkdir imagenet100-{coarse,fine}ood
./gen-symlinks.sh imagenet100-ood-labels.csv <IMAGENETDIR>/val imagenet100-coarse coarse
./gen-symlinks.sh imagenet100-ood-labels.csv <IMAGENETDIR>/val imagenet100-fine fine

# Setup balanced100 Granularity Sorted OOD Datasets
mkdir balanced100-{coarse,medium,fine,finemedium}ood
./gen-symlinks.sh balanced100-ood-labels.csv <IMAGENETDIR>/val balanced100-coarse coarse
./gen-symlinks.sh balanced100-ood-labels.csv <IMAGENETDIR>/val balanced100-medium medium
./gen-symlinks.sh balanced100-ood-labels.csv <IMAGENETDIR>/val balanced100-fine fine
./gen-symlinks.sh balanced100-ood-labels.csv <IMAGENETDIR>/val balanced100-finemedium finemedium

# Setup imagenet1000 Granularity Sorted OOD Datasets
mkdir imagenet1000-{coarse,medium,fine}ood
./gen-symlinks.sh imagenet1000-ood-labels.csv <IMAGENETDIR>/val imagenet1000-coarse coarse
./gen-symlinks.sh imagenet1000-ood-labels.csv <IMAGENETDIR>/val imagenet1000-medium medium
./gen-symlinks.sh imagenet1000-ood-labels.csv <IMAGENETDIR>/val imagenet1000-fine fine

# Far OOD datasets
ln -s <INATURALISTDIR> iNaturalist
ln -s <SUNDIR> SUN
ln -s <PLACESDIR> Places
ln -s <TexturesDIR> Textures
```
Finally, ensure that your symlinks are not broken.
