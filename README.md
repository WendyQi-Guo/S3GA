# S3GA: Towards Scalable Self-Supervised Learning for Large Scale Graph Alignment

## Directory descriptions

1. **train_partition.py**: trains AAC and TAR in the S3GA framework.
2. **train.py**: trains the self-supervised learning GA for small-scale graph pairs.
3. **dataset**: contains the data loader.
4. **data**: includes the dataset and preprocessing files.
5. **experiments**: contains configuration files for various datasets.
6. **GSSL**: training model for Graph alignment in Self-Supervised Learning (GSSL).
7. **baselines**: the baseline methods we implemented and adapted for unsupervised scenarios.

## Dataset 

The IDS benchmark is provided by [OpenEA](https://github.com/nju-websoft/OpenEA). 
We use the 2.0 version of IDS dataset to avoid name bias issue.

The DBP1M benchmark is provided by [LargeEA](https://github.com/ZJU-DAILY/LargeEA).

First download and unzip dataset files, place them to the project root folder:

    unzip OpenEA_dataset_v2.0.zip
    unzip mkdata.zip
