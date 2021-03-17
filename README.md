# Generating Image Captions

## Background
* Paper: [Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks](https://arxiv.org/abs/2004.06165)
* Original Source Code: [github/Oscar](https://github.com/microsoft/Oscar)
* Original README: [README.md](README_original.md)

There are two parts to this project. `Oscar` is the model that generates captions
using transformers and bottom-up-attention.
In the source code they have used Faster-RCNN with Visual Genome tags to extract
ROI features and object tags. However, they do not include this in their source
code. 

## Hardware
* Oscar (currently) needs a GPU.
* All the installation instructions below have been tested using a
**g4dn.4xlarge** instance with an Ubuntu 18.04 Deep 
  Learning AMI (`ami-0e5d3cb86ff6f2dcb`). You will need a large storage drive - 
  as these are cheap I'd go for 1000GB.

## Software
* Ubuntu 18.04
* anaconda
* Python 3.6(.13)
* CUDA 10.0 (????)

## Installation

### Installation from BradleyGrantham modifications
**Recommended**. There was a bunch of things that had to be changed to get this 
working.

1. Clone the modified `Oscar` repository. This has to use `--recursive` option because of the subpackages within the repo.
```shell
git clone --recursive git@github.com:BradleyGrantham/Oscar.git
```

2. Download the COCO dataset (optional but recommended as it's the quickest way to get results/check things are working)
```shell
hello
```




### Installation from source
1. Clone the `Oscar` repository. This has to use `--recursive` option because of the subpackages within the repo.
```
git clone --recursive git@github.com:microsoft/Oscar.git
```




