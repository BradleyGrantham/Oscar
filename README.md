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
* Python 3.7(.10)
* CUDA 10.0(.130)

## Installation

### Installation from BradleyGrantham modifications
**Recommended**. There was a bunch of things that had to be changed to get this 
working.

1. Clone the modified `Oscar` repository. This has to use `--recursive` option because of the subpackages within the repo.
```shell
git clone --recursive git@github.com:BradleyGrantham/Oscar.git
```

2. Download the COCO dataset (optional but recommended as it's the quickest way to get results/check things are working).
*You probably want to use [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy) for this. Download and untar.*
```shell
cd Oscar
mkdir datasets
cd datasets

path/to/azcopy copy https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip .

# or wget https://biglmdiag.blob.core.windows.net/oscar/datasets/coco_caption.zip
unzip coco_caption.zip
```

3. Download the pretrained Oscar model (this model is finetuned to image captioning and has been trained using ROIs from Faster-RCNN, with ResNet-101, and Visual Genome tags.)
```shell
cd Oscar  # cd back to the root of the Oscar repo
mkdir models
cd models
wget https://biglmdiag.blob.core.windows.net/oscar/exp/coco_caption/base/checkpoint.zip
unzip checkpoint.zip
```

4. Create a conda environment and install the requirements
```shell
conda create --name oscar python=3.7
conda activate oscar

# install pytorch
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch

# install apex
cd Oscar  # cd back to the root of the Oscar repo
git clone https://github.com/NVIDIA/apex.git
cd apex

pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# install Oscar
cd ..  # cd back to the root of the Oscar repo
cd coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

pip install -r requirements.txt
```

5. Install the below fork of [detectron2](https://github.com/facebookresearch/detectron2) to allow extraction of Faster-RCNN features and Visual Genome object tags.
```shell
cd ~
git clone git@github.com:BradleyGrantham/py-bottom-up-attention.git

cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
pip install -e .

# or if you want to run on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

## Running on the COCO caption dataset
If you followed step 2 above, you can now get results on the COCO caption test set 
with the following.
```shell
cd Oscar  # cd back to the root of the Oscar repo

python oscar/run_captioning.py \
    --do_test \
    --do_eval \
    --test_yaml test.yaml \
    --per_gpu_eval_batch_size 64 \
    --num_beams 5 \
    --max_gen_length 20 \
    --eval_model_dir models/checkpoint-29-66420/
```

You should see the same results (or *very, very* similar results) to below:
```shell
Bleu_1: 0.756
Bleu_2: 0.601
Bleu_3: 0.469
Bleu_4: 0.366
METEOR: 0.304
ROUGE_L: 0.586
CIDEr: 1.241
```

## Running on your own custom images

1. Create a directory containing your images. E.g.
```shell
cd ~
ls my-images
# IMG_0010.JPG  IMG_0012.JPG  IMG_2837.JPG  IMG_2874.JPG  IMG_2911.JPG  IMG_3053.JPG  IMG_4444.JPG
```

2. Run bottom up attention to extract the features for Oscar 
   (it may take a while on the first run as it has to download the model weights)
```shell
cd py-bottom-up-attention/

python demo/extract_oscar_features.py ~/my-images/
```
This command outputs the necessary feature files to a directory named `custom` within
your current working directory. The output directory can be changed using command line 
options.

In this example the files get output to `py-bottom-up-attention/custom/`

3. Point Oscar to your custom features.
```shell
cd ~/Oscar # cd back to the root of the Oscar repo

python oscar/run_captioning.py \
      --do_test \
      --do_eval \
      --test_yaml custom.yaml \
      --per_gpu_eval_batch_size 64 \
      --num_beams 5 \
      --max_gen_length 20 \
      --eval_model_dir models/checkpoint-29-66420/ \
      --data_dir ~/py-bottom-up-attention/custom/
```

Results saved under `--eval_model_dir models/checkpoint-29-66420/`

4. Visualise the results using
```shell
cd py-bottom-up-attention/

python demo/visualise_oscar_captions.py ~/my-images/ ~/Oscar/models/checkpoint-29-66420/resuls.tsv
```


