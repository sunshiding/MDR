# MDR

A PyTorch implementation of *Multi-View Disentangled Representation*.

More details will be gradually shown later.

## Setup/Installation

Open a new conda environment and install the necessary dependencies. 

```
conda create -n mdr python=3.7 anaconda
# activate the environment
source activate mdr

conda install numpy
conda install pytorch torchvision -c pytorch
```

MNIST-CDCB Dataset can be obtained from [MNIST-CD/CB](http://www.cvc.uab.es/lamp/wp-content/shared_files/cross-domain-disen/MNIST-CDCB-256Ims.tar.gz).
CelebA-related datasets can be downloaded from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Example Experiments

This repository contains a subset of the experiments mentioned in the paper. In each folder, there are 3 scripts that one can run: `run.py` to fit the MDR; `sample.py` to (conditionally) reconstruct from samples in the latent space; and `model.py` to build the model. And more code will be released gradually.

### Qualitative Analysis on the MNIST-CDCB dataset

The detailed code is in the folder MNIST.

Assuming the path of the dataset is `./MNIST/datasets/MNIST`

You can run the code with the following steps:

```
cd MNSIT
CUDA_VISIBLE_DEVICES=0 python run.py --cuda
CUDA_VISIBLE_DEVICES=0 python sample.py
```
