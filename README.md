# SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues

This repository contains pre-trained models and demo code for the project 'SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues'

## Prerequisites

#### Torch
We use Torch 7 (http://torch.ch) for our implementation.

#### Cuda
We reply on Cuda (https://developer.nvidia.com/cuda-toolkit) for computation.

## Installation
Our current release has been tested on Ubuntu 14.04.

#### Clone the repository
```sh
git clone https://github.com/anchen1011/ssflow.git
```

#### Install dependency
```sh
cd ssflow/src/stnbhwd
luarocks make
```

#### Download pretrained models (53MB) 
```sh
cd ../../
./download_models.sh
``` 

#### Run test code
```sh
cd src
th demo.lua
```

There are a few options in demo.lua:

**gpuId**: GPU device ID

**mode**: Set it to 'denoise' to run denoising algorithm. Set it to 'deblock' to run deblocking algorithm. Set it to 'interp' to run interpolation algorithm. Set it to 'sr' to run super-resolution algorithm (bicubic pre-upscale required).

**inpath**: The input sequence directory.

**outpath**: The location to store the result (data/tmp) by default.
