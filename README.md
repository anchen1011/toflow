# SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues

This repository contains pre-trained models and demo code for the project 'SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues'

## Prerequisites

#### Torch
We use Torch 7 (http://torch.ch) for our implementation.

#### Cuda
We reply on Cuda (https://developer.nvidia.com/cuda-toolkit) for computation.

#### Matlab [optional]
We use Matlab (https://www.mathworks.com/products/matlab.html) to generate noisy/blur sequences for training/testing.

#### FFmpeg [optional]
We use FFmpeg (http://ffmpeg.org) to generate blocky sequences for training/testing.

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

**mode**: Options include
- 'denoise': video denoising 
- 'deblock': video deblocking
- 'interp': video interpolation
- 'sr': video super-resolution

**inpath**: The input sequence directory.

**outpath**: The location to store the result (data/tmp by default).


## The Vimeo Dataset

#### Triplets

73k frame RGB triplets (73k sequences, each sequence with 3 consecutive frames) from 15k video clips with fixed resolution 448 x 256. This dataset is designed for video interpolation. 

The originals can be downloaded here: link (33G)

The test set can be downloaded here: link (1.7G)

The list of training sequences: data/tri_trainlist.txt

The list of testing sequences: data/tri_testlist.txt

#### Septuplets

92k frame septuplets (92k sequences, each sequence with 7 consecutive frames) from 39k video clips with fixed resolution 448 x 256. This dataset is designed to video denoising, deblocking, and super-resolution.

The originals can be downloaded here: link (82G)

The noisy testing set can be downloaded here: link (16G)

The blur testing set can be downloaded here: link (5G)

The blocky testing set can be downloaded here: link (11G)

The list of training sequences: data/sep_trainlist.txt

The list of testing sequences: data/sep_testlist.txt

#### Poluting Code

The code used to generate noisy/blur sequences is provided under data/pollute

Generate noisy sequences with Matlab
```
noise(input_path);
``` 
Result will be stored under input_path/noisy

Generate blur sequences with Matlab
```
blur(input_path);
```
Result will be stored under input_path/blur

Blocky sequences are compressed by FFmpeg. Our test set is generated with the following configuration:
```sh
ffmpeg -i *.png -q 20 -vcodec jpeg2000 -format j2k name.mov 
```
