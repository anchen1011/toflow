# SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues

This repository contains pre-trained models and demo code for the project 'SSFlow: Self-Supervised Video Enhancement with Task-Oriented Motion Cues'

## Prerequisites

#### Torch
We use Torch 7 (http://torch.ch) for our implementation.

#### Cuda [optional]
Cuda is suggested (https://developer.nvidia.com/cuda-toolkit) for computation.

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
This will install 'stn' package for Lua. The list of components:
```lua
require 'stn'
nn.AffineGridGeneratorBHWD(height, width)
-- takes B x 2 x 3 affine transform matrices as input, 
-- outputs a height x width grid in normalized [-1,1] coordinates
-- output layout is B,H,W,2 where the first coordinate in the 4th dimension is y, and the second is x
nn.BilinearSamplerBHWD()
-- takes a table {inputImages, grids} as inputs
-- outputs the interpolated images according to the grids
-- inputImages is a batch of samples in BHWD layout
-- grids is a batch of grids (output of AffineGridGeneratorBHWD)
-- output is also BHWD
nn.AffineTransformMatrixGenerator(useRotation, useScale, useTranslation)
-- takes a B x nbParams tensor as inputs
-- nbParams depends on the contrained transformation
-- The parameters for the selected transformation(s) should be supplied in the
-- following order: rotationAngle, scaleFactor, translationX, translationY
-- If no transformation is specified, it generates a generic affine transformation (nbParams = 6)
-- outputs B x 2 x 3 affine transform matrices
```

#### Download pretrained models (104MB) 
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

**cuda**: Whether Cuda is used.

**gpuId**: GPU device ID.

**mode**: Options include
- 'denoise': video denoising 
- 'deblock': video deblocking
- 'interp': video interpolation
- 'sr': video super-resolution

**inpath**: The input sequence directory.

**outpath**: The location to store the result (data/tmp by default).


## The Vimeo Dataset

#### Triplets

73171 RGB frame triplets (73k sequences, each sequence with 3 consecutive frames) from 15k video clips with fixed resolution 448 x 256. This dataset is designed for video interpolation. 

The originals can be downloaded [here](http://data.csail.mit.edu/tofu/dataset/vimeo_tri.zip). (33G)

The test set can be downloaded [here](http://data.csail.mit.edu/tofu/testset/vimeo_tri_test.zip). (1.7G)

The list of training sequences: data/tri_trainlist.txt

The list of testing sequences: data/tri_testlist.txt

#### Septuplets

91701 RGB frame septuplets (92k sequences, each sequence with 7 consecutive frames) from 39k video clips with fixed resolution 448 x 256. This dataset is designed to video denoising, deblocking, and super-resolution.

The originals can be downloaded [here](http://data.csail.mit.edu/tofu/dataset/vimeo_sep.zip). (82G)

The noisy testing set can be downloaded [here](http://data.csail.mit.edu/tofu/testset/vimeo_sep_noisy.zip). (16G)

The blur testing set can be downloaded [here](http://data.csail.mit.edu/tofu/testset/vimeo_sep_blur.zip). (5G)

The low resolution testing set can be downloaded [here](http://data.csail.mit.edu/tofu/testset/vimeo_sep_low.zip). (649M)

The blocky testing set can be downloaded [here](http://data.csail.mit.edu/tofu/testset/vimeo_sep_block.zip). (11G)

The list of training sequences: data/sep_trainlist.txt

The list of testing sequences: data/sep_testlist.txt

#### Generate Testing Sequences

The code used to generate noisy/blur sequences is provided under src/generate_testing_sample

Generate noisy sequences with Matlab under src/generate_testing_sample
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

#### Download the dataset (115G) [optional]
```sh
./download_dataset.sh
``` 

#### Download the testset (35G) 
```sh
./download_testset.sh
``` 

#### Run test code
```sh
cd src
th demo_vimeo.lua -mode denoise
th demo_vimeo.lua -mode interp
th demo_vimeo.lua -mode sr
th demo_vimeo.lua -mode deblock
```

#### Evaluate

The code used to evaluate results in PSNR, SSIM, Abs metrics is provided under src/evaluation

Evaluate results with Matlab under src/evaluation
```
evaluate([output_root='../../output']);
``` 

Results will be returned by the function and printed to the screen.

It is assumed that our datasets are unzipped under data/ and not renamed. It is also assumed that your results are put under [output_root]/[task_name] e.g. result/sr result/interp result/denoise result/deblock. 

You could further modify src/evaluation/get_tasks.m to specify what you are evaluating and where is the location. 

## References
1. Our warping code is based on [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd).
2. Our flow and transformation utilities are based on [anuragranj/spynet](https://github.com/anuragranj/spynet)
