# stnbhwd

## Main modules

These are the basic modules (BHWD layout) needed to implement a Spatial Transformer Network (Jaderberg et al.) http://arxiv.org/abs/1506.02025

``` lua
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
```

## Advanced module

This module allows the user to put a constraint on the possible transformations.
It should be placed between the localisation network and the grid generator.

``` lua
require 'stn'

nn.AffineTransformMatrixGenerator(useRotation, useScale, useTranslation)
-- takes a B x nbParams tensor as inputs
-- nbParams depends on the contrained transformation
-- The parameters for the selected transformation(s) should be supplied in the
-- following order: rotationAngle, scaleFactor, translationX, translationY
-- If no transformation is specified, it generates a generic affine transformation (nbParams = 6)
-- outputs B x 2 x 3 affine transform matrices
```


If this code is useful to your research, please cite this repository.
