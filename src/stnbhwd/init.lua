require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstn'
if withCuda then
   require 'libcustn'
end

require('stn.AffineTransformMatrixGenerator')
require('stn.AffineGridGeneratorBHWD')
require('stn.BilinearSamplerBHWD')
--require('stn.ScaleBHWD')

require('stn.test')

return nn
