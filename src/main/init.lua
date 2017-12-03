---------------- Require -----------------------
require 'image'
if opt.cuda then
  require 'cutorch'
  cutorch.setDevice(opt.gpuID) -- goes after cutorch and before nn
end
require 'nn'
if opt.cuda then
  require 'cudnn'
  require 'cunn'
end
require 'stn'
require 'util/nn/WarpFlowNew'
require 'util/nn/ShuffleTable'
require 'optim'
require 'util/ut'

----------------------- Set cuda ----------------------
if opt.cuda then
  cudnn.fastest = true
  cudnn.benchmark = true
end
