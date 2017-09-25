---------------- Require -----------------------
require 'image'
require 'cutorch'
cutorch.setDevice(opt.gpuID) -- goes after cutorch and before nn
require 'nn'
require 'cudnn'
require 'stn'
require 'util/nn/WarpFlowNew'
require 'util/nn/ShuffleTable'
require 'optim'
require 'util/ut'

----------------------- Set cuda ----------------------
cudnn.fastest = true
cudnn.benchmark = true
