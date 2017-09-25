-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

require 'cunn'
require 'cudnn'
require 'image'
require 'optim'
paths.dofile('Optim.lua')

use_stn = true

-- distorted mnist dataset
paths.dofile('distort_mnist.lua')
datasetTrain, datasetVal = createDatasetsDistorted()

-- model
model = nn.Sequential()
model:add(nn.View(32*32))
model:add(nn.Linear(32*32, 128))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(128, 128))
model:add(cudnn.ReLU(true))
model:add(nn.Linear(128, 10))
model:add(nn.LogSoftMax())

if use_stn then 
   require 'stn'
   paths.dofile('spatial_transformer.lua')
   model:insert(spanet,1)
end

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

optimState = {learningRate = 0.01, momentum = 0.9, weightDecay = 5e-4}
optimizer = nn.Optim(model, optimState)

local w1,w2

for epoch=1,30 do
   model:training()
   local trainError = 0
   for batchidx = 1, datasetTrain:getNumBatches() do
      local inputs, labels = datasetTrain:getBatch(batchidx)
      err = optimizer:optimize(optim.sgd, inputs:cuda(), labels:cuda(), criterion)
      --print('epoch : ', epoch, 'batch : ', batchidx, 'train error : ', err)
      trainError = trainError + err
   end
   print('epoch : ', epoch, 'trainError : ', trainError / datasetTrain:getNumBatches())
   
   model:evaluate()
   local valError = 0
   local correct = 0
   local all = 0
   for batchidx = 1, datasetVal:getNumBatches() do
      local inputs, labels = datasetVal:getBatch(batchidx)
      local pred = model:forward(inputs:cuda())
      valError = valError + criterion:forward(pred, labels:cuda())
      _, preds = pred:max(2)
      correct = correct + preds:eq(labels:cuda()):sum()
      all = all + preds:size(1)
   end
   print('validation error : ', valError / datasetVal:getNumBatches())
   print('accuracy % : ', correct / all * 100)
   print('')
   
   if use_stn then
      w1=image.display({image=spanet.output, nrow=16, legend='STN-transformed inputs, epoch : '..epoch, win=w1})
      w2=image.display({image=tranet:get(1).output, nrow=16, legend='Inputs, epoch : '..epoch, win=w2})
   end
   
end

