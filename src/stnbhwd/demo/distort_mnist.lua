-- wget 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'
-- tar -xf mnist.t7.tgz

function distortData(foo)
   local res=torch.FloatTensor(foo:size(1), 1, 42, 42):fill(0)
   for i=1,foo:size(1) do
      baseImg=foo:select(1,i)
      distImg=res:select(1,i)
      
      r = image.rotate(baseImg, torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 42-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
   end
   return res
end

function distortData32(foo)
   local res=torch.FloatTensor(foo:size(1), 1, 32, 32):fill(0)
   local distImg=torch.FloatTensor(1, 42, 42):fill(0)
   for i=1,foo:size(1) do
      baseImg=foo:select(1,i)
     
      r = image.rotate(baseImg, torch.uniform(-3.14/4,3.14/4))
      scale = torch.uniform(0.7,1.2)
      sz = torch.floor(scale*32)
      s = image.scale(r, sz, sz)
      rest = 42-sz
      offsetx = torch.random(1, 1+rest)
      offsety = torch.random(1, 1+rest)
      
      distImg:zero()
      distImg:narrow(2, offsety, sz):narrow(3,offsetx, sz):copy(s)
      res:select(1,i):copy(image.scale(distImg,32,32))
   end
   return res
end

function createDatasetsDistorted()
   local testFileName = 'mnist.t7/test_32x32.t7'
   local trainFileName = 'mnist.t7/train_32x32.t7'
   local train = torch.load(trainFileName, 'ascii')
   local test = torch.load(testFileName, 'ascii')
   train.data = train.data:float()
   train.labels = train.labels:float()
   test.data = test.data:float()
   test.labels = test.labels:float()
   
   -- distortion   
   train.data = distortData32(train.data)
   test.data = distortData32(test.data)

   local mean = train.data:mean()
   local std = train.data:std()
   train.data:add(-mean):div(std)
   test.data:add(-mean):div(std)
   
   local batchSize = 256
   
   local datasetTrain = {
      getBatch = function(self, idx)
         local data = train.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         local labels = train.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         return data, labels, batchSize
      end,
      getNumBatches = function()
         return torch.floor(60000 / batchSize)
      end
   }
   
   local datasetVal = {
      getBatch = function(self, idx)
         local data = test.data:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         local labels = test.labels:narrow(1, (idx - 1) * batchSize + 1, batchSize)
         return data, labels, batchSize
      end,
      getNumBatches = function()
         return torch.floor(10000 / batchSize)
      end
   }
   
   return datasetTrain, datasetVal
end