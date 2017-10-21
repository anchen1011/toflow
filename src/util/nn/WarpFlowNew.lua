-- Warp image with flow

local WarpFlowNew, parent = torch.class('nn.WarpFlowNew', 'nn.Module')

local function createAddTerm(H, W, squeezed, cuda)

  local addTerm
  if cuda then
    addTerm = torch.CudaTensor(1, 2, H, W)
  else
    addTerm = torch.DoubleTensor(1, 2, H, W):float()
  end
  addTerm[{1, 1,{},{}}] = 
    nn.Replicate(H,1):forward(
      torch.linspace(0, W-1, W))
  addTerm[{1, 2,{},{}}] = 
    nn.Replicate(W,2):forward(
      torch.linspace(0, H-1, H))
  if squeezed then
    addTerm = addTerm:view(2, H, W)
  end

  return addTerm

end

function WarpFlowNew:initFloNet()

  -- Delay the creation of network at first forward.
  -- The good thing about this is that we don't need write a complicate sanitize function
  -- And we don't need to safe the whole network to disk

  self.innerModel = nn.Sequential()
  if self.squeezed then
    self.innerModel:add(nn.ParallelTable():add(nn.Transpose({1,2},{2,3})):add(nn.Transpose({1,2},{2,3})))
      :add(nn.BilinearSamplerBHWD())
      :add(nn.Transpose({2,3},{1,2}))
  else
    self.innerModel:add(nn.ParallelTable():add(nn.Transpose({2,3},{3,4})):add(nn.Transpose({2,3},{3,4})))
      :add(nn.BilinearSamplerBHWD())
      :add(nn.Transpose({3,4},{2,3}))
  end

  local floNet = nn.Sequential()
  floNet:add(nn.CAddTable())
    :add(nn.SplitTable(1,3))
    :add(nn.ConcatTable()
      :add(nn.Sequential()
        :add(nn.SelectTable(2))
        :add(nn.MulConstant(2/(self.h-1)))
        :add(nn.AddConstant(-1))
        :add(nn.Unsqueeze(1,2)))
      :add(nn.Sequential()
        :add(nn.SelectTable(1))
        :add(nn.MulConstant(2/(self.w-1)))
        :add(nn.AddConstant(-1))
        :add(nn.Unsqueeze(1,2))))
    :add(nn.JoinTable(1,3))
    
  self.floNet = floNet:float()
  self.innerModel = self.innerModel:float()

  if self.cuda then
    self.innerModel = self.innerModel:cuda()
    self.floNet = self.floNet:cuda()
  end
end


function WarpFlowNew:__init(squeezed, cuda)
  if squeezed == nil then squeezed = false end
  self.squeezed = squeezed
  self.cuda = cuda
  self.innerModel = nil
  self.floNet = nil
  self.midInput = nil
  self.midGradInput = nil
  self.addTerm = nil
  self.addTermList = {}
  self.h = -1
  self.w = -1
end

function WarpFlowNew:getAddTerm(inputSize)
  if self.squeezed then
    return self.addTerm
  else
    local b = inputSize[1]
    if self.addTermList[b] == nil then
      self.addTermList[b] = torch.expand(self.addTerm,b,2,self.h,self.w) 
    end
    return self.addTermList[b]
  end
end

function WarpFlowNew:type(type,tensorCache)
  if self.innerModel ~= nil then
    self.innerModel = self.innerModel:type(type,tensorCache)
  end
  if self.floNet ~= nil then 
    self.floNet = self.floNet:type(type,tensorCache)
  end
end

function WarpFlowNew:checkSizeChange(inputSize)

  sizeChanged = false
  if self.squeezed then
    if self.h ~= inputSize[2] or self.w ~= inputSize[3] then
      self.h = inputSize[2]
      self.w = inputSize[3]
      sizeChanged = true
    end
  else
    if self.h ~= inputSize[3] or self.w ~= inputSize[4] then
      self.h = inputSize[3]
      self.w = inputSize[4]
      sizeChanged = true
    end
  end
  if sizeChanged or self.addTerm == nil then
    self.addTerm = createAddTerm(self.h, self.w, self.squeezed, self.cuda)
  end
  if sizeChanged or self.addTermList == nil then
    self.addTermList = {}
  end
  if sizeChanged or self.floNet == nil then
    self:initFloNet()
  end
  return sizeChanged

end

function WarpFlowNew:sanitize()

  self.innerModel = nil
  self.floNet = nil
  self.midInput = nil
  self.midGradInput = nil
  self.addTerm = nil
  self.addTermList = nil

end

function WarpFlowNew:updateOutput(input)
  if opt.cuda == nil then
    opt.cuda = false
  end
  if opt.cuda ~= self.cuda then
    self.cuda = opt.cuda
    self:sanitize()
  end
  local inputSize = input[2]:size()
  self:checkSizeChange(inputSize)
  self.midInput = {input[1], self.floNet:forward({input[2], self:getAddTerm(inputSize)})}
  self.output = self.innerModel:updateOutput(self.midInput)
  return self.output
end

-- must call updateOutput before call updateGradInput
function WarpFlowNew:updateGradInput(input, gradOutput)
  local inputSize = input[2]:size()
  assert(not self:checkSizeChange(inputSize)) -- Input size can only change in forward
  self.midGradInput = self.innerModel:updateGradInput(self.midInput, gradOutput)
  self.gradInput = {self.midGradInput[1], 
    self.floNet:updateGradInput({input[2], self:getAddTerm(inputSize)}, self.midGradInput[2])[1]}
  return self.gradInput
end



