require 'nn'

local M = {}

function M.calVecDiff(v1, v2)

  local sum = 0
  local num = 0
  if type(v1) == 'table' then
    for i = 1, #v1 do
      local suc, tsum, tnum = calVecDiff(v1[i], v2[i])
      if not suc then
        return false,nil,nil
      end
      sum = sum + tsum
      num = num + tnum
    end
  else
    if v1:nElement() ~= v2:nElement() then
      return false,nil,nil
    else
      sum = sum + (v1 - v2):abs():sum()
      num = num + v1:nElement()
    end
  end
  return true, sum, num

end

function M.checkNetworkDiff(m1, m2, index)

  if m1.modules == nil then
    local suc, sum, num = M.caldiff(m1.output, m2.output)
    if not suc or sum / num > 0.01 then
      return true
    end
  else
    local tlen = #index+1
    table.insert(index, -1)
    for i = 1,#m1.modules do
      index[tlen] = i
      if M.checkNetworkDiff(m1.modules[i], m2.modules[i], index) then
        return true 
      end
    end
    index[tlen] = nil
  end
  return false

end

function M.addLinearModel(net, linstates, params)

  if params == nil then
    params = {}
  end
  if params.lastReLU == nil then
    params.lastReLU = false
  end
  if params.batchnorm == nil then
    params.batchnorm = false
  end
  local nlayer = #linstates - 1

  for stage = 1,nlayer do
    local instate = linstates[stage]
    local outstate = linstates[stage+1]
    net:add(nn.Linear(instate, outstate))
    if params.lastReLU or (stage ~= nlayer) then
        net:add(nn.ReLU(true))
    end
    if params.batchnorm and (stage ~= nlayer) and (outstate ~= 1) then
      net:add(nn.BatchNormalization(outstate))
    end
  end

end

function M.createLinearModel(linstates, params)
  local net = nn.Sequential()
  M.addLinearModel(net, linstates, params)
  return net
end

function M.createConvModel2(convstates, filtsize, params)

  if params == nil then
    params = {}
  end
  if params.lastReLU == nil then
    params.lastReLU = false
  end
  if params.batchnorm == nil then
    params.batchnorm = false
  end
  if params.leakyReLU == nil then
    params.leakyReLU = -1 -- no leakyReLU
  end
  if params.reweight == nil then
    params.reweight = -1
  end
  if params.res == nil then
    params.res = false
  end

  -- {v41, v42, ...} ==> W
  local net = nn.Sequential()
  for i = 1,#filtsize do
    local fs = filtsize[i]
    local instate = convstates[i]
    local outstate = convstates[i+1]
    pad = (fs - 1) / 2
    local convLayer = cudnn.SpatialConvolution(instate, outstate, fs, fs, 1, 1, pad, pad)
    if params.reweight > 0 then
      convLayer.weight:mul(params.reweight)
      convLayer.bias:mul(params.reweight)
    end
    local tnet = nn.Sequential()
    tnet:add(convLayer)
    if params.lastReLU or (i ~= #filtsize) then
      if params.batchnorm then
        tnet:add(nn.SpatialBatchNormalization(outstate))
      end
      if params.leakyReLU > 0 then
        tnet:add(nn.LeakyReLU(leakyReLU, true))
      else
        tnet:add(nn.ReLU(true))
      end
    end
    if params.res and instate == outstate then
      net:add(nn.ConcatTable()
           :add(nn.Identity())
           :add(tnet))
         :add(nn.CAddTable())
    else
      for i = 1,#tnet.modules do
        net:add(tnet:get(i))
      end
    end
  end
  return net

end

-- This function is deprecated, please use createConvModel2
function M.createConvModel(convstates, filtsize, batchnorm, lastReLU, leakyReLU)

  if leakyReLU == nil then
    leakyReLU = -1 -- no leakyReLU
  end

  -- {v41, v42, ...} ==> W
  local net = nn.Sequential()
  for i = 1,#filtsize do
    local fs = filtsize[i]
    local instate = convstates[i]
    local outstate = convstates[i+1]
    pad = (fs - 1) / 2
    net:add(cudnn.SpatialConvolution(instate, outstate, fs, fs, 1, 1, pad, pad))
    if lastReLU or (i ~= #filtsize) then
      if batchnorm then
        net:add(nn.SpatialBatchNormalization(outstate))
      end
      if leakyReLU > 0 then
        net:add(nn.LeakyReLU(leakyReLU, true))
      else
        net:add(nn.ReLU(true))
      end
    end
  end
  return net

end


function M.initModel(model, weight)
  ut.nn.convInit(model, 'cudnn.SpatialConvolution', weight)
  ut.nn.convInit(model, 'nn.SpatialConvolution', weight)
  ut.nn.BNInit(model, 'fbnn.SpatialBatchNormalization')
  ut.nn.BNInit(model, 'cudnn.SpatialBatchNormalization')
  ut.nn.BNInit(model, 'nn.SpatialBatchNormalization')
  for k, v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
  end
end


function M.convInit(model, name, weight)
  if weight == nil then weight = 2 end
  for k, v in pairs(model:findModules(name)) do
    local n = v.kW * v.kH * v.nOutputPlane
    v.weight:normal(0, math.sqrt(weight / n))
    v.bias:zero()
  end
  return model
end

function M.convInitIdentity(model, val, biasVal)
  x = (model.kW + 1) / 2
  y = (model.kH + 1) / 2
  nc = math.min(model.nInputPlane, model.nOutputPlane)
  model.weight:zero()
  for i = 1,nc do
    model.weight[{i,i,x,y}] = val
  end
  if biasVal == nil then
  model.bias:zero()
  else
    model.bias[{{}}] = biasVal
  end
  return model
end

function M.BNInit(model, name)
  for k, v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:zero()
  end
  return model
end

function M.createShellModel(model)
  local modelOut = {}

  modelOut.innerModel = model

  function modelOut:forward(input)
    return self.innerModel:forward(input)
  end

  function modelOut:backward(input, gradOutput)
    return self.innerModel:backward(input, gradOutput)
  end

  function modelOut:cuda()
    self.innerModel:cuda()
    return model
  end

  function modelOut:double()
    self.innerModel:double()
    return model
  end

  function modelOut:sanitize()
    M.sanitize(self.innerModel)
  end

  function modelOut:evaluate()
    self.innerModel:evaluate()
    return model
  end

  function modelOut:training()
    self.innerModel:training()
    return model
  end

  function modelOut:getParameters()
    return self.innerModel:getParameters()
  end

  function modelOut:parameters()
    return self.innerModel:parameters()
  end

  return modelOut
end

function M.sanitizeLayer(layer)

  for name, field in pairs(layer) do
    if torch.type(field) == 'cdata' then layer[name] = nil end
    if name == 'homeGradBuffers' then layer[name] = nil end
    if name == '_input' then layer[name] = nil end
    if name == '_gradOutput' then layer[name] = nil end
    if name == 'batchsize' then layer[name][1] = 0 end

    if (name == 'input' or name == 'input_gpu' or
        name == 'gradOutput' or name == 'gradOutput_gpu' or 
        name == 'finput' or  name == 'fgradInput' or
        name == 'foutput' or  name == 'fgradOutput' or
        name == 'workspace') then
       layer[name] = nil
    end
    if (name == 'output' or name == 'output_gpu' or
        name == 'gradInput' or name == 'gradInput_gpu' or
        name == 'output_slice' or name == 'input_slice') then 
      if torch.type(field) == 'table' then
        layer[name] = {}
      else
        layer[name] = field.new()
      end
    end
    
    if name == 'buffer' or name == 'buffer2' or name == 'normalized'
      or name == 'centered' or name == 'addBuffer' then         
      layer[name] = nil
    end
  end

  if layer.sanitize ~= nil then
    layer:sanitize()
  end

end

function M.sanitize(net)

  if net.sanitize ~= nil then
    net:sanitize()
    return net
  end

  local list = net:listModules()
  for nameL, val in ipairs(list) do
    M.sanitizeLayer(val)
  end

  return net

end

function M.addOutputBack(net)
  
  local list = net:listModules()
  local cudaif = net:type() == 'torch.CudaTensor'
  for nameL, val in ipairs(list) do
    if cudaif then
      val.output = torch.CudaTensor()
    else
      val.output = torch.DoubleTensor()
    end
  end

end

function M.sanitizeCriterion(criterion)
  if criterion.__typename == 'nn.MultiCriterion' or 
      criterion.__typename == 'nn.ParallelCriterion' then
    criterion.gradInput = {}
  else
   if criterion.gradInput ~= nil then
     if type(criterion.gradInput) == 'table' then
       criterion.gradInput = {}
     elseif criterion.gradInput:type() == 'torch.CudaTensor' then
       criterion.gradInput = torch.CudaTensor()
     else
       criterion.gradInput = torch.DoubleTensor()
     end
   end
  end

  if criterion.__typename == 'nn.MultiCriterion' or 
      criterion.__typename == 'nn.ParallelCriterion' then
    for i = 1, #criterion.criterions do
      M.sanitizeCriterion(criterion.criterions[i])
    end
  end
end

-- A multi-concat function.  
-- Replaces the 'concat' in torch, which can't deal with cuda tensors
function M.concatTensors (tensors, outputDimension)
  local nTensors = table.getn(tensors)

  local sumOutputSizes = 0
  for iTensor = 1,nTensors do
    sumOutputSizes = sumOutputSizes + tensors[iTensor]:size(outputDimension)
  end

  local outputSize = tensors[1]:size()
  outputSize[outputDimension] = sumOutputSizes

  -- We clone and then resize to make sure it's the right kind of tensor.
  local res = tensors[1]:clone()
  res:resize(outputSize)

  local curOutputOffset = 1
  for iTensor = 1,nTensors do
    local accessor = {}
    for j = 1,outputSize:size() do
      accessor[j] = {}
    end

    local outputDimSize = tensors[iTensor]:size(outputDimension)
    accessor[outputDimension] = {curOutputOffset, curOutputOffset + outputDimSize - 1}
    res[accessor]:copy(tensors[iTensor])
    curOutputOffset = curOutputOffset + outputDimSize
  end

  return res
end


function M.createWarpModel()

  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local imgOut = nn.Transpose({2,3},{3,4})(imgData)
  local floOut = nn.Transpose({2,3},{3,4})(floData)

  local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWDSpyNet()({imgOut, floOut}))
  local model = nn.gModule({imgData, floData}, {warpImOut})

  return model
  
end

function M.createSqueezeWarpModel()

  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local imgOut = nn.Transpose({1,2},{2,3})(imgData)
  local floOut = nn.Transpose({1,2},{2,3})(floData)

  local warpImOut = nn.Transpose({2,3},{1,2})(nn.BilinearSamplerBHWDSpyNet()({imgOut, floOut}))
  local model = nn.gModule({imgData, floData}, {warpImOut})

  return model

end

function M.cudanize(x)
  if x == nil then
    return nil
  elseif type(x) ~= 'table' then
    return x:cuda()
  else
    -- This is a recursive function
    -- To make sure it does not depends on any global
    -- variables, we define a local function and call it
    -- recursively
    --
    local function cudanizeL(y)
      if type(y) == 'table' then
        local list = {}
        for name, field in pairs(y) do
          list[name] = cudanizeL(field)
        end
        return list
      else
        return y:cuda()
      end
    end
    return cudanizeL(x)
  end    
end

function M.doublize(x)
  if x == nil then
    return nil
  elseif type(x) ~= 'table' then
    return x:double()
  else
    local function doublizeL(y)
      if type(y) == 'table' then
        local list = {}
        for name, field in pairs(y) do
          list[name] = doublizeL(field)
        end
        return list
      else
        return y:double()
      end
    end
    return doublizeL(x)
  end
end

function M.floatize(x)
  if x == nil then
    return nil
  elseif type(x) ~= 'table' then
    return x:float()
  else
    local function floatizeL(y)
      if type(y) == 'table' then
        local list = {}
        for name, field in pairs(y) do
          list[name] = floatizeL(field)
        end
        return list
      else
        return y:float()
      end
    end
    return floatizeL(x)
  end
end


function M.tensorDimsStr (A)
  if torch.isTensor(A) then
    local tmp = A:size(1)
    for iDim = 2,A:nDimension() do
      tmp = tmp .. ' x ' .. A:size(iDim)
    end
    return tmp
  else
    local tmp = 'Length ' .. #A .. ' Table\n'
    for i = 1, #A do
      tmp = tmp .. 'Table[' .. i ..']: ' .. M.tensorDimsStr(A[i]) .. '\n'
    end
    return tmp
  end
end


function M.dumpNetwork(layer, inputData, prefix)
  prefix = prefix or ''
  local prefixExtension = "    "
  local output
  local strLayer = tostring(layer)
  if (strLayer:sub(1,13) == 'nn.Sequential') then
    local nLayers = layer:size()
    print (prefix .. 'Layer type: nn.Sequential (' .. nLayers .. ')')
    print (prefix .. 'Input: ' .. M.tensorDimsStr(inputData))
    local layerInput = inputData
    for iLayer = 1,nLayers do
      print (prefix .. 'Sequential layer ' .. iLayer)
      local curLayer = layer:get(iLayer)
      local res = M.dumpNetwork (curLayer, layerInput, prefix .. prefixExtension)
      layerInput = res
    end

    output = layerInput
  elseif (strLayer:sub(1,16) ~= "nn.ParallelTable" and strLayer:sub(1,11) == "nn.Parallel") then
    local nLayers = table.getn(layer.modules)
    print (prefix .. 'Layer type: nn.Parallel (' .. nLayers .. ')')
    local inputDimension = layer.inputDimension
    local outputDimension = layer.outputDimension
    print (prefix .. 'Split on ' .. inputDimension)
    print (prefix .. 'Input: ' .. M.tensorDimsStr(inputData))

    local layerRes = {}
    local sumOutputSizes = 0
    for iLayer = 1,nLayers do
      print (prefix .. 'Parallel layer ' .. iLayer)
      local curLayer = layer:get(iLayer)
      local curInput = inputData:select(inputDimension, iLayer)
      local res = M.dumpNetwork (curLayer, curInput, prefix .. prefixExtension)
      layerRes[iLayer] = res
    end

    output = M.concatTensors (layerRes, outputDimension)
  else
    print (prefix .. 'Layer type: ' .. strLayer)
    print (prefix .. 'Input: ' .. M.tensorDimsStr(inputData))
    output = layer:forward(inputData)
  end
  if torch.isTensor(output) and output:ne(output):sum() > 0 then
    print( prefix .. '!!!!!!!!!!!!!!!!!!!!!!! Found NaN in output !!!!!!!!!!!!!!!!!!!!!!!')
  end

  print (prefix .. 'Output: ' .. M.tensorDimsStr(output))
  return output
end

function M.noNAN(y)

    return y:ne(y):sum()

end

function M.noInvalid(y)

    return y:ne(y):sum() + y:eq(1/0):sum() + y:eq(-1/0):sum()

end

-- Forward a network. 
-- suc, output, gradInput, input_subnet = tryForward(model, x)
-- If it successed, return true, output
-- If it failed, return: false, modelindexlist, net, input_to_that_net
function M.tryForward(model, input, level)
  if type(model) == 'table' and model.innerModel ~= nil and
      torch.type(model) == 'nn.ShellModel' then
    print('Ignoring the network: ', torch.type(model))
    return M.tryForward(model.innerModel, input, level)
  end

  if level == nil then level = 1 end
  local suc,output,subnet,input_subnet,tdata

  local model_type = torch.type(model)
  if model_type == 'nn.Sequential' then
      local nLayer = #model.modules
      tdata = input
      for i=1,nLayer do
          suc,tdata,subnet,input_subnet = M.tryForward(model.modules[i],tdata,level+1)
          if not suc then
              output = tdata
              output[level] = i
              return false,output,subnet,input_subnet
          end
      end
      output = tdata
  elseif model_type == 'nn.ConcatTable' then
      local nLayer = #model.modules
      output = {}
      for i=1,nLayer do
          suc,tdata,subnet,input_subnet = M.tryForward(model.modules[i],input,level+1)
          if not suc then
              output = tdata
              output[level] = i
              return false,output,subnet,input_subnet
          else
              table.insert(output,tdata)
          end
      end
  elseif model_type == 'nn.ParallelTable' then
      local nLayer = #model.modules
      output = {}
      if type(input)~='table' or #input ~= nLayer then
          return false,{},model,input
      end
      for i=1,nLayer do
          suc,tdata,subnet,input_subnet = M.tryForward(model.modules[i],input[i],level+1)
          if not suc then
              output = tdata
              output[level] = i
              return false,output,subnet,input_subnet
          else
              table.insert(output,tdata)
          end
      end
  else
      if not pcall(function() output = model:forward(input) end) then
          return false,{},model,input
      end
  end
  return true,output,nil,nil

end

-- Forward a network in a memory efficient way
function M.memFriendlyForward(model, input)

  if type(model) == 'table' and model.innerModel ~= nil and 
      string.sub(torch.type(model),0,3) ~= 'nn.' then
    print('Ignoring the network: ', torch.type(model))
    return M.memFriendlyForward(model.innerModel, input, level)
  end

  local output
  local model_type = torch.type(model)
  if model_type == 'nn.Sequential' then
    local nLayer = #model.modules
    output = input
    for i=1,nLayer do
      output = M.memFriendlyForward(model.modules[i],output)
      if i ~= 1 then
        M.sanitize(model.modules[i-1])
      end
    end
  elseif model_type == 'nn.ConcatTable' then
    local nLayer = #model.modules
    output = {}
    for i=1,nLayer do
      output[i] = M.memFriendlyForward(model.modules[i],input)
    end
  elseif model_type == 'nn.ParallelTable' then
    local nLayer = #model.modules
    output = {}
    for i=1,nLayer do
      output[i] = M.memFriendlyForward(model.modules[i],input[i])
    end
  else
    if not pcall(function() output = model:updateOutput(input) end) then
      collectgarbage()
      output = model:updateOutput(input)
    end
  end
  return output

end

local function printObj(obj)
  if type(obj) == 'table' then
    print(obj)
  else
    print(obj:size())
  end
end

-- Backward a network. 
-- suc, [gradInput,modelIdx], subnet, inputSubnet, gradOutputSubnet
--   = tryForward(model, x)
-- If it successed, return true, gradInput 
-- If it failed, return: false, modelindexlist, net, input_to_that_net
--
-- This function does not provide correct backward propagation result!!!
-- It should only be used to check whether the network is correctly desigend
--
function M.tryBackward(model, input, gradOutput, level)

  if type(model) == 'table' and model.innerModel ~= nil then
    return M.tryBackward(model.innerModel, input, gradOutput, level)
  end

  if level == nil then level = 1 end
  local suc,gradInput,subnet,inputSubnet,gradOutputSubnet,tdata

  local model_type = torch.type(model)
  if model_type == 'nn.Sequential' then
    local nLayer = #model.modules
    gradInput = gradOutput
    for i=nLayer,1,-1 do
      local tinput
      if i==1 then
        tinput = input
      else
        tinput = model.modules[i-1].output
      end
      suc,gradInput,subnet,inputSubnet,gradOutputSubnet =
        M.tryBackward(model.modules[i],tinput,gradInput,level+1)
      if not suc then
        gradInput[level] = i
        return false,gradInput,subnet,inputSubnet,gradOutputSubnet
      end
    end
  elseif model_type == 'nn.ConcatTable' then
    -- This part actually does not provide a correct back-prop result.
    -- We do this just to make the implementation easier
    local nLayer = #model.modules
    for i=1,nLayer do
      suc,gradInput,subnet,inputSubnet,gradOutputSubnet =
        M.tryBackward(model.modules[i],input,gradOutput[i],level+1)
      if not suc then
        gradInput[level] = i
        return false,gradInput,subnet,inputSubnet,gradOutputSubnet
      end
    end
  elseif model_type == 'nn.ParallelTable' then
    local nLayer = #model.modules
    if type(input)~='table' or #input ~= nLayer then
      return false,{},model,input,gradOutput
    end
    gradInput = {}
    local tdata
    for i=1,nLayer do
      suc,tdata,subnet,inputSubnet,gradOutputSubnet = 
        M.tryBackward(model.modules[i],input[i],gradOutput[i],level+1)
      if not suc then
        gradInput = tdata
        gradInput[level] = i
        return false,gradInput,subnet,inputSubnet,gradOutputSubnet
      else
        table.insert(gradInput,tdata)
      end
    end
  else
    suc = pcall(function() gradInput = model:backward(input,gradOutput) end)
    if not suc then
      print(string.format('Error catched in %s', torch.type(model)))
      printObj(input)
      return false,{},model,input,gradOutput
    end
  end
  return true,gradInput,nil,nil,nil

end
return M
