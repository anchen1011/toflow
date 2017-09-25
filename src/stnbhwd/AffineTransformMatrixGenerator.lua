local ATMG, parent = torch.class('nn.AffineTransformMatrixGenerator', 'nn.Module')

--[[
AffineTransformMatrixGenerator(useRotation, useScale, useTranslation) :
AffineTransformMatrixGenerator:updateOutput(transformParams)
AffineTransformMatrixGenerator:updateGradInput(transformParams, gradParams)

This module can be used in between the localisation network (that outputs the
parameters of the transformation) and the AffineGridGeneratorBHWD (that expects
an affine transform matrix as input).

The goal is to be able to use only specific transformations or a combination of them.

If no specific transformation is specified, it uses a fully parametrized
linear transformation and thus expects 6 parameters as input. In this case
the module is equivalent to nn.View(2,3):setNumInputDims(2).

Any combination of the 3 transformations (rotation, scale and/or translation)
can be used. The transform parameters must be supplied in the following order:
rotation (1 param), scale (1 param) then translation (2 params).

Example:
AffineTransformMatrixGenerator(true,false,true) expects as input a tensor of
if size (B, 3) containing (rotationAngle, translationX, translationY).
]]

function ATMG:__init(useRotation, useScale, useTranslation)
  parent.__init(self)

  -- if no specific transformation, use fully parametrized version
  self.fullMode = not(useRotation or useScale or useTranslation)

  if not self.fullMode then
    self.useRotation = useRotation
    self.useScale = useScale
    self.useTranslation = useTranslation
  end
end

function ATMG:check(input)
  if self.fullMode then
    assert(input:size(2)==6, 'Expected 6 parameters, got ' .. input:size(2))
  else
    local numberParameters = 0
    if self.useRotation then
      numberParameters = numberParameters + 1
    end
    if self.useScale then
      numberParameters = numberParameters + 1
    end
    if self.useTranslation then
      numberParameters = numberParameters + 2
    end
    assert(input:size(2)==numberParameters, 'Expected '..numberParameters..
                                            ' parameters, got ' .. input:size(2))
  end
end

local function addOuterDim(t)
   local sizes = t:size()
   local newsizes = torch.LongStorage(sizes:size()+1)
   newsizes[1]=1
   for i=1,sizes:size() do
      newsizes[i+1]=sizes[i]
   end
   return t:view(newsizes)
end

function ATMG:updateOutput(_tranformParams)
  local transformParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
  else
    transformParams = _tranformParams
  end

  self:check(transformParams)
  local batchSize = transformParams:size(1)

  if self.fullMode then
    self.output = transformParams:view(batchSize, 2, 3)
  else
    local completeTransformation = torch.zeros(batchSize,3,3):typeAs(transformParams)
    completeTransformation:select(3,1):select(2,1):add(1)
    completeTransformation:select(3,2):select(2,2):add(1)
    completeTransformation:select(3,3):select(2,3):add(1)
    local transformationBuffer = torch.Tensor(batchSize,3,3):typeAs(transformParams)

    local paramIndex = 1
    if self.useRotation then
      local alphas = transformParams:select(2, paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,3):select(2,3):add(1)
      local cosines = torch.cos(alphas)
      local sinuses = torch.sin(alphas)
      transformationBuffer:select(3,1):select(2,1):copy(cosines)
      transformationBuffer:select(3,2):select(2,2):copy(cosines)
      transformationBuffer:select(3,1):select(2,2):copy(sinuses)
      transformationBuffer:select(3,2):select(2,1):copy(-sinuses)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end
    self.rotationOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()

    if self.useScale then
      local scaleFactors = transformParams:select(2,paramIndex)
      paramIndex = paramIndex + 1

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):copy(scaleFactors)
      transformationBuffer:select(3,2):select(2,2):copy(scaleFactors)
      transformationBuffer:select(3,3):select(2,3):add(1)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end
    self.scaleOutput = completeTransformation:narrow(2,1,2):narrow(3,1,2):clone()

    if self.useTranslation then
      local txs = transformParams:select(2,paramIndex)
      local tys = transformParams:select(2,paramIndex+1)

      transformationBuffer:zero()
      transformationBuffer:select(3,1):select(2,1):add(1)
      transformationBuffer:select(3,2):select(2,2):add(1)
      transformationBuffer:select(3,3):select(2,3):add(1)
      transformationBuffer:select(3,3):select(2,1):copy(txs)
      transformationBuffer:select(3,3):select(2,2):copy(tys)

      completeTransformation = torch.bmm(completeTransformation, transformationBuffer)
    end

    self.output=completeTransformation:narrow(2,1,2)
  end

  if _tranformParams:nDimension()==1 then
    self.output = self.output:select(1,1)
  end
  return self.output
end


function ATMG:updateGradInput(_tranformParams, _gradParams)
  local transformParams, gradParams
  if _tranformParams:nDimension()==1 then
    transformParams = addOuterDim(_tranformParams)
    gradParams = addOuterDim(_gradParams):clone()
  else
    transformParams = _tranformParams
    gradParams = _gradParams:clone()
  end

  local batchSize = transformParams:size(1)
  if self.fullMode then
    self.gradInput = gradParams:view(batchSize, 6)
  else
    local paramIndex = transformParams:size(2)
    self.gradInput:resizeAs(transformParams)
    if self.useTranslation then
      local gradInputTranslationParams = self.gradInput:narrow(2,paramIndex-1,2)
      local tParams = torch.Tensor(batchSize, 1, 2):typeAs(transformParams)
      tParams:select(3,1):copy(transformParams:select(2,paramIndex-1))
      tParams:select(3,2):copy(transformParams:select(2,paramIndex))
      paramIndex = paramIndex-2

      local selectedOutput = self.scaleOutput
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,3,1):transpose(2,3)
      gradInputTranslationParams:copy(torch.bmm(selectedGradParams, selectedOutput))

      local gradientCorrection = torch.bmm(selectedGradParams:transpose(2,3), tParams)
      gradParams:narrow(3,1,2):narrow(2,1,2):add(1,gradientCorrection)
    end

    if self.useScale then
      local gradInputScaleparams = self.gradInput:narrow(2,paramIndex,1)
      local sParams = transformParams:select(2,paramIndex)
      paramIndex = paramIndex-1

      local selectedOutput = self.rotationOutput
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
      gradInputScaleparams:copy(torch.cmul(selectedOutput, selectedGradParams):sum(2):sum(3))

      gradParams:select(3,1):select(2,1):cmul(sParams)
      gradParams:select(3,2):select(2,1):cmul(sParams)
      gradParams:select(3,1):select(2,2):cmul(sParams)
      gradParams:select(3,2):select(2,2):cmul(sParams)
    end

    if self.useRotation then
      local gradInputRotationParams = self.gradInput:narrow(2,paramIndex,1)
      local rParams = transformParams:select(2,paramIndex)

      local rotationDerivative = torch.zeros(batchSize, 2, 2):typeAs(rParams)
      torch.sin(rotationDerivative:select(3,1):select(2,1),-rParams)
      torch.sin(rotationDerivative:select(3,2):select(2,2),-rParams)
      torch.cos(rotationDerivative:select(3,1):select(2,2),rParams)
      torch.cos(rotationDerivative:select(3,2):select(2,1),rParams):mul(-1)
      local selectedGradParams = gradParams:narrow(2,1,2):narrow(3,1,2)
      gradInputRotationParams:copy(torch.cmul(rotationDerivative,selectedGradParams):sum(2):sum(3))
    end
  end

  if _tranformParams:nDimension()==1 then
    self.gradInput = self.gradInput:select(1,1)
  end
  return self.gradInput
end


