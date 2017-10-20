--[[
    Input: A table of tables 
    Output: A table of talbes or a table
        X2 = ShuffleTable.forward(X1)
    Then:
        X2[i][j] = X1[idx[i][j][1] ][idx[i][j][2] ]

    This function is slower than ShuffleTable. Use it
    only if shuffling function from the input to output
    is not a one-to-one mapping
--]]

local ShuffleTable, parent = torch.class('nn.ShuffleTable', 'nn.Module')

function ShuffleTable:__init(idx, noSecondParam)
    assert(noSecondParam == nil)

    parent.__init(self, idx)

    self.gradInput = {}
    self.idx = idx
    assert(type(idx) == 'table')
end

local function _getOutput(input, tpair)

  if type(tpair) == 'table' then
    assert(type(input[tpair.i]) == 'table')
    return input[tpair.i][tpair.j]
  else
    assert(type(input) == 'table')
    return input[tpair]
  end

end

function ShuffleTable:updateOutput(input)
  self.output = {}
  for i = 1,#self.idx do
    if type(self.idx[i]) == 'table' and self.idx[i].i == nil then
      self.output[i] = {}
      for j = 1,#self.idx[i] do
        self.output[i][j] = _getOutput(input, self.idx[i][j])
      end
    else
      self.output[i] = _getOutput(input, self.idx[i])
    end
  end
  return self.output
end

local function addGrad(gradInput, gradOutput, tpair)

  if type(tpair) == 'table' then
    if gradInput[tpair.i][tpair.j] == nil then
      gradInput[tpair.i][tpair.j] = gradOutput
    else
      gradInput[tpair.i][tpair.j] = gradInput[tpair.i][tpair.j]:clone():add(gradOutput)
    end
  else
    if gradInput[tpair] == nil then
      gradInput[tpair] = gradOutput
    else
      gradInput[tpair] = gradInput[tpair]:clone():add(gradOutput)
    end
  end

end

function ShuffleTable:updateGradInput(input, gradOutput)
  
  self.gradInput = {}
  
  -- Initialization
  for i = 1,#input do
    if type(input[i]) == 'table' then
      self.gradInput[i] = {}
    end
  end

  -- Shuffle
  for i = 1,#self.idx do
    if type(self.idx[i]) == 'table' and self.idx[i].i == nil then
      for j = 1,#self.idx[i] do
        tpair = self.idx[i][j]
        addGrad(self.gradInput, gradOutput[i][j], tpair)
      end
    else
      tpair = self.idx[i]
      addGrad(self.gradInput, gradOutput[i], tpair)
    end
  end

  -- Zero grad
  for i = 1,#input do
    if type(input[i]) == 'table' then
      for j = 1,#self.gradInput[i] do
        if self.gradInput[i][j] == nil then
          self.gradInput[i][j] = torch.zeros(input[i][j]:size())
        end
      end
    else
      if self.gradInput[i] == nil then
        self.gradInput[i] = torch.zeros(input[i]:size())
      end
    end
  end

  -- Cuda
  if opt == nil or opt.cuda then
    for i = 1,#self.gradInput do
      if type(self.gradInput[i]) == 'table' then
        for j = 1,#self.gradInput[i] do
          self.gradInput[i][j] = self.gradInput[i][j]:cuda()
        end
      else
        self.gradInput[i] = self.gradInput[i]:cuda()
      end
    end
  end

  return self.gradInput
end

function ShuffleTable:__tostring__()
  return string.format('%s()', torch.type(self))
end













