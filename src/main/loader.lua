require 'util/ut'

local npMax = 8500000

local function resizeInput(inputs, mode, h, w)

  local nFrames
  if mode == 'sep' then
    nFrames = 7
    local w8 = w/8
    local h8 = h/8
    inputs[8] = image.scale(inputs[8][1], w8, h8, 'bicubic'):resize(1,2,h8,w8)
  else
    nFrames = 2 
  end
  for i = 1,nFrames do
    inputs[i] = image.scale(inputs[i][1], w, h, 'bicubic'):resize(1,3,h,w)
  end

end

local function gen_path(filepath, mode)
  local num = 7
  if mode == 'tri' then
    num = 2
  end
  local lst = dir.getallfiles(filepath)
  table.sort(lst)
  local st = {}
  for i = 1,(#lst - num + 1) do
    local subst = {}
    for j = 1,num do
      table.insert(subst, lst[i + j - 1])
    end
    table.insert(st, subst)
  end
  return st
end

local function get_file(inputpath, mode, h, w)
  local mode = mode or 'sep'
  local inputs = {}

  if mode == 'sep' then
    inputs[1] = ut.tf.defaultTestTransform(ut.datasets.loadImage(inputpath[4]))
    for i = 1,3 do
      inputs[i+1] = ut.tf.defaultTestTransform(ut.datasets.loadImage(inputpath[i]))
    end
    for i = 5,7 do
      inputs[i] = ut.tf.defaultTestTransform(ut.datasets.loadImage(inputpath[i]))
    end
    inputs[8] = torch.zeros(2, inputs[1]:size(2)/8, inputs[1]:size(3)/8)
    for i = 1,8 do
      inputs[i] = inputs[i]:clone():resize(1,inputs[i]:size(1),inputs[i]:size(2),inputs[i]:size(3))
    end
  else
    for i = 1,2 do
      inputs[i] = ut.tf.defaultTestTransform(
        ut.datasets.loadImage(inputpath[i]))
      inputs[i] = inputs[i]:clone():resize(
        1,inputs[i]:size(1),inputs[i]:size(2),inputs[i]:size(3))
    end
  end

  -- Downsample images
  local hOrig, wOrig
  hOrig = inputs[1]:size(3)
  wOrig = inputs[1]:size(4)

  local nFrames
  if h ~= nil then
    if hOrig ~= h or wOrig ~= w then
      resizeInput(inputs, mode, h, w)
    end
  else
    local npOrig = hOrig * wOrig
    if npMax < npOrig then
      local scale = math.sqrt(npMax / npOrig)
      h = math.floor(hOrig * scale / 8) * 8
      w = math.floor(wOrig * scale / 8) * 8
      resizeInput(inputs, mode, h, w)
    else
      h = hOrig
      w = wOrig
    end
  end
 
  return inputs, h, w
end

return {gen_path=gen_path, get_file=get_file}
