-------------------------------------------------------------------------------
print '==> initializing...'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('ssflow runnable')
cmd:text()
cmd:text('Options:')
cmd:option('-gpuID',            1,            'ID of GPUs to use')
cmd:option('-mode',            'denoise',     'Model class for evaluation')
cmd:option('-inpath',          'data/example','The input sequence directory')
cmd:option('-outpath',         'data/tmp',    'The location to store the result')
cmd:text()  

opt = cmd:parse(arg or {})

require('main/init')
local gen = require('main/gen')
local get = require('main/get')

mode = opt.mode
inpath = opt.inpath
outpath = opt.outpath
print('  input sequence: '..inpath)
print('  result stored in: '..outpath)
if ~exist(outpath, 'dir') then
  mkdir(outpath)
end

print '==> loading...'

if mode == 'denoise' then
  modelpath = 'models/denoise.t7'
elseif mode == 'sr' then 
  modelpath = 'models/sr.t7'
elseif mode == 'interp' then
  modelpath = 'interp.t7'
elseif mode == 'deblock' then
  modelpath = 'deblock.t7'
  mode = 'denoise'
end

local loadMode = 'sep'
if mode == 'interp' then loadMode = 'tri' end
local st = gen(inpath, loadMode)
local model = torch.load(modelpath):cuda()
model:evaluate()
local size = #st

local time = 0
local timer = torch.Timer()
local sample, h, w

sample, h, w = get(st[1], loadMode, h, w)
h = 16 * math.floor(h / 16)
w = 16 * math.floor(w / 16)

print '==> processing...'

for i = 1,size do
  sample, h, w = get(st[i], loadMode, h, w)
  sample = ut.nn.cudanize(sample)
  timer:reset()
  local output = ut.nn.memFriendlyForward(model, sample):double()
  output = output:double()
  if mode == 'sr' then 
    output:add(sample[1]:double())
  end
  output = ut.tf.defaultDetransform(output:squeeze())
  local curTime = timer:time().real
  time = time + timer:time().real

  -- Clean memory
  ut.nn.sanitize(model)
  collectgarbage()
  collectgarbage()

  -- Save results
  if mode == 'interp' then
    local p1 = paths.concat(outpath, string.format('%04d.png',2*i-1))
    local p2 = paths.concat(outpath, string.format('%04d.png', 2*i))
    image.save(p1, ut.tf.defaultDetransform(sample[1]:squeeze()))
    image.save(p2, output)
  else
    local p2 = paths.concat(outpath, string.format('%04d.png', i))
    image.save(p2, output)
  end

  print('  frame '..i..' is done, it takes '..curTime..'s')
end

print '==> finishing...'

sample = nil
print('  '..mode..' takes '..time..'s on '..size..' frames sequence')
