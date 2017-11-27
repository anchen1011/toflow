-------------------------------------------------------------------------------
print '==> initializing...'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('tofu runnable')
cmd:text()
cmd:text('Options:')
cmd:option('-nocuda',           false,            'Whether disable cuda')
cmd:option('-gpuID',            1,               'ID of GPUs to use')
cmd:option('-mode',            'denoise',        'Model class for evaluation')
cmd:option('-inpath',          '../data/example/noisy','The input sequence directory')
cmd:option('-outpath',         '../demo_output',    'The location to store the result')
cmd:text()  

opt = cmd:parse(arg or {})
opt.cuda = not opt.nocuda

require('main/init')
local loader = require('main/loader')
local gen_path = loader.gen_path
local get_file = loader.get_file

mode = opt.mode
inpath = opt.inpath
outpath = opt.outpath
print('  input sequence: '..inpath)
print('  result stored in: '..outpath)
paths.mkdir(outpath)

print '==> loading...'

if mode == 'denoise' then
  modelpath = '../models/denoise.t7'
elseif mode == 'sr' then 
  modelpath = '../models/sr.t7'
elseif mode == 'interp' then
  modelpath = '../models/interp.t7'
elseif mode == 'deblock' then
  modelpath = '../models/deblock.t7'
  mode = 'denoise'
end

local loadMode = 'sep'
if mode == 'interp' then loadMode = 'tri' end
local st = gen_path(inpath, loadMode)
-- st is a table of input files.
-- 
-- Suppose there are 9 images under inpath: im1.png, im2.png, ..., im9.png
-- 
-- For interpolation:
--   st[1] = {'im1.png', 'im2.png'}  => the algorithm will generate frame at T=1.5
--   st[2] = {'im2.png', 'im3.png'}  => T=2.5
--   ...
--   st[8] = {'im8.png', 'im9.png'}  => T=8.5
--
-- For denoising/deblocking/sr:
--   st[1] = {'im1.png', 'im2.png', ..., 'im7.png}  => the algorithm will a noise-free/block-free/high-res version of im4.png
--   st[2] = {'im2.png', 'im3.png', ..., 'im8.png}  => im5.png
--   ...

model = torch.load(modelpath):float()
if opt.cuda then 
  model = cudnn.convert(model,cudnn)
  model = model:cuda()
end
model:evaluate()
local size = #st

local time = 0
local timer = torch.Timer()
local sample, h, w

sample, h, w = get_file(st[1], loadMode, h, w)
h = 16 * math.floor(h / 16)
w = 16 * math.floor(w / 16)

print '==> processing...'

for i = 1,size do
  sample, h, w = get_file(st[i], loadMode, h, w)
  if opt.cuda then 
    sample = ut.nn.cudanize(sample)
  else
    sample = ut.nn.floatize(sample)
  end
  timer:reset()
  local output = ut.nn.memFriendlyForward(model, sample):float()
  if mode == 'sr' then 
    output:add(sample[1]:float())
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
