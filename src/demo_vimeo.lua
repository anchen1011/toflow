-------------------------------------------------------------------------------
print '==> initializing...'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('tofu runnable')
cmd:text()
cmd:text('Options:')
cmd:option('-cuda',             true,         'Whether using cuda')
cmd:option('-gpuID',            1,            'ID of GPUs to use')
cmd:option('-mode',            'denoise',     'Model class for evaluation')
cmd:option('-inpath',          nil,           'The input sequence directory')
cmd:option('-outpath',         nil,           'The location to store the result')
cmd:text()  

opt = cmd:parse(arg or {})

require('main/init')
local loader = require('main/loader')
local gen_path = loader.gen_path
local get_file = loader.get_file

mode = opt.mode

if opt.inpath ~= nil then
  inpath = opt.inpath
else
  if mode == 'denoise' then
    inpath = '../data/vimeo_sep_noisy'
  elseif mode == 'deblock' then
    inpath = '../data/vimeo_sep_block'
  elseif mode == 'sr' then
    inpath = '../data/vimeo_sep_blur'
  elseif mode == 'interp' then
    inpath = '../data/vimeo_tri_test'
  end
end

if opt.outpath ~= nil then
  outpath = opt.outpath
else
  if mode == 'denoise' then
    outpath = '../output/dnoise'
  elseif mode == 'deblock' then
    outpath = '../output/deblock'
  elseif mode == 'sr' then
    outpath = '../output/sr'
  elseif mode == 'interp' then
    outpath = '../output/interp'
  end
end

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
model = torch.load(modelpath):float()
if opt.cuda then 
  model = cudnn.convert(model,cudnn)
  model = model:cuda()
end
model:evaluate()

local time = 0
local timer = torch.Timer()
local sample, h, w

subpaths = dir.getdirectories(inpath)
for m = 1,#subpaths do
  subsubpaths = dir.getdirectories(subpaths[m])
  for n = 1,#subsubpaths do
    finalpath = subsubpaths[n]
    paths.mkdir(finalpath)

    local st = gen_path(finalpath, loadMode)
    local size = #st

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
      output = output:float()
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
      local p2 = paths.concat(outpath, 'out.png')
      image.save(p2, output)

      print('  frame '..i..' is done, it takes '..curTime..'s')
    end

  end
end

print '==> finishing...'

sample = nil
print('  '..mode..' takes '..time..'s on '..size..' frames sequence')
-- TODO(Baian): Can you print the average runtime too?
