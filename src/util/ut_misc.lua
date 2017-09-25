local M = {}

function M.trim(s)
  return (string.gsub(s, "^%s*(.-)%s*$", "%1"))
end

function M.getHostname()

  local tfile = assert(io.popen('hostname'))
  local hostname = tfile:read('*all')
  tfile:close()
  hostname = M.trim(hostname)
  return hostname

end

function M.getUsername()

  local tfile = assert(io.popen('echo $USER'))
  local username = tfile:read('*all')
  tfile:close()
  username = M.trim(username)
  return username

end

function M.cmkdir(inpath)

  if not path.isdir(inpath) then
    path.mkdir(inpath)
  end

end

function M.listfiles(inpath, ext)

  filelist = {}
  extlen = #ext
  for file in path.dir(inpath) do
    filelen = #file
    if path.isfile(path.join(inpath,file)) and #file > #ext
        and string.sub(file, filelen-extlen+1) == ext then
      table.insert(filelist, file)
    end
  end
  return filelist

end

function M.getResolution(level, dataset)

  if dataset:sub(1,6) == 'sintel' then 
    local h = 448 / (2 ^ (5 - level))
    local w = 1024 / (2 ^ (5 - level))
    return 448, 1024, h, w
  end

  if dataset:sub(1,5) == 'vimeo' then 
    local h = 256 / (2 ^ (4 - level))
    local w = 448 / (2 ^ (4 - level))
    return 256, 448, h, w
  end

  if dataset == 'toy' or dataset == 'vd_toy' then
    return nil, nil, nil, nil
  end

  error('undefined dataset for resolution computation')

  return
end

function M.getAllResolutions(level, dataset)

  local hlist = {}
  local wlist = {}
  for i = 1,level do
    local hfull, wfull, h, w = M.getResolution(i, dataset)
    hlist[i] = h
    wlist[i] = w
  end
  return hlist, wlist

end

function M.shuffleTensor(tensor)
  local shuffle_indexes = torch.randperm(tensor:size(1))
  local tensor_shuffled = torch.FloatTensor(tensor:size())
  for i=1,tensor:size(1),1 do
    tensor_shuffled[i] = tensor[shuffle_indexes[i]]
  end
  return tensor_shuffled
end

local function serializeTable(val, name, skipnewlines, depth)
    skipnewlines = skipnewlines or false
    depth = depth or 0

    local tmp = string.rep(" ", depth)

    if name then tmp = tmp .. name .. " = " end

    if type(val) == "table" then
        tmp = tmp .. "{" .. (not skipnewlines and "\n" or "")

        for k, v in pairs(val) do
            tmp =  tmp .. serializeTable(v, k, skipnewlines, depth + 1) .. "," .. (not skipnewlines and "\n" or "")
        end

        tmp = tmp .. string.rep(" ", depth) .. "}"
    elseif type(val) == "number" then
        tmp = tmp .. tostring(val)
    elseif type(val) == "string" then
        tmp = tmp .. string.format("%q", val)
    elseif type(val) == "boolean" then
        tmp = tmp .. (val and "true" or "false")
    else
        tmp = tmp .. "\"[inserializeable datatype:" .. type(val) .. "]\""
    end

    return tmp
end

function M.arrayToFile(table, file)
  file = io.open(file, "w")
  file:write(serializeTable(table))
  file:close()
end

return M
