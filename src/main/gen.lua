local function gen(pth, mode)
  local num = 7
  if mode == 'tri' then
    num = 3
  end
  local lst = dir.getallfiles(pth)
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

return gen
