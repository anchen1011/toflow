local M = {}
local TAG_FLOAT = 202021.25

--   Functions in this file might be called by a thread
--   In that case, add following lines to _init_():
--
--   require 'image'

function M.loadImage(path)
	local input = image.load(path, 3, 'float')
	return input
end

function M.loadFlow(filename)

  local ff = torch.DiskFile(filename):binary()
	local tag = ff:readFloat()
	if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
	   ' perhaps bigendian error','readflo()')
	end
	
	local w = ff:readInt()
	local h = ff:readInt()
	local nbands = 2
	local tf = torch.FloatTensor(h, w, nbands)
	ff:readFloat(tf:storage())
	ff:close()

	local flow = tf:permute(3,1,2)	
	return flow

end

return M
