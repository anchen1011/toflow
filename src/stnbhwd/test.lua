-- you can easily test specific units like this:
-- th -lnn -e "nn.test{'LookupTable'}"
-- th -lnn -e "nn.test{'LookupTable', 'Add'}"

local mytester = torch.Tester()
local jac
local sjac

local precision = 1e-5
local expprecision = 1e-4

local stntest = {}

function stntest.AffineGridGeneratorBHWD_batch()
   local nframes = torch.random(2,10)
   local height = torch.random(2,5)
   local width = torch.random(2,5)
   local input = torch.zeros(nframes, 2, 3):uniform()
   local module = nn.AffineGridGeneratorBHWD(height, width)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function stntest.AffineGridGeneratorBHWD_single()
   local height = torch.random(2,5)
   local width = torch.random(2,5)
   local input = torch.zeros(2, 3):uniform()
   local module = nn.AffineGridGeneratorBHWD(height, width)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision, 'error on state ')

   -- IO
   local ferr,berr = jac.testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ')

end

function stntest.BilinearSamplerBHWD_batch()
   local nframes = torch.random(2,10)
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(nframes, height, width, channels):uniform()
   local grids = torch.zeros(nframes, height, width, 2):uniform(-1, 1)
   local module = nn.BilinearSamplerBHWD()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end

function stntest.BilinearSamplerBHWD_single()
   local height = torch.random(1,5)
   local width = torch.random(1,5)
   local channels = torch.random(1,6)
   local inputImages = torch.zeros(height, width, channels):uniform()
   local grids = torch.zeros(height, width, 2):uniform(-1, 1)
   local module = nn.BilinearSamplerBHWD()

   -- test input images (first element of input table)
   module._updateOutput = module.updateOutput
   function module:updateOutput(input)
      return self:_updateOutput({input, grids})
   end
   
   module._updateGradInput = module.updateGradInput
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({input, grids}, gradOutput)
      return self.gradInput[1]
   end

   local errImages = jac.testJacobian(module,inputImages)
   mytester:assertlt(errImages,precision, 'error on state ')

   -- test grids (second element of input table)
   function module:updateOutput(input)
      return self:_updateOutput({inputImages, input})
   end
   
   function module:updateGradInput(input, gradOutput)
      self:_updateGradInput({inputImages, input}, gradOutput)
      return self.gradInput[2]
   end

   local errGrids = jac.testJacobian(module,grids)
   mytester:assertlt(errGrids,precision, 'error on state ')
end

function stntest.AffineTransformMatrixGenerator_batch()
   -- test all possible transformations
   for _,useRotation in pairs{true,false} do
      for _,useScale in pairs{true,false} do
         for _,useTranslation in pairs{true,false} do
            local currTest = ''
            if useRotation then currTest = currTest..'rotation ' end
            if useScale then currTest = currTest..'scale ' end
            if useTranslation then currTest = currTest..'translation' end
            if currTest=='' then currTest = 'full' end

            local nbNeededParams = 0
            if useRotation then nbNeededParams = nbNeededParams + 1 end
            if useScale then nbNeededParams = nbNeededParams + 1 end
            if useTranslation then nbNeededParams = nbNeededParams + 2 end
            if nbNeededParams == 0 then nbNeededParams = 6 end -- full affine case

            local nframes = torch.random(2,10)
            local params = torch.zeros(nframes,nbNeededParams):uniform()
            local module = nn.AffineTransformMatrixGenerator(useRotation,useScale,useTranslation)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test '..currTest)

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test '..currTest)
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test '..currTest)

         end
      end
   end
end

function stntest.AffineTransformMatrixGenerator_single()
   -- test all possible transformations
   for _,useRotation in pairs{true,false} do
      for _,useScale in pairs{true,false} do
         for _,useTranslation in pairs{true,false} do
            local currTest = ''
            if useRotation then currTest = currTest..'rotation ' end
            if useScale then currTest = currTest..'scale ' end
            if useTranslation then currTest = currTest..'translation' end
            if currTest=='' then currTest = 'full' end

            local nbNeededParams = 0
            if useRotation then nbNeededParams = nbNeededParams + 1 end
            if useScale then nbNeededParams = nbNeededParams + 1 end
            if useTranslation then nbNeededParams = nbNeededParams + 2 end
            if nbNeededParams == 0 then nbNeededParams = 6 end -- full affine case

            local params = torch.zeros(nbNeededParams):uniform()
            local module = nn.AffineTransformMatrixGenerator(useRotation,useScale,useTranslation)

            local err = jac.testJacobian(module,params)
            mytester:assertlt(err,precision, 'error on state for test '..currTest)

            -- IO
            local ferr,berr = jac.testIO(module,params)
            mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err for test '..currTest)
            mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err for test '..currTest)

         end
      end
   end
end

mytester:add(stntest)

if not nn then
   require 'nn'
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   mytester:run()
else
   jac = nn.Jacobian
   sjac = nn.SparseJacobian
   function stn.test(tests)
      -- randomize stuff
      math.randomseed(os.time())
      mytester:run(tests)
      return mytester
   end
end
