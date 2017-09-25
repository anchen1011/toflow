#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ScaleBHWD.c"
#else

#include <stdbool.h>


static int nn_(ScaleBHWD_updateOutput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_checkudata(L, 4, torch_Tensor);

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int output_height = output->size[1];
  int output_width = output->size[2];
  int inputImages_channels = inputImages->size[3];

  int output_strideBatch = output->stride[0];
  int output_strideHeight = output->stride[1];
  int output_strideWidth = output->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];

  real *inputImages_data, *output_data, *grids_data;
  inputImages_data = THTensor_(data)(inputImages);
  output_data = THTensor_(data)(output);
  grids_data = THTensor_(data)(grids);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < output_height; yOut++)
    {
      for(xOut=0; xOut < output_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);

        

        const int outAddress = output_strideBatch * b + output_strideHeight * yOut + output_strideWidth * xOut;
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;
        // interpolation happens here
        for(t=0; t<inputImages_channels; t++)
        {
           if(topLeftIsIn) inTopLeft = inputImages_data[inTopLeftAddress + t];
           if(topRightIsIn) inTopRight = inputImages_data[inTopRightAddress + t];
           if(bottomLeftIsIn) inBottomLeft = inputImages_data[inBottomLeftAddress + t];
           if(bottomRightIsIn) inBottomRight = inputImages_data[inBottomRightAddress + t];

           v = xWeightTopLeft * yWeightTopLeft * inTopLeft
             + (1 - xWeightTopLeft) * yWeightTopLeft * inTopRight
             + xWeightTopLeft * (1 - yWeightTopLeft) * inBottomLeft
             + (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * inBottomRight;
           
           output_data[outAddress + t] = v;
        }

      }
    }
  }

  return 1;
}

static int nn_(ScaleBHWD_updateGradInput)(lua_State *L)
{
  THTensor *inputImages = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *grids = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInputImages = luaT_checkudata(L, 4, torch_Tensor);
  THTensor *gradGrids = luaT_checkudata(L, 5, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 6, torch_Tensor);

  bool onlyGrid=false;

  int batchsize = inputImages->size[0];
  int inputImages_height = inputImages->size[1];
  int inputImages_width = inputImages->size[2];
  int gradOutput_height = gradOutput->size[1];
  int gradOutput_width = gradOutput->size[2];
  int inputImages_channels = inputImages->size[3];

  int gradOutput_strideBatch = gradOutput->stride[0];
  int gradOutput_strideHeight = gradOutput->stride[1];
  int gradOutput_strideWidth = gradOutput->stride[2];

  int inputImages_strideBatch = inputImages->stride[0];
  int inputImages_strideHeight = inputImages->stride[1];
  int inputImages_strideWidth = inputImages->stride[2];

  int gradInputImages_strideBatch = gradInputImages->stride[0];
  int gradInputImages_strideHeight = gradInputImages->stride[1];
  int gradInputImages_strideWidth = gradInputImages->stride[2];

  int grids_strideBatch = grids->stride[0];
  int grids_strideHeight = grids->stride[1];
  int grids_strideWidth = grids->stride[2];

  int gradGrids_strideBatch = gradGrids->stride[0];
  int gradGrids_strideHeight = gradGrids->stride[1];
  int gradGrids_strideWidth = gradGrids->stride[2];

  real *inputImages_data, *gradOutput_data, *grids_data, *gradGrids_data, *gradInputImages_data;
  inputImages_data = THTensor_(data)(inputImages);
  gradOutput_data = THTensor_(data)(gradOutput);
  grids_data = THTensor_(data)(grids);
  gradGrids_data = THTensor_(data)(gradGrids);
  gradInputImages_data = THTensor_(data)(gradInputImages);

  int b, yOut, xOut;

  for(b=0; b < batchsize; b++)
  {
    for(yOut=0; yOut < gradOutput_height; yOut++)
    {
      for(xOut=0; xOut < gradOutput_width; xOut++)
      {
        //read the grid
        real yf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth];
        real xf = grids_data[b*grids_strideBatch + yOut*grids_strideHeight + xOut*grids_strideWidth + 1];

        // get the weights for interpolation
        int yInTopLeft, xInTopLeft;
        real yWeightTopLeft, xWeightTopLeft;
 
        real xcoord = (xf + 1) * (inputImages_width - 1) / 2;
        xInTopLeft = floor(xcoord);
        xWeightTopLeft = 1 - (xcoord - xInTopLeft);

        real ycoord = (yf + 1) * (inputImages_height - 1) / 2;
        yInTopLeft = floor(ycoord);
        yWeightTopLeft = 1 - (ycoord - yInTopLeft);

        
        const int inTopLeftAddress = inputImages_strideBatch * b + inputImages_strideHeight * yInTopLeft + inputImages_strideWidth * xInTopLeft;
        const int inTopRightAddress = inTopLeftAddress + inputImages_strideWidth;
        const int inBottomLeftAddress = inTopLeftAddress + inputImages_strideHeight;
        const int inBottomRightAddress = inBottomLeftAddress + inputImages_strideWidth;

        const int gradInputImagesTopLeftAddress = gradInputImages_strideBatch * b + gradInputImages_strideHeight * yInTopLeft + gradInputImages_strideWidth * xInTopLeft;
        const int gradInputImagesTopRightAddress = gradInputImagesTopLeftAddress + gradInputImages_strideWidth;
        const int gradInputImagesBottomLeftAddress = gradInputImagesTopLeftAddress + gradInputImages_strideHeight;
        const int gradInputImagesBottomRightAddress = gradInputImagesBottomLeftAddress + gradInputImages_strideWidth;

        const int gradOutputAddress = gradOutput_strideBatch * b + gradOutput_strideHeight * yOut + gradOutput_strideWidth * xOut;

        real topLeftDotProduct = 0;
        real topRightDotProduct = 0;
        real bottomLeftDotProduct = 0;
        real bottomRightDotProduct = 0;

        real v=0;
        real inTopLeft=0;
        real inTopRight=0;
        real inBottomLeft=0;
        real inBottomRight=0;

        // we are careful with the boundaries
        bool topLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool topRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft >= 0 && yInTopLeft <= inputImages_height-1;
        bool bottomLeftIsIn = xInTopLeft >= 0 && xInTopLeft <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;
        bool bottomRightIsIn = xInTopLeft+1 >= 0 && xInTopLeft+1 <= inputImages_width-1 && yInTopLeft+1 >= 0 && yInTopLeft+1 <= inputImages_height-1;

        int t;

        for(t=0; t<inputImages_channels; t++)
        {
           real gradOutValue = gradOutput_data[gradOutputAddress + t];
           if(topLeftIsIn)
           {
              real inTopLeft = inputImages_data[inTopLeftAddress + t];
              topLeftDotProduct += inTopLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopLeftAddress + t] += xWeightTopLeft * yWeightTopLeft * gradOutValue;
           }

           if(topRightIsIn)
           {
              real inTopRight = inputImages_data[inTopRightAddress + t];
              topRightDotProduct += inTopRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesTopRightAddress + t] += (1 - xWeightTopLeft) * yWeightTopLeft * gradOutValue;
           }

           if(bottomLeftIsIn)
           {
              real inBottomLeft = inputImages_data[inBottomLeftAddress + t];
              bottomLeftDotProduct += inBottomLeft * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomLeftAddress + t] += xWeightTopLeft * (1 - yWeightTopLeft) * gradOutValue;
           }

           if(bottomRightIsIn)
           {
              real inBottomRight = inputImages_data[inBottomRightAddress + t];
              bottomRightDotProduct += inBottomRight * gradOutValue;
              if(!onlyGrid) gradInputImages_data[gradInputImagesBottomRightAddress + t] += (1 - xWeightTopLeft) * (1 - yWeightTopLeft) * gradOutValue;
           }
        }

        yf = - xWeightTopLeft * topLeftDotProduct + xWeightTopLeft * bottomLeftDotProduct - (1-xWeightTopLeft) * topRightDotProduct + (1-xWeightTopLeft) * bottomRightDotProduct;
        xf = - yWeightTopLeft * topLeftDotProduct + yWeightTopLeft * topRightDotProduct - (1-yWeightTopLeft) * bottomLeftDotProduct + (1-yWeightTopLeft) * bottomRightDotProduct;

        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth] = yf * (inputImages_height-1) / 2;
        gradGrids_data[b*gradGrids_strideBatch + yOut*gradGrids_strideHeight + xOut*gradGrids_strideWidth + 1] = xf * (inputImages_width-1) / 2;

      }
    }
  }

  return 1;
}

static const struct luaL_Reg nn_(ScaleBHWD__) [] = {
  {"ScaleBHWD_updateOutput", nn_(ScaleBHWD_updateOutput)},
  {"ScaleBHWD_updateGradInput", nn_(ScaleBHWD_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ScaleBHWD_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ScaleBHWD__), "nn");
  lua_pop(L,1);
}

#endif
