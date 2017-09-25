#include "luaT.h"
#include "THC.h"

#include "utils.c"

#include "BilinearSamplerBHWD.cu"
//#include "ScaleBHWD.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcustn(lua_State *L);

int luaopen_libcustn(lua_State *L)
{
  lua_newtable(L);
  cunn_BilinearSamplerBHWD_init(L);
//  cunn_ScaleBHWD_init(L);

  return 1;
}
