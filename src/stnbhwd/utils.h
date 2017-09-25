#ifndef CUNN_UTILS_H
#define CUNN_UTILS_H

#include <lua.h>
#include "THCGeneral.h"

THCState* getCutorchState(lua_State* L);

#endif
