#ifndef _NBL_THIS_EXAMPLE_PATHTRACE_COMMON_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_PATHTRACE_COMMON_HLSL_INCLUDED_


#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/spirv_intrinsics/raytracing.hlsl"

namespace nbl
{
namespace this_example
{
NBL_CONSTEXPR uint32_t SceneDSIndex = 0;
NBL_CONSTEXPR uint32_t SessionDSIndex = 1;
}
}
#include "renderer/shaders/scene.hlsl"
#include "renderer/shaders/session.hlsl"
#include "renderer/shaders/pathtrace/push_constants.hlsl"


#endif  // _NBL_THIS_EXAMPLE_PATHTRACE_COMMON_HLSL_INCLUDED_
