#ifndef _DRAW_AABB_SIMPLE_COMMON_HLSL
#define _DRAW_AABB_SIMPLE_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct SSimplePushConstants
{
    float32_t4x4 MVP;
    uint64_t pVertices;
};

#ifdef __HLSL_VERSION
struct PSInput
{
    float32_t4 position : SV_Position;
    float32_t4 color : TEXCOORD0;
};
#endif

#endif
