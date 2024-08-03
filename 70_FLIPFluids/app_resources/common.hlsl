#ifndef _FLIP_EXAMPLE_COMMON_HLSL
#define _FLIP_EXAMPLE_COMMON_HLSL

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

NBL_CONSTEXPR uint32_t WorkgroupSize = 128;

#ifdef __HLSL_VERSION
struct Particle
{
    float id;
    float pad0[3];

    float4 position;
    float4 velocity;
};

struct SMVPParams
{
	float4x4 MVP;
	float3x4 MV;
	float3x3 normalMat;
};
#endif

#endif
