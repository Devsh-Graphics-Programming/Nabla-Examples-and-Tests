#ifndef _FLIP_EXAMPLE_COMMON_HLSL
#define _FLIP_EXAMPLE_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 128;
NBL_CONSTEXPR uint32_t WorkgroupGridDim = 8;
NBL_CONSTEXPR float ratioFLIPPIC = 0.95;
NBL_CONSTEXPR float deltaTime = 1.0f / 90.0f;
NBL_CONSTEXPR float gravity = 15.0f;

#ifdef __HLSL_VERSION
struct Particle
{
    float32_t3 position;
    float32_t3 velocity;
};

struct SMVPParams
{
    float4 camPos;

	float4x4 MVP;
	float4x4 M;
	float4x4 V;
    float4x4 P;
};
#endif

#endif
