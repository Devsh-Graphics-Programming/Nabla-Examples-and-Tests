#ifndef _FLIP_EXAMPLE_COMMON_HLSL
#define _FLIP_EXAMPLE_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define NUM_THREADS 128

NBL_CONSTEXPR uint32_t WorkgroupSize = NUM_THREADS;
NBL_CONSTEXPR float ratioFLIPPIC = 0.95;
NBL_CONSTEXPR float deltaTime = 1.0f / 60.0f;

#ifdef __HLSL_VERSION

static const float FLT_MIN = 1.175494351e-38;
static const float FLT_MAX = 3.402823466e+38;

struct Particle
{
    float4 position;
    float4 velocity;

    uint32_t id;
    uint3 pad;
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
