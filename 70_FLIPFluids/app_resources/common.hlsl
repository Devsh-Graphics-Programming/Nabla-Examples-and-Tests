#ifndef _FLIP_EXAMPLE_COMMON_HLSL
#define _FLIP_EXAMPLE_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// TODO: Increase the `WorkgroupSize` to 512
NBL_CONSTEXPR uint32_t WorkgroupSize = 128;
// TODO: Use "virtual workgroups" and Morton Code to map the 512 linear above to 8x8x8
NBL_CONSTEXPR uint32_t WorkgroupGridDim = 8;
NBL_CONSTEXPR float ratioFLIPPIC = 0.95;
NBL_CONSTEXPR float deltaTime = 1.0f / 90.0f;
NBL_CONSTEXPR float gravity = 15.0f;

#ifdef __HLSL_VERSION
// TODO: don't share this struct with C++
struct Particle
{
    float32_t3 position;
    float32_t3 velocity;
};

// TODO: after trimming this should fit in push constants
struct SMVPParams
{
    float4 camPos; // TODO: make it a `float32_t3`

	float4x4 MVP;
	float4x4 M; // TODO: remove
	float4x4 V; // TODO: only one `float32_t3` is needed out of the view matrix
    float4x4 P; // TODO: remove
};
#endif

#endif
