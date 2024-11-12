#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

#ifdef __HLSL_VERSION
struct VertexInfo
{
    uint vertexStride;
    uint byteOffset;
};

struct SCameraParameters
{
    float3 camPos;

	float4x4 MVP;
    float4x4 invMVP;
	float3x4 V;
    float3x4 P;
};
#endif

#endif  // RQG_COMMON_HLSL
