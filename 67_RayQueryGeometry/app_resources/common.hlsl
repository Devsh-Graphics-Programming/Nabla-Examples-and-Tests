#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

#ifdef __HLSL_VERSION
struct VertexInfo
{
    uint vertexStride;
    uint byteOffset;
    uint vertexType;
};

struct SCameraParameters
{
    float3 camPos;

	float4x4 MVP;
    float4x4 invMVP;
	float3x4 V;
    float3x4 P;
};

// need a more elegant way of handling different indexTypes
template<uint32_t type>
uint32_t getIndex(uint64_t addr, uint64_t offset)
{
    return -1;  // invalid?
}

template<>
uint32_t getIndex<0>(uint64_t addr, uint64_t offset)   // EIT_16BIT = 0
{
    return uint32_t(vk::RawBufferLoad<uint16_t>(addr + offset * sizeof(uint16_t)));
}

template<>
uint32_t getIndex<1>(uint64_t addr, uint64_t offset)   // EIT_32BIT = 1
{
    return vk::RawBufferLoad<uint32_t>(addr + offset * sizeof(uint32_t));
}
#endif

#endif  // RQG_COMMON_HLSL
