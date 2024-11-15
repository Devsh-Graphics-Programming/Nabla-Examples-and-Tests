#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

struct SGeomBDA
{
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
};

#ifdef __HLSL_VERSION
//NBL_CONSTEXPR uint32_t OT_COUNT = 8;
enum ObjectType : uint32_t  // matches c++
{
	OT_CUBE = 0,
	OT_SPHERE,
	OT_CYLINDER,
	OT_RECTANGLE,
	OT_DISK,
	OT_ARROW,
	OT_CONE,
	OT_ICOSPHERE,

    OT_COUNT
};

struct SVertexInfo
{
    uint indexType; // 16 bit, 32 bit or none
    uint geomType;  // defines both vertex stride and byte offset
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
