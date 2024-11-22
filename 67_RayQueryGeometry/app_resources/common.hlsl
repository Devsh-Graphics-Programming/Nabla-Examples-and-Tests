#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

struct SGeomInfo
{
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;

	uint32_t indexType; // 16 bit, 32 bit or none
	uint32_t vertexStride;
	uint32_t byteOffset;
	uint32_t smoothNormals;	// flat for cube, rectangle, disk
};

struct SPushConstants
{
    uint64_t geometryInfoBuffer;

	float32_t3 camPos;
	float32_t4x4 invMVP;

	float32_t2 scaleNDC;
	float32_t2 offsetNDC;
};

#ifdef __HLSL_VERSION
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
#endif

#endif  // RQG_COMMON_HLSL
