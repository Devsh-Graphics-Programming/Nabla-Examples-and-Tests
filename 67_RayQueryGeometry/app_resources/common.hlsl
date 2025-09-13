#ifndef RQG_COMMON_HLSL
#define RQG_COMMON_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSize = 16;

enum NormalType : uint32_t
{
    NT_R8G8B8A8_SNORM,
    NT_R32G32B32_SFLOAT,
};

// we need bitfield support in NBL_HLSL_DECLARE_STRUCT it seems
struct SGeomInfo
{
    uint64_t vertexBufferAddress;
    uint64_t indexBufferAddress;
    uint64_t normalBufferAddress;

    uint32_t normalType : 1;
    uint32_t indexType : 1; // 16 bit, 32 bit
};

struct SPushConstants
{
    uint64_t geometryInfoBuffer;

    float32_t3 camPos;
    float32_t4x4 invMVP;

    float32_t2 scaleNDC;
    float32_t2 offsetNDC;
};

#endif  // RQG_COMMON_HLSL
