#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR uint32_t WorkgroupSizeX = 16;
NBL_CONSTEXPR uint32_t WorkgroupSizeY = 16;
NBL_CONSTEXPR uint32_t WorkgroupSize = WorkgroupSizeX*WorkgroupSizeY;

struct PushConstants
{
    uint32_t histogramBufferOffset;
};