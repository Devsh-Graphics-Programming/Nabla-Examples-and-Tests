#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSizeX = 16;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSizeY = 16;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSize = WorkgroupSizeX*WorkgroupSizeY;

static const uint32_t FRAMES_IN_FLIGHT = 3u;

static const uint32_t RED_OFFSET = 0u;
static const uint32_t GREEN_OFFSET = 256u;
static const uint32_t BLUE_OFFSET = 256u * 2u;

static const uint32_t CHANEL_CNT = 3;
static const uint32_t VAL_PER_CHANEL_CNT = 256;
static const uint32_t HISTOGRAM_SIZE = CHANEL_CNT * VAL_PER_CHANEL_CNT;
static const uint32_t HISTOGRAM_BYTE_SIZE = HISTOGRAM_SIZE * sizeof(uint32_t);
static const uint32_t COMBINED_HISTOGRAM_BUFFER_BYTE_SIZE = HISTOGRAM_BYTE_SIZE * FRAMES_IN_FLIGHT;

struct PushConstants
{
    uint32_t histogramBufferOffset;
};