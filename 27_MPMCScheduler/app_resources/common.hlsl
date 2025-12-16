#include "nbl/builtin/hlsl/cpp_compat.hlsl"

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSizeX = 8;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSizeY = 8;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t WorkgroupSize = WorkgroupSizeX*WorkgroupSizeY;

struct PushConstants
{
    uint32_t sharedAcceptableIdleCount : 10;
    uint32_t globalAcceptableIdleCount : 10;
};