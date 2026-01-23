#ifndef _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_
#define _NBL_HLSL_EXT_PATHTRACING_COMMON_INCLUDED_

#include <nbl/builtin/hlsl/spirv_intrinsics/core.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/glsl.std.450.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/numbers.hlsl>
#include <nbl/builtin/hlsl/bxdf/common.hlsl>

namespace nbl
{
namespace hlsl
{
namespace path_tracing
{

template<typename T>
struct Tolerance
{
    NBL_CONSTEXPR_STATIC_INLINE T INTERSECTION_ERROR_BOUND_LOG2 = -8.0;

    static T __common(uint32_t depth)
    {
        T depthRcp = 1.0 / T(depth);
        return INTERSECTION_ERROR_BOUND_LOG2;
    }

    static T getStart(uint32_t depth)
    {
        return nbl::hlsl::exp2(__common(depth));
    }

    static T getEnd(uint32_t depth)
    {
        return 1.0 - nbl::hlsl::exp2(__common(depth) + 1.0);
    }
};

}
}
}

#endif
