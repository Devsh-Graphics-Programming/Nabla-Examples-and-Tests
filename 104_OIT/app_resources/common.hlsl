#ifndef _OIT_COMMON_HLSL_
#define _OIT_COMMON_HLSL_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct SVertex
{
    nbl::hlsl::float32_t2 position;
    nbl::hlsl::float32_t4 color;
};

struct Smlab
{
    nbl::hlsl::float32_t depth;
    uint32_t colorDepth;
};

#ifndef __HLSL_VERSION
static_assert(sizeof(SVertex) == 24);
static_assert(sizeof(Smlab) == 8);
#endif

namespace nbl
{
namespace hlsl
{
namespace this_example
{

struct SPushConstants
{
    float32_t4x4 viewProj;
    uint64_t bdaAddress;
    uint32_t width;
};

}
}
}

#endif
