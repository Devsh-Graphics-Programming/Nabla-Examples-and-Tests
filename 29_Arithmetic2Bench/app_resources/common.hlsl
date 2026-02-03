#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

struct PushConstantData
{
    uint64_t pOutputBuf[2];
};

namespace arithmetic
{
template<typename T>
struct plus : nbl::hlsl::plus<T>
{
    using base_t = nbl::hlsl::plus<T>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 0;
#ifndef __HLSL_VERSION
    static inline constexpr const char* name = "plus";
#endif
};

template<typename T>
struct ballot : nbl::hlsl::plus<T>
{
    using base_t = nbl::hlsl::plus<T>;

    NBL_CONSTEXPR_STATIC_INLINE uint16_t BindingIndex = 1;
#ifndef __HLSL_VERSION
    static inline constexpr const char* name = "bitcount";
#endif
};
}

#include "nbl/builtin/hlsl/glsl_compat/subgroup_basic.hlsl"
