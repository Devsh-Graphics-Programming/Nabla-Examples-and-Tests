#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// Unfortunately not every piece of C++14 metaprogramming syntax is available in HLSL 202x
// https://github.com/microsoft/DirectXShaderCompiler/issues/5751#issuecomment-1800847954
typedef nbl::hlsl::float32_t3 input_t;
typedef nbl::hlsl::float32_t output_t;

NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxPossibleElementCount = 1 << 20;

struct PushConstantData
{
	uint64_t inputAddress;
	uint64_t outputAddress;
	uint32_t dataElementCount;
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"