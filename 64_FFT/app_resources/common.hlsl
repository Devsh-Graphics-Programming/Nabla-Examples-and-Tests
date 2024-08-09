#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// Unfortunately not every piece of C++14 metaprogramming syntax is available in HLSL 202x
// https://github.com/microsoft/DirectXShaderCompiler/issues/5751#issuecomment-1800847954
using scalar_t = nbl::hlsl::float32_t;

struct PushConstantData
{
	uint64_t inputAddress;
	uint64_t outputAddress;
	uint32_t dataElementCount;
};

#define _NBL_HLSL_WORKGROUP_SIZE_ 64
NBL_CONSTEXPR uint32_t WorkgroupSize = _NBL_HLSL_WORKGROUP_SIZE_;
NBL_CONSTEXPR uint32_t ElementsPerThread = 8;
NBL_CONSTEXPR uint32_t complexElementCount = WorkgroupSize * ElementsPerThread;

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"