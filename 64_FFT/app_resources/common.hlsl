#include "nbl/builtin/hlsl/cpp_compat.hlsl"

using scalar_t = nbl::hlsl::float32_t;

struct PushConstantData
{
	uint64_t inputAddress;
	uint64_t outputAddress;
	uint32_t dataElementCount;
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 64;
NBL_CONSTEXPR uint32_t ElementsPerThread = 8;
NBL_CONSTEXPR uint32_t complexElementCount = WorkgroupSize * ElementsPerThread;