#include "nbl/builtin/hlsl/cpp_compat.hlsl"

using scalar_t = nbl::hlsl::float32_t;

struct PushConstantData
{
	uint64_t deviceBufferAddress;
	uint32_t dataElementCount;
};

NBL_CONSTEXPR uint32_t WorkgroupSizeLog2 = 6;
NBL_CONSTEXPR uint32_t ElementsPerThreadLog2 = 3;
NBL_CONSTEXPR uint32_t complexElementCount = uint32_t(1) << (WorkgroupSizeLog2 + ElementsPerThreadLog2);