#ifndef _BITONIC_SORT_COMMON_INCLUDED_
#define _BITONIC_SORT_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PushConstantData
{
	uint64_t deviceBufferAddress;
};

NBL_CONSTEXPR uint32_t WorkgroupSizeLog2 = 10;  // 1024 threads (2^10)
NBL_CONSTEXPR uint32_t ElementsPerThreadLog2 = 2;  // 4 elements per thread (2^2) - VIRTUAL THREADING!
NBL_CONSTEXPR uint32_t elementCount = uint32_t(1) << (WorkgroupSizeLog2 + ElementsPerThreadLog2);  // 4096 elements (2^12)
#endif