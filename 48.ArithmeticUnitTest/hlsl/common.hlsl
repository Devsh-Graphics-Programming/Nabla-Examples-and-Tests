#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"

template<uint32_t kScanDwordCount=256*1024>
struct Output
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t ScanDwordCount = kScanDwordCount;

	uint32_t subgroupSize;
	uint32_t output[ScanDwordCount];
};