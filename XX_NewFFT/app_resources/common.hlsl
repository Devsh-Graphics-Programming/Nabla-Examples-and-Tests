#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup2/fft.hlsl"

using scalar_t = nbl::hlsl::float32_t;

struct PushConstantData
{
	uint64_t deviceBufferAddress;
};

NBL_CONSTEXPR uint32_t WorkgroupSizeLog2 = 9;
NBL_CONSTEXPR uint32_t WorkgroupSize = uint32_t(1) << WorkgroupSizeLog2;
NBL_CONSTEXPR uint32_t SubgroupSizeLog2 = 5; // hardcoded to my nvidia gpu, should be queried at compilation time
NBL_CONSTEXPR uint32_t SubgroupSize = uint32_t(1) << SubgroupSizeLog2;

NBL_CONSTEXPR uint32_t Radix2ElementsPerInvocationLog2 = 2;
NBL_CONSTEXPR uint32_t ExtraPrimeFactor = uint32_t(5);
NBL_CONSTEXPR uint32_t ElementsPerThread = ExtraPrimeFactor * (uint32_t(1) << Radix2ElementsPerInvocationLog2);

NBL_CONSTEXPR uint32_t complexElementCount = ElementsPerThread * (uint32_t(1) << WorkgroupSizeLog2);
NBL_CONSTEXPR uint32_t complexElementCountPerChannel = uint32_t(1) << (WorkgroupSizeLog2 + 1);
NBL_CONSTEXPR uint32_t Channels = ElementsPerThread / 2;

NBL_CONSTEXPR uint32_t ShuffledChannelsPerRound = nbl::hlsl::mpl::min_v<uint32_t, Channels, 4>;