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

NBL_CONSTEXPR uint32_t Radix2ElementsPerInvocationPerChannelLog2 = 1;
NBL_CONSTEXPR uint32_t ExtraPrimeFactor = uint32_t(1);
NBL_CONSTEXPR uint32_t ElementsPerInvocationPerChannel = ExtraPrimeFactor * (uint32_t(1) << Radix2ElementsPerInvocationPerChannelLog2);

NBL_CONSTEXPR uint32_t Channels = 1;
NBL_CONSTEXPR uint32_t complexElementCountPerChannel = ElementsPerInvocationPerChannel * (uint32_t(1) << WorkgroupSizeLog2);
NBL_CONSTEXPR uint32_t complexElementCount = Channels * complexElementCountPerChannel;

NBL_CONSTEXPR uint16_t InnerVirtualChannels = Channels * (ElementsPerInvocationPerChannel >> 1);
NBL_CONSTEXPR uint32_t ShuffledVirtualChannelsPerRound = nbl::hlsl::mpl::min_v<uint32_t, InnerVirtualChannels, 4>;

NBL_CONSTEXPR bool ShareTwiddles = true;

NBL_CONSTEXPR uint16_t DivisionPolicyForward = workgroup2::fft::DivisionPolicy::DivBySqrtHalfway;
NBL_CONSTEXPR uint16_t DivisionPolicyInverse = workgroup2::fft::DivisionPolicy::DivBySqrtHalfway;

using ConstevalParametersForward = workgroup2::fft::ConstevalParameters<ElementsPerInvocationPerChannel, Channels, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledVirtualChannelsPerRound, false, true, DivisionPolicyForward, scalar_t>;
using ConstevalParametersInverse = workgroup2::fft::ConstevalParameters<ElementsPerInvocationPerChannel, Channels, SubgroupSizeLog2, WorkgroupSizeLog2, ShuffledVirtualChannelsPerRound, false, true, DivisionPolicyInverse, scalar_t>;