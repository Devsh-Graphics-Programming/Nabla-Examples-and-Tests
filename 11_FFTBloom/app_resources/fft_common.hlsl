#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

groupshared uint32_t sharedmem[FFTParameters::SharedMemoryDWORDs];

struct SharedMemoryAccessor
{
	void set(uint32_t idx, uint32_t value)
	{
		sharedmem[idx] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(uint32_t) value)
	{
		value = sharedmem[idx];
	}

	void workgroupExecutionAndMemoryBarrier()
	{
		glsl::barrier();
	}

};

struct PreloadedAccessorCommonBase
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = FFTParameters::ElementsPerInvocationLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = FFTParameters::WorkgroupSizeLog2;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = FFTParameters::ElementsPerInvocation;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = FFTParameters::WorkgroupSize;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t TotalSize = FFTParameters::TotalSize;

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}
};

struct PreloadedAccessorBase : PreloadedAccessorCommonBase
{
	void set(uint32_t idx, complex_t<scalar_t> value)
	{
		preloaded[idx >> WorkgroupSizeLog2] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(complex_t<scalar_t>) value)
	{
		value = preloaded[idx >> WorkgroupSizeLog2];
	}

	complex_t<scalar_t> preloaded[ElementsPerInvocation];
};

// In the case for preloading all channels at once we make it stateful so we track which channel we're running FFT on
struct MultiChannelPreloadedAccessorBase : PreloadedAccessorCommonBase
{
	void set(uint32_t idx, complex_t<scalar_t> value)
	{
		preloaded[currentChannel][idx >> WorkgroupSizeLog2] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(complex_t<scalar_t>) value)
	{
		value = preloaded[currentChannel][idx >> WorkgroupSizeLog2];
	}

	complex_t<scalar_t> preloaded[Channels][ElementsPerInvocation];
	uint16_t currentChannel;
};