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

template<typename fft_consteval_parameters_t>
struct PreloadedAccessorBase {

	using scalar_t = typename fft_consteval_parameters_t::scalar_t;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocationLog2 = fft_consteval_parameters_t::ElementsPerInvocationLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSizeLog2 = fft_consteval_parameters_t::WorkgroupSizeLog2;

	NBL_CONSTEXPR_STATIC_INLINE uint16_t ElementsPerInvocation = fft_consteval_parameters_t::ElementsPerInvocation;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t WorkgroupSize = fft_consteval_parameters_t::WorkgroupSize;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t TotalSize = fft_consteval_parameters_t::TotalSize;

	void set(uint32_t idx, complex_t<scalar_t> value)
	{
		preloaded[idx >> WorkgroupSizeLog2] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(complex_t<scalar_t>) value)
	{
		value = preloaded[idx >> WorkgroupSizeLog2];
	}

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ElementsPerInvocation];
};