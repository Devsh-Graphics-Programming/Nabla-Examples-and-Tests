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

	// TODO: Explain this a bit better in the readme
	// When unpacking a packed FFT of two real signals, to obtain `DFT[T]` you need `DFT[-T]`, `-T` being the mirror along Nyquist of `T`.
	// This funciton does the math to find out which thread holds `DFT[-T]` for the `T` (in DFT order) our thread is currently holding. 
	// Then since that thread is currently wanting to unpack `DFT[T']` for some `T' != T`, it turns out rather miraculously (have not yet proven this) that our thread
	// happens to also hold `DFT[-T']`. So we do a shuffle to provide `DFT[-T']` to the other thread and get `DFT[-T]` from them.
	template<typename SharedmemAdaptor>
	complex_t<scalar_t> getDFTMirror(uint32_t localElementIdx, SharedmemAdaptor sharedmemAdaptor)
	{
		uint32_t globalElementIdx = localElementIdx * WorkgroupSize | workgroup::SubgroupContiguousIndex();
		uint32_t otherElementIdx = FFTIndexingUtils::getNablaMirrorIndex(globalElementIdx);
		uint32_t otherThreadID = otherElementIdx & (WorkgroupSize - 1);
		uint32_t otherThreadGlobalElementIdx = localElementIdx * WorkgroupSize | otherThreadID;
		uint32_t elementToTradeGlobalIdx = FFTIndexingUtils::getNablaMirrorIndex(otherThreadGlobalElementIdx);
		uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / WorkgroupSize;
		complex_t<scalar_t> toTrade = preloaded[elementToTradeLocalIdx];
		vector<scalar_t, 2> toTradeVector = { toTrade.real(), toTrade.imag() };
		workgroup::Shuffle<SharedmemAdaptor, vector<scalar_t, 2> >::__call(toTradeVector, otherThreadID, sharedmemAdaptor);
		toTrade.real(toTradeVector.x);
		toTrade.imag(toTradeVector.y);
		return toTrade;
	}

	complex_t<scalar_t> preloaded[ElementsPerInvocation];
};