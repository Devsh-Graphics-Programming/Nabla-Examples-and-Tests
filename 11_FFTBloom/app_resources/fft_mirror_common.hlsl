#include "fft_common.hlsl"

template<typename fft_consteval_parameters_t>
struct PreloadedAccessorMirrorTradeBase : PreloadedAccessorBase<fft_consteval_parameters_t> {
	
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
};
