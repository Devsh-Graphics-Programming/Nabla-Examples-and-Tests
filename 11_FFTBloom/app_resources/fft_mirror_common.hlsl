#include "fft_common.hlsl"

struct PreloadedAccessorMirrorTradeBase : PreloadedAccessorBase<FFTParameters> {
	
	// TODO: Explain this a bit better in the readme
	// Some operations require a thread to have both elements `DFT[T]` and `DFT[-T]` (where the latter is the mirror around Nyquist, or the "negative frequency" of T). For example,
	// this is needed when unpacking two different FFTs of real sequences `x, y` from the FFT of a single packed sequence `z = x + iy`. 
	// Suppose we are on a particular thread, we have an index `globalElementIdx` for a particular element (this index is an index into the Nabla-ordered array, not the proper DFT-ordered one)
	// If `NablaFFT[globalElementIdx] = DFT[T]` for some T, first we must find the `otherElementIdx` such that `NablaFFT[otherElementIdx] = DFT[-T]`. This way we know which is the other
	// element to get to do something such as unpacking. This is achieved via `FFTIndexingUtils::getNablaMirrorIndex`. 
	// Once we have the otherElementIdx, we must know which thread holds that other element. Thankfully, this is just the lower bits of the index. 
	// Now here's the deal with the rest of the function: Just like our thread has one element and expects to receive an element from the other thread to do an operation, the other thread
	// is currently doing exactly the same ("currently" here being up to an execution barrier because the other thread will usually be in another subgroup). So the other thread currently has an index
	// "otherThreadGlobalElementIdx" such that `NablaFFT[otherThreadGlobalElementIdx] = DFT[T']` for some `T' `,. And similarly, that other thread want to get the element at `elementToTradeGlobalIdx`
	// such that  `NablaFFT[elementToTradeGlobalIdx] = DFT[-T']`.  
	// Rather miraculously (haven't proven this yet) it is ALWAYS the case that at each step every two pair of threads trade their elements between themselves, 
	// so it's always the case that you give one element to a thread and receive one from the same thread. So for the current thread we find which of our elements the other thread expects to get
	// (since we know its global index, we divide by WorkgroupSize to get the element in the preloaded array). Finally we do a shuffle to trade these elements.
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
