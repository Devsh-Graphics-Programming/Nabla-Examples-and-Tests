#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

using namespace nbl::hlsl;

#define CHANNELS 3

struct PushConstantData
{
	uint64_t inputAddress;
	uint64_t outputAddress;
	uint32_t dataElementCount;
};

// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
template<typename Scalar>
void unpack(NBL_CONST_REF_ARG(complex_t<Scalar>) lo, NBL_CONST_REF_ARG(complex_t<Scalar>) hi)
{
	complex_t<Scalar> x = (lo + conj(hi)) * Scalar(0.5);
	hi = rotateRight<Scalar>(lo - conj(hi)) * 0.5;
	lo = x;
}

// Util to trade values between threads, needed for FFT unpacking. We're going to abuse the SharedmemAccessor :)
// An even local element of `localElementIdx` contains the element of `T = globalElementIdx = WorkgroupSize * localElementIdx + threadID` in the global array. 
// Then to unpack we need element at `U = F^{-1}(-F(T))` (see readme). Turns out U's lower `log2(WorkgroupSize)` bits give the ID `otherThreadID` of the thread holding U. 
// That thread has to store element `V = WorkgroupSize * localElementIdx + otherThreadID`, for which it needs access to element `W = F^{-1}(-F(V))`. Rather surprisingly,
// but yet unproven, W's lower bits give as ID the current thread's ID. That means that the other thread expects one of our elements to unpack, just like we expect one of their
// elements. This discussion is again in the readme, but it turns out W's upper bits give its local element index.
template<typename Scalar, typename SharedmemAdaptor>
complex_t<Scalar> trade(uint32_t localElementIdx, SharedmemAdaptor sharedmemAdaptor)
{
	uint32_t globalElementIdx = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex();
	uint32_t otherElementIdx = workgroup::fft::getOutputIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(-workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(globalElementIdx));
	uint32_t otherThreadID = otherElementIdx & (_NBL_HLSL_WORKGROUP_SIZE_ - 1);
	uint32_t elementToTradeGlobalIdx = workgroup::fft::getOutputIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(-workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(otherElementIdx));
	uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / _NBL_HLSL_WORKGROUP_SIZE_;
	complex_t<Scalar> toTrade = preloaded[elementToTradeLocalIdx];
	workgroup::Shuffle<SharedmemAdaptor, complex_t<Scalar> >::__call(toTrade, otherThreadID, sharedmemAdaptor);
	return toTrade;
}