#include "nbl/builtin/hlsl/cpp_compat.hlsl"

#define CHANNELS 3

using namespace nbl::hlsl;

struct PushConstantData
{
	// After running FFT along a column, we want to store the result in column major order for coalesced writes, and similarly after running an FFT in row major order
	// All FFTs that read from a buffer and write to a buffer (kernel second FFT, image second FFT + conv + IFFT) read in one order and write in another. 
	// Since workgroups start and finish at arbitrary times it's not possible to pickup the data from a buffer in one layout and write it back in another layout due to
	// possibility of writing over data that still hasn't been read by some workgroups.
	uint64_t colMajorBufferAddress;
	uint64_t rowMajorBufferAddress;
	// To save some work, we don't mirror the image along both directions when doing the FFT. This means that when doing the FFT along the second axis, we do an FFT of length
	// `RoundUpToPoT(dataElementCount + kernelPadding)` where `dataElementCount` is the actual length of the image along the second axis. We need it to keep track of the image's original dimension.
	uint32_t dataElementCount;
	float32_t2 kernelHalfPixelSize;
	uint32_t numWorkgroupsLog2;
};

#ifdef __HLSL_VERSION

#include "nbl/builtin/hlsl/complex.hlsl"
#include "nbl/builtin/hlsl/bda/legacy_bda_accessor.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

[[vk::push_constant]] PushConstantData pushConstants;

using namespace nbl::hlsl;

template<uint16_t ElementsPerThread, uint16_t WorkgroupSize, typename Scalar>
struct PreloadedAccessorBase {

	void set(uint32_t idx, complex_t<Scalar> value)
	{
		preloaded[idx / WorkgroupSize] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(complex_t<Scalar>) value)
	{
		value = preloaded[idx / WorkgroupSize];
	}

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	// TODO: Explain this a bit better in the readme
	// When unpacking a packed FFT of two real signals, to obtain `DFT[T]` you need `DFT[-T]`, `-T` being the mirror along Nyquist of `T`.
	// This funciton does the math to find out which thread holds `DFT[-T]` for the `T` (in DFT order) our thread is currently holding. 
	// Then since that thread is currently wanting to unpack `DFT[T']` for some `T' != T`, it turns out that rather miraculously (have not yet proven this) that our thread
	// happens to also hold `DFT[-T']`. So we do a shuffle to provide `DFT[-T']` to the other thread and get `DFT[-T]` from them.
	template<typename SharedmemAdaptor>
	complex_t<Scalar> getDFTMirror(uint32_t localElementIdx, SharedmemAdaptor sharedmemAdaptor)
	{
		uint32_t globalElementIdx = WorkgroupSize * localElementIdx | workgroup::SubgroupContiguousIndex();
		uint32_t otherElementIdx = workgroup::fft::FFTIndexingUtils<ElementsPerThread, WorkgroupSize>::getNablaMirrorIndex(globalElementIdx);
		uint32_t otherThreadID = otherElementIdx & (WorkgroupSize - 1);
		uint32_t otherThreadGlobalElementIdx = WorkgroupSize * localElementIdx | otherThreadID;
		uint32_t elementToTradeGlobalIdx = workgroup::fft::FFTIndexingUtils<ElementsPerThread, WorkgroupSize>::getNablaMirrorIndex(otherThreadGlobalElementIdx);
		uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / WorkgroupSize;
		complex_t<Scalar> toTrade = preloaded[elementToTradeLocalIdx];
		vector<Scalar, 2> toTradeVector = { toTrade.real(), toTrade.imag() };
		workgroup::Shuffle<SharedmemAdaptor, vector<Scalar, 2> >::__call(toTradeVector, otherThreadID, sharedmemAdaptor);
		toTrade.real(toTradeVector.x);
		toTrade.imag(toTradeVector.y);
		return toTrade;
	}

	complex_t<Scalar> preloaded[ElementsPerThread];
};

#endif
