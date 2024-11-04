#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

// TODO: remove usage of KERNEL_SCALE in image FFT stuff

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#else
#define scalar_t float32_t
#endif

[[vk::push_constant]] PushConstantData pushConstants;
// Can't specify format
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] /* [[vk::image_format(rgba16f)]]*/ Texture2D<float32_t4> texture;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

#define FFT_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

groupshared uint32_t sharedmem[workgroup::fft::SharedMemoryDWORDs<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1); }

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

// ---------------------------------------------------- Utils ---------------------------------------------------------

// After first FFT we only store half of a column, so the offset per column is half the image side length
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (FFT_LENGTH / 2) | y;
}

// We store half of a column, but unlike the kernel we only store as many columns as there actually are in the image
// We launch one workgroup every two columns in the image, and we only write (per channel) one column per column in the image
// This is unlike the kernel, in which we always upsample it to nearest PoT
// So there are `2 * glsl::gl_NumWorkGroups().x` columns per channel, each of size `FFT_LENGTH / 2` (we only store half) and the 2's cancel out
uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * FFT_LENGTH * sizeof(complex_t<scalar_t>);
}

struct PreloadedAccessorBase {

	void set(uint32_t idx, nbl::hlsl::complex_t<scalar_t> value)
	{
		preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<scalar_t>) value)
	{
		value = preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_];
	}

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};


// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines. This time we launch `inputImageSize.x / 2` workgroups 

struct PreloadedFirstAxisAccessor : PreloadedAccessorBase {

	void preload(uint32_t channel)
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.x = (float32_t(2 * glsl::gl_WorkGroupID().x) + 0.5f) / (inputImageSize.x * KERNEL_SCALE);
		normalizedCoordsSecondLine.x = (float32_t(2 * glsl::gl_WorkGroupID().x + 1) + 0.5f) / (inputImageSize.x * KERNEL_SCALE);

		// Remember to add padding before and after - we will be sampling the original image mirrored at the borders - this avoids loss of brightness at the edges
		const uint32_t padding = uint32_t(FFT_LENGTH - inputImageSize.y) >> 1;

		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			normalizedCoordsFirstLine.y = (float32_t(index) - padding + 0.5f) / (inputImageSize.y * KERNEL_SCALE);
			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			preloaded[localElementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KERNEL_SCALE), 0)[channel]));
			preloaded[localElementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KERNEL_SCALE), 0)[channel]));
			if (normalizedCoordsFirstLine.y < 0.f || normalizedCoordsFirstLine.y > 1.f)
			{
				vector<scalar_t, 2> aux = { preloaded[localElementIndex].real(), preloaded[localElementIndex].imag() };
				aux = saturate(aux);
				preloaded[localElementIndex].real(aux.x);
				preloaded[localElementIndex].imag(aux.y);
			}
		}
	}

	// Util to write values to output buffer in column major order - this ensures coalesced writes
	void storeColMajor(uint64_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) firstValue, NBL_CONST_REF_ARG(complex_t<scalar_t>) secondValue)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(2 * glsl::gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>), firstValue);
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, index) * sizeof(complex_t<scalar_t>), secondValue);
	}

	template<typename Scalar, typename SharedmemAdaptor>
	complex_t<Scalar> trade(uint32_t localElementIdx, SharedmemAdaptor sharedmemAdaptor)
	{
		uint32_t globalElementIdx = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex();
		uint32_t otherElementIdx = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(globalElementIdx);
		uint32_t otherThreadID = otherElementIdx & (_NBL_HLSL_WORKGROUP_SIZE_ - 1);
		uint32_t otherThreadGlobalElementIdx = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | otherThreadID;
		uint32_t elementToTradeGlobalIdx = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(otherThreadGlobalElementIdx);
		uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / _NBL_HLSL_WORKGROUP_SIZE_;
		complex_t<Scalar> toTrade = preloaded[elementToTradeLocalIdx];
		vector<Scalar, 2> toTradeVector = { toTrade.real(), toTrade.imag() };
		workgroup::Shuffle<SharedmemAdaptor, vector<Scalar, 2> >::__call(toTradeVector, otherThreadID, sharedmemAdaptor);
		toTrade.real(toTradeVector.x);
		toTrade.imag(toTradeVector.y);
		return toTrade;
	}

	// Once the FFT is done, each thread should write its elements back. Storage is in column-major order because this avoids cache misses when writing.
	// Channels will be contiguous in buffer memory. We only need to store outputs 0 through Nyquist, since the rest can be recovered via complex conjugation:
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
	// FFT unpacking rules explained in the readme
	// Also, elements 0 and Nyquist fit into a single complex element since they're both real, and it's always thread 0 holding these values
	template<typename SharedmemAdaptor>
	void unload(uint32_t channel, SharedmemAdaptor sharedmemAdaptor)
	{
		const uint64_t startAddress = getChannelStartAddress(channel);

		// Storing even elements of NFFT is storing the bitreversed lower half of DFT - see readme
		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex += 2)
		{
			// First element of 0th thread has a special packing rule
			if (!workgroup::SubgroupContiguousIndex() && !localElementIndex)
			{
				complex_t<scalar_t> packedZeroNyquistLo = { preloaded[0].real(), preloaded[1].real() };
				complex_t<scalar_t> packedZeroNyquistHi = { preloaded[0].imag(), preloaded[1].imag() };
				storeColMajor(startAddress, 0, packedZeroNyquistLo, packedZeroNyquistHi);
			}
			else
			{
				complex_t<scalar_t> lo = preloaded[localElementIndex];
				complex_t<scalar_t> hi = trade<scalar_t, SharedmemAdaptor>(localElementIndex, sharedmemAdaptor);
				workgroup::fft::unpack<scalar_t>(lo, hi);
				// Divide localElementIdx by 2 to keep even elements packed together when writing
				storeColMajor(startAddress, _NBL_HLSL_WORKGROUP_SIZE_ * (localElementIndex >> 1) | workgroup::SubgroupContiguousIndex(), lo, hi);
			}
		}

	}
};

void firstAxisFFT()
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, _NBL_HLSL_WORKGROUP_SIZE_>;
	adaptor_t sharedmemAdaptor;

	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		// Update state after FFT run
		sharedmemAdaptor.accessor = sharedmemAccessor;
		preloadedAccessor.unload<adaptor_t>(channel, sharedmemAdaptor);
		// Remember to update the accessor's state
		sharedmemAccessor = sharedmemAdaptor.accessor;
	}
}

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	firstAxisFFT();
}