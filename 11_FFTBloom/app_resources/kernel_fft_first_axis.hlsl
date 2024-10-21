#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

/*
 * Remember we have these defines: 
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

// TODOS:
//        - You can get away with saving only half of the kernel (didn't do it here), especially if FFT of the image is always done in the same order (in that case you can just
//          store the same half of the kernel spectrum as you do the image's).

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] Texture2D texture;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] SamplerState samplerState;

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#else
#define scalar_t float32_t
#endif

#define IMAGE_SIDE_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

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
uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (IMAGE_SIDE_LENGTH / 2) | y;
}

// Each channel after first FFT will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
// for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
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
// to get the x coordinates for each of the consecutive lines
// Since the output images (one per channel) are square of size IMAGE_SIDE_LENGTH (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : PreloadedAccessorBase {
	
	void preload(uint32_t channel)
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.x = (float32_t(2 * glsl::gl_WorkGroupID().x)+0.5f)/(inputImageSize.x*KERNEL_SCALE);
		normalizedCoordsSecondLine.x = (float32_t(2 * glsl::gl_WorkGroupID().x + 1)+0.5f)/(inputImageSize.x*KERNEL_SCALE);

		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			normalizedCoordsFirstLine.y = (float32_t(index) + 0.5f) / (inputImageSize.y * KERNEL_SCALE);
			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			preloaded[localElementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KERNEL_SCALE), -log2(KERNEL_SCALE))[channel]));
			preloaded[localElementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promote<float32_t2, float32_t>(0.5 - 0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[channel]));
		}
	}

	// Util to write values to output buffer in column major order - this ensures coalesced writes
	void storeColMajor(uint64_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) firstValue, NBL_CONST_REF_ARG(complex_t<scalar_t>) secondValue)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(2 * glsl::gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>), firstValue);
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, index) * sizeof(complex_t<scalar_t>), secondValue);
	}

	// TODO: Explain this a bit better in the readme
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