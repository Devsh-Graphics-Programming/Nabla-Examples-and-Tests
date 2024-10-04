#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"

/*
 * Remember we have these defines: 
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

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
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * IMAGE_SIDE_LENGTH | y;
}

// Each channel after first FFT will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
// for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
uint32_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.outputAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

struct PreloadedAccessorBase {
	
	void set(uint32_t idx, nbl::hlsl::complex_t<scalar_t> value) 
	{
		preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_] = value;
	}
	
	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<scalar_t>) value) 
	{
		value = preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_]
	}

	void memoryBarrier() 
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};


// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive horizontal scanlines (fixed y for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the y coordinates for each of the consecutive lines
// Since the image is square of size IMAGE_SIDE_LENGTH (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : PreloadedAccessorBase {
	
	void preload(uint32_t channel)
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.y = (float32_t(2 * gl_WorkGroupID().x)+0.5f)/(inputImageSize*KERNEL_SCALE);
		normalizedCoordsSecondLine.y = (float32_t(2 * gl_WorkGroupID().x + 1)+0.5f)/(inputImageSize*KERNEL_SCALE);
		Promote<float32_t2, float32_t> promoter;

		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			normalizedCoordsFirstLine.x = (float32_t(index) + 0.5f) / (inputImageSize * KERNEL_SCALE);
			normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x;
			preloaded[localElementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promoter(0.5 - 0.5 / KERNEL_SCALE), -log2(KERNEL_SCALE))[channel]));
			preloaded[localElementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[channel]));
		}
	}

	// Util to write values to output buffer in column major order
	void storeColMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) firstValue, NBL_CONST_REF_ARG(complex_t<scalar_t>) secondValue)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(index, 2 * gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>), firstValue);
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + colMajorOffset(index, 2 * gl_WorkGroupID().x + 1) * sizeof(complex_t<scalar_t>), secondValue);
	}

	// Once the FFT is done, each thread should write its elements back. We want the storage to be in column-major order since the next FFT will be on y axis.
	// Channels will be contiguous in buffer memory. We only need to store outputs 0 through Nyquist, since the rest can be recovered via complex conjugation:
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
	// FFT unpacking rules explained in the readme
	// Also, elements 0 and Nyquist fit into a single complex element since they're both real, and it's always thread 0 holding these values
	template<typename SharedmemAdaptor>
	void unload(uint32_t channel, SharedmemAdaptor sharedmemAdaptor)
	{
		const uint32_t startAddress = getChannelStartAddress(channel);

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
				unpack<scalar_t>(lo, hi);
				storeColMajor(startAddress, _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex(), lo, hi);
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
	sharedmemAdaptor.accessor = sharedmemAccessor;

	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload(channel);
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload<adaptor_t>(channel, sharedmemAdaptor);
	}
	// Remember to update the accessor's state
	sharedmemAccessor = sharedmemAdaptor.accessor;
}

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	firstAxisFFT();
}