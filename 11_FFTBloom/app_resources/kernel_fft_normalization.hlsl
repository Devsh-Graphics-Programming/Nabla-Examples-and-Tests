#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"

using namespace nbl::hlsl;

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
#define CHANNELS 3

groupshared uint32_t sharedmem[workgroup::fft::sharedMemSize<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

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

// Each Workgroup computes the FFT along two consecutive horizontal scanlines (fixed y for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the y coordinates for each of the consecutive lines
// Since the image is square of size IMAGE_SIDE_LENGTH (defined above) we will be launching half that amount of workgroups
struct PreloadedFirstAxisAccessor {
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

	template<uint32_t Channel>
	void preload()
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.y = (float32_t(2 * gl_WorkGroupID().x)+0.5f)/(inputImageSize*KERNEL_SCALE);
		normalizedCoordsSecondLine.y = (float32_t(2 * gl_WorkGroupID().x + 1)+0.5f)/(inputImageSize*KERNEL_SCALE);
		Promote<float32_t2, float32_t> promoter;

		const uint32_t stride = IMAGE_SIDE_LENGTH / 2; // Initial stride of global array in Forward FFT
		for (uint32_t virtualThreadID = workgroup::SubgroupContiguousIndex(); virtualThreadID < IMAGE_SIDE_LENGTH / 2; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
        {
            const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
			normalizedCoordsFirstLine.x = (float32_t(loIx)+0.5f)/(inputImageSize*KERNEL_SCALE);
			normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x;
			preloaded[loIx / _NBL_HLSL_WORKGROUP_SIZE_].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
			preloaded[loIx / _NBL_HLSL_WORKGROUP_SIZE_].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
            
			const uint32_t hiIx = loIx | stride;
			normalizedCoordsFirstLine.x = (float32_t(hiIx)+0.5f)/(inputImageSize*KERNEL_SCALE);
			normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x;
			preloaded[hiIx / _NBL_HLSL_WORKGROUP_SIZE_].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
			preloaded[hiIx / _NBL_HLSL_WORKGROUP_SIZE_].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
		}
	}

	// Utils to write values to output buffer in column major order
	uint32_t colMajorOffset(uint32_t x, uint32_t y)
	{
		return x * IMAGE_SIDE_LENGTH + y;
	}

	void storeColMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		vk::RawBufferStore<complex_t<scalar_t>>(startAddress + colMajorOffset(index, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>), value);
	}

	// Once the FFT is done, each thread should write its elements back. We want the storage to be in column-major order since the next FFT will be on y axis.
	// Channels will be contiguous in buffer memory. We only need to store outputs 0 through Nyquist, since the rest can be recovered via complex conjugation:
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
	// Also, elements 0 and Nyquist fit into a single complex element since they're both real, and it's always thread 0 holding these values
	template<uint32_t Channel>
	void unload()
	{
		// Each channel will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
		// for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
		const uint32_t channelStride = Channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		const uint32_t channelStartAddress = pushConstants.outputAddress + channelStride;
		
		// All but one subgroups are coherent - Thread 0 has different storing rules
		if (! workgroup::SubgroupContiguousIndex())
		{
			complex_t<scalar_t> packedZeroNyquist;
			packedZeroNyquist.real(preloaded[0].real());
			packedZeroNyquist.imag(preloaded[ELEMENTS_PER_THREAD / 2].real());
			storeColMajor(channelStartAddress, 0, packedZeroNyquist);
			for (uint32_t element = 1; element < ELEMENTS_PER_THREAD; element++)
			{
				storeColMajor(channelStartAddress, _NBL_HLSL_WORKGROUP_SIZE_ * element, preloaded[element]);
			}
		}
		else 
		{
			for (uint32_t element = 0; element < ELEMENTS_PER_THREAD; element++)
			{
				storeColMajor(channelStartAddress, _NBL_HLSL_WORKGROUP_SIZE_ * element + workgroup::SubgroupContiguousIndex(), preloaded[element]);
			}
		}
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};

void firstAxisFFT()
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload<channel>();
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload<channel>();
	}
}