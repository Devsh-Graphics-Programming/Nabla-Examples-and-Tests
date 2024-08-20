#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"

/*
 * Remember we have these defines: 
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * USE_HALF_PRECISION
 * KERNEL_SCALE
*/

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] Texture2D texture;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] SamplerState samplerState;

#ifdef USE_HALF_PRECISION
#define UINTS_PER_COMPLEX 1
#define scalar_t float16_t
#else
#define UINTS_PER_COMPLEX 2
#define scalar_t float32_t
#endif

groupshared uint32_t sharedmem[UINTS_PER_COMPLEX * _NBL_HLSL_WORKGROUP_SIZE_];

// Users MUST define this method for FFT to work
namespace nbl { namespace hlsl { namespace glsl{
uint32_t3 gl_WorkGroupSize() { return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1); }
} } }

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
		AllMemoryBarrierWithGroupSync();
    }

};

// Each Workgroup computes the FFT along a scanline (fixed x for the whole Workgroup) so we use `gl_WorkGroupID().x` to get the x coordinate for sampling
// For now leaving as old GLSL example, sampling is done one time per channel. Maybe could be changed so that all channels are preloaded on a single sample instead
struct PreloadedAccessor {
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
		AllMemoryBarrier();
	}

	template<uint32_t Channel>
	void preload()
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoords;
		normalizedCoords.y = (float32_t(gl_WorkGroupID().x)+0.5f)/(inputImageSize*KERNEL_SCALE);
		Promote<float32_t2, float32_t> promoter;

		const uint32_t stride = (ELEMENTS_PER_THREAD / 2) * _NBL_HLSL_WORKGROUP_SIZE_; // Initial stride of global array in Forward FFT
		for (uint32_t virtualThreadID = SubgroupContiguousIndex(); virtualThreadID < (ELEMENTS_PER_THREAD / 2) * _NBL_HLSL_WORKGROUP_SIZE_; virtualThreadID += _NBL_HLSL_WORKGROUP_SIZE_)
        {
            const uint32_t loIx = ((virtualThreadID & (~(stride - 1))) << 1) | (virtualThreadID & (stride - 1));
			normalizedCoords.x = (float32_t(loIx)+0.5f)/(inputImageSize*KERNEL_SCALE);
			if (Channel == 0)
			preloaded[loIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).r;
			if (Channel == 1)
			preloaded[loIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).g;
			if (Channel == 2)
			preloaded[loIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).b;
            
			const uint32_t hiIx = loIx | stride;
			normalizedCoords.x = (float32_t(hiIx)+0.5f)/(inputImageSize*KERNEL_SCALE);
			if (Channel == 0)
			preloaded[hiIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).r;
			if (Channel == 1)
			preloaded[hiIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).g;
			if (Channel == 2)
			preloaded[hiIx / _NBL_HLSL_WORKGROUP_SIZE_] = texture.SampleLevel(samplerState, normalizedCoords + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE)).b;
		}
	}

	// Once the FFT is done, each thread should write its elements back. We want the storage to be in column-major order since the next FFT will be on y axis.
	template<uint32_t Channel>
	void unload()
	{
		// Each channel will be stored as the whole image in col-major order, and its size is N^2 for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
		const uint32_t channelStride = Channel * _NBL_HLSL_WORKGROUP_SIZE_ * _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD * ELEMENTS_PER_THREAD; 

	}

	nbl::hlsl::complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};