#include "fft_mirror_common.hlsl"

// TODOS:
//        - You can get away with saving only half of the kernel (didn't do it here), especially if FFT of the image is always done in the same order (in that case you can just
//          store the same half of the kernel spectrum as you do the image's).

[[vk::binding(0,0)]] Texture2D<float32_t4> texture;
[[vk::binding(1,0)]] SamplerState samplerState;

// ---------------------------------------------------- Utils ---------------------------------------------------------

uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * FFTParameters::TotalSize | y;
}

// Each channel after first FFT will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
// for N = FFTParameters::TotalSize
uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * FFTParameters::TotalSize * FFTParameters::TotalSize / 2 * sizeof(complex_t<scalar_t>);
}

// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines
// Since the output images (one per channel) are square of size ConstevalParameters::TotalSize (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : PreloadedAccessorMirrorTradeBase
{
	NBL_CONSTEXPR_STATIC_INLINE float32_t KernelScale = ConstevalParameters::KernelScale;
	NBL_CONSTEXPR_STATIC_INLINE float32_t2 KernelDimensions;
	
	void preload(uint32_t channel)
	{
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		// Good compiler turns this into a single FMA
		normalizedCoordsFirstLine.x = float32_t(glsl::gl_WorkGroupID().x) * 2 / (KernelDimensions.x * KernelScale) + 0.5f / (KernelDimensions.x * KernelScale);
		normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x + 1 / (KernelDimensions.x * KernelScale);

		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
			const uint32_t index = localElementIndex * WorkgroupSize | workgroup::SubgroupContiguousIndex();
			normalizedCoordsFirstLine.y = float32_t(index) / (KernelDimensions.y * KernelScale) + 0.5f / (KernelDimensions.y * KernelScale);
			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			preloaded[localElementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KernelScale), 0)[channel]));
			preloaded[localElementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KernelScale), 0)[channel]));
		}

		// Set LegacyBdaAccessor for posterior writing
		colMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(getChannelStartAddress(channel));
	}

	// Util to write values to output buffer in column major order - this ensures coalesced writes
	void storeColMajor(uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		colMajorAccessor.set(colMajorOffset(glsl::gl_WorkGroupID().x, index), value);
	}

	// Once the FFT is done, each thread should write its elements back. Storage is in column-major order because this avoids cache misses when writing.
	// Channels will be contiguous in buffer memory.
	template<typename sharedmem_adaptor_t>
	void unload(uint32_t channel, sharedmem_adaptor_t adaptorForSharedMemory)
	{
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			storeColMajor(localElementIndex * WorkgroupSize | workgroup::SubgroupContiguousIndex(), preloaded[localElementIndex]);

		}
	}
	LegacyBdaAccessor<complex_t<scalar_t> > colMajorAccessor;
};
NBL_CONSTEXPR_STATIC_INLINE float32_t2 PreloadedFirstAxisAccessor::KernelDimensions = ConstevalParameters::KernelDimensions;

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using sharedmem_adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, FFTParameters::WorkgroupSize>;
	sharedmem_adaptor_t adaptorForSharedMemory;

	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
		// Update state after FFT run
		adaptorForSharedMemory.accessor = sharedmemAccessor;
		preloadedAccessor.unload(channel, adaptorForSharedMemory);
		// Remember to update the accessor's state
		sharedmemAccessor = adaptorForSharedMemory.accessor;
	}
}