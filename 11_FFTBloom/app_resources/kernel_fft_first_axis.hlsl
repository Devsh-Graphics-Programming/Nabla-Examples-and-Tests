#include "fft_common.hlsl"

// TODOS:
//        - You can get away with saving only half of the kernel (didn't do it here), especially if FFT of the image is always done in the same order (in that case you can just
//          store the same half of the kernel spectrum as you do the image's).

[[vk::binding(0, 0)]] Texture2D<float32_t4> texture;
[[vk::binding(1, 0)]] SamplerState samplerState;

// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines
// Since the output images (one per channel) are square of size ConstevalParameters::TotalSize (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : MultiChannelPreloadedAccessorBase
{
	NBL_CONSTEXPR_STATIC_INLINE float32_t KernelScale = ConstevalParameters::KernelScale;
	NBL_CONSTEXPR_STATIC_INLINE float32_t2 KernelDimensions;

	// ---------------------------------------------------- Utils ---------------------------------------------------------

	uint32_t colMajorOffset(uint32_t x, uint32_t y)
	{
		return x * TotalSize | y;
	}

	// Each channel after first FFT will be stored as half the image (every two columns have been packed into one column of complex numbers) in col-major order, and the whole size of the image is N^2, 
	// for N = TotalSize
	uint64_t getChannelStartOffsetBytes(uint16_t channel)
	{
		return uint64_t(channel) * TotalSize * TotalSize / 2 * sizeof(complex_t<scalar_t>);
	}

	// ---------------------------------------------------- End Utils ---------------------------------------------------------

	void preload()
	{
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		// Good compiler turns this into a single FMA
		normalizedCoordsFirstLine.x = float32_t(glsl::gl_WorkGroupID().x) * 2.f / (KernelDimensions.x * KernelScale) + 0.5f / (KernelDimensions.x * KernelScale);
		normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x + 1.f / (KernelDimensions.x * KernelScale);

		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
			const uint32_t index = localElementIndex * WorkgroupSize | workgroup::SubgroupContiguousIndex();

			normalizedCoordsFirstLine.y = float32_t(index) / (KernelDimensions.y * KernelScale) + 0.5f / (KernelDimensions.y * KernelScale);
			const float32_t4 firstLineTexValue = texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KernelScale), 0);
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].real(scalar_t(firstLineTexValue[channel]));

			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			const float32_t4 secondLineTexValue = texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promote<float32_t2, float32_t>(0.5 - 0.5 / KernelScale), 0);
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].imag(scalar_t(secondLineTexValue[channel]));
		}
	}

	// Once the FFT is done, each thread should write its elements back. Storage is in column-major order because this avoids cache misses when writing.
	// Channels will be contiguous in buffer memory.
	void unload()
	{
		for (uint16_t channel = 0; channel < Channels; channel++)
		{
			const uint64_t channelStartOffsetBytes = getChannelStartOffsetBytes(channel);
			const LegacyBdaAccessor<complex_t<scalar_t> > colMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(pushConstants.colMajorBufferAddress + channelStartOffsetBytes);

			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				const uint32_t globalElementIdx = localElementIndex * WorkgroupSize | workgroup::SubgroupContiguousIndex();
				colMajorAccessor.set(colMajorOffset(glsl::gl_WorkGroupID().x, globalElementIdx), preloaded[channel][localElementIndex]);
			}
		}
	}
};
NBL_CONSTEXPR_STATIC_INLINE float32_t2 PreloadedFirstAxisAccessor::KernelDimensions = ConstevalParameters::KernelDimensions;

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedFirstAxisAccessor preloadedAccessor;

	preloadedAccessor.preload();
	for (uint16_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.currentChannel = channel;
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	}
	preloadedAccessor.unload();
}