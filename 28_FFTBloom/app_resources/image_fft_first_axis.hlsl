#include "fft_common.hlsl"

[[vk::binding(0, 0)]] Texture2D<float32_t4> texture;
[[vk::binding(4, 0)]] SamplerState samplerState;
// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines. This time we launch `inputImageSize.x / 2` workgroups 

// After first axis FFT we store results packed. This is because unpacking on load is faster: Unpacking on store would require many workgroup shuffles, which incur a barrier
struct PreloadedFirstAxisAccessor : MultiChannelPreloadedAccessorBase
{
	// ---------------------------------------------------- Utils ---------------------------------------------------------

	// After first FFT we only store half of a column, so the offset per column is half the image side length
	uint32_t colMajorOffset(uint32_t x, uint32_t y)
	{
		return x * TotalSize | y;
	}

	// Each channel after first FFT will be stored as half the image (every two columns have been packed into one column of complex numbers) in col-major order.
	// We launch one workgroup every two columns in the image, so there are `glsl::gl_NumWorkGroups().x` columns (of complex elements) per channel, 
	// each of size `FFTParameters::TotalSize`
	uint64_t getChannelStartOffsetBytes(uint16_t channel)
	{
		return uint64_t(channel) * glsl::gl_NumWorkGroups().x * TotalSize * sizeof(complex_t<scalar_t>);
	}

	// ---------------------------------------------------- End Utils ---------------------------------------------------------

	void preload()
	{
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.x = float32_t(glsl::gl_WorkGroupID().x) * pushConstants.imageTwoPixelSize_x + pushConstants.imageHalfPixelSize.x;
		normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x + pushConstants.imagePixelSize.x;
		normalizedCoordsFirstLine.y = (int32_t(workgroup::SubgroupContiguousIndex()) - pushConstants.padding) * pushConstants.imagePixelSize.y + pushConstants.imageHalfPixelSize.y;

		[unroll]
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			const float32_t4 firstLineTexValue = texture.SampleLevel(samplerState, normalizedCoordsFirstLine, 0);
			
			[unroll]
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].real(scalar_t(firstLineTexValue[channel]));

			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			const float32_t4 secondLineTexValue = texture.SampleLevel(samplerState, normalizedCoordsSecondLine, 0);
			
			[unroll]
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].imag(scalar_t(secondLineTexValue[channel]));

			normalizedCoordsFirstLine.y += pushConstants.imageWorkgroupSizePixelSize_y;
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

			uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
			[unroll]
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				colMajorAccessor.set(colMajorOffset(glsl::gl_WorkGroupID().x, globalElementIndex), preloaded[channel][localElementIndex]);
				globalElementIndex += WorkgroupSize;
			}
		}
	}
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedFirstAxisAccessor preloadedAccessor;

	preloadedAccessor.preload();
	
	for (uint16_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.currentChannel = channel;
		// Wait on previous FFT pass to ensure no thread is on previous FFT trying to read from sharedmem
		if (channel)
			sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	}
	preloadedAccessor.unload();
}