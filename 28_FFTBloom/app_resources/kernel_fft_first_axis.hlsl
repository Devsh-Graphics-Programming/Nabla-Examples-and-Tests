#include "fft_common.hlsl"

[[vk::binding(0, 0)]] Texture2D<float32_t4> texture;

// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines
// Since the output images (one per channel) are square of size TotalSize (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : MultiChannelPreloadedAccessorBase
{
	// ---------------------------------------------------- Utils ---------------------------------------------------------

	uint32_t rowMajorOffset(uint32_t x, uint32_t y)
	{
		return y * TotalSize | x;
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
		uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
		[unroll]
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			const float32_t4 firstLineTexValue = texture[uint32_t2(globalElementIndex, 2 * glsl::gl_WorkGroupID().x)];
			[unroll]
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].real(scalar_t(firstLineTexValue[channel]));

			const float32_t4 secondLineTexValue = texture[uint32_t2(globalElementIndex, 2 * glsl::gl_WorkGroupID().x + 1)];
			[unroll]
			for (uint16_t channel = 0; channel < Channels; channel++)
				preloaded[channel][localElementIndex].imag(scalar_t(secondLineTexValue[channel]));

			globalElementIndex += WorkgroupSize;
		}
	}

	// Once the FFT is done, each thread should write its elements back. Storage is in column-major order because this avoids cache misses when writing.
	// Channels will be contiguous in buffer memory.
	void unload()
	{
		[unroll]
		for (uint16_t channel = 0; channel < Channels; channel++)
		{
			const uint64_t channelStartOffsetBytes = getChannelStartOffsetBytes(channel);
			const LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(pushConstants.rowMajorBufferAddress + channelStartOffsetBytes);

			uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
			[unroll]
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				rowMajorAccessor.set(rowMajorOffset(globalElementIndex, glsl::gl_WorkGroupID().x), preloaded[channel][localElementIndex]);
				globalElementIndex += WorkgroupSize;
			}
		}
	}
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedFirstAxisAccessor preloadedAccessor;

	preloadedAccessor.preload();
	[unroll]
	for (uint16_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.currentChannel = channel;
		if (channel)
			sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	}
	preloadedAccessor.unload();
}