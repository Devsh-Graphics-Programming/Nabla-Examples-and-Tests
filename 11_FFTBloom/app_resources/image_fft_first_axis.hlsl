#include "fft_mirror_common.hlsl"

[[vk::binding(0, 0)]] Texture2D<float32_t4> texture;
[[vk::binding(1, 0)]] SamplerState samplerState;

// ---------------------------------------------------- Utils ---------------------------------------------------------

// After first FFT we only store half of a column, so the offset per column is half the image side length
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * FFTParameters::TotalSize | y;
}

// Each channel after first FFT will be stored as half the image (every two columns have been packed into one column of complex numbers) in col-major order.
// We launch one workgroup every two columns in the image, so there are `glsl::gl_NumWorkGroups().x` columns (of complex elements) per channel, 
// each of size `FFTParameters::TotalSize`
uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * FFTParameters::TotalSize * sizeof(complex_t<scalar_t>);
}


// -------------------------------------------- FIRST AXIS FFT ------------------------------------------------------------------

// Each Workgroup computes the FFT along two consecutive vertical scanlines (fixed x for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the x coordinates for each of the consecutive lines. This time we launch `inputImageSize.x / 2` workgroups 

struct PreloadedFirstAxisAccessor : PreloadedAccessorMirrorTradeBase
{
	void preload(uint32_t channel)
	{
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.x = float32_t(glsl::gl_WorkGroupID().x) * pushConstants.imageTwoPixelSize_x + pushConstants.imageHalfPixelSize.x;
		normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x + pushConstants.imagePixelSize.x;                                         

		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			// Index computation here is easier than FFT since the stride is fixed at WorkgroupSize
			const int32_t index = int32_t(WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex());
			normalizedCoordsFirstLine.y = float32_t(index - pushConstants.padding) * pushConstants.imagePixelSize.y + pushConstants.imageHalfPixelSize.y;
			normalizedCoordsSecondLine.y = normalizedCoordsFirstLine.y;
			preloaded[localElementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine, 0)[channel]));
			preloaded[localElementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine, 0)[channel]));
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
	void unload(uint32_t channel)
	{
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			storeColMajor(localElementIndex * WorkgroupSize | workgroup::SubgroupContiguousIndex(), preloaded[localElementIndex]);
		}
	}
	LegacyBdaAccessor<complex_t<scalar_t> > colMajorAccessor;
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;

	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}