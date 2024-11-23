#include "fft_mirror_common.hlsl"

[[vk::binding(0, 0)]] Texture2D<float32_t4> texture;
[[vk::binding(1, 0)]] SamplerState samplerState;

// ---------------------------------------------------- Utils ---------------------------------------------------------

// After first FFT we only store half of a column, so the offset per column is half the image side length
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (FFTParameters::TotalSize / 2) | y;
}

// We store half of a column, but unlike the kernel we only store as many columns as there actually are in the image
// We launch one workgroup every two columns in the image, and we only write (per channel) one column per column in the image
// This is unlike the kernel, in which we always upsample it to nearest PoT
// So there are `2 * glsl::gl_NumWorkGroups().x` columns per channel, each of size `FFTParameters::TotalSize / 2` (we only store half) and the 2's cancel out
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
	void storeColMajor(uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) firstValue, NBL_CONST_REF_ARG(complex_t<scalar_t>) secondValue)
	{
		colMajorAccessor.set(colMajorOffset(2 * glsl::gl_WorkGroupID().x, index), firstValue);
		colMajorAccessor.set(colMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, index), secondValue);
	}

	// Once the FFT is done, each thread should write its elements back. Storage is in column-major order because this avoids cache misses when writing.
	// Channels will be contiguous in buffer memory. We only need to store outputs 0 through Nyquist, since the rest can be recovered via complex conjugation:
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
	// FFT unpacking rules explained in the readme
	// Also, elements 0 and Nyquist fit into a single complex element since they're both real, and it's always thread 0 holding these values
	template<typename sharedmem_adaptor_t>
	void unload(uint32_t channel, sharedmem_adaptor_t adaptorForSharedMemory)
	{
		// Storing even elements of NFFT is storing the bitreversed lower half of DFT - see readme
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
		{
			complex_t<scalar_t> lo, hi;
			// First element of 0th thread has a special packing rule
			if (!localElementIndex && !workgroup::SubgroupContiguousIndex())
			{
				lo.real(preloaded[0].real());
				lo.imag(preloaded[1].real());
				hi.real(preloaded[0].imag());
				hi.imag(preloaded[1].imag());
			}
			else
			{
				lo = preloaded[localElementIndex];
				hi = getDFTMirror<sharedmem_adaptor_t>(localElementIndex, adaptorForSharedMemory);
				fft::unpack<scalar_t>(lo, hi);
			}
			// Divide localElementIdx by 2 to keep even elements packed together when writing
			storeColMajor(localElementIndex * (WorkgroupSize / 2) | workgroup::SubgroupContiguousIndex(), lo, hi);
		}

	}
	LegacyBdaAccessor<complex_t<scalar_t> > colMajorAccessor;
};

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