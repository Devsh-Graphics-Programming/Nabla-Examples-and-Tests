#include "fft_common.hlsl"

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (FFTParameters::TotalSize / 2) | y;
}

uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * FFTParameters::TotalSize | x;
}

uint64_t getColMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * FFTParameters::TotalSize * FFTParameters::TotalSize / 2 * sizeof(complex_t<scalar_t>);
}

uint64_t getRowMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * FFTParameters::TotalSize * FFTParameters::TotalSize / 2 * sizeof(complex_t<scalar_t>);
}


// ------------------------------------------ SECOND AXIS FFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have launched FFTParameters::TotalSize / 2 workgroups, as there are exactly 
// that amount of rows in the buffer. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index FFTParameters::TotalSize / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisAccessor : PreloadedAccessorBase<FFTParameters>
{
	void preload(uint32_t channel)
	{
		// Set up accessor to point at channel offsets
		bothBuffersAccessor = DoubleLegacyBdaAccessor<complex_t<scalar_t> >::create(getColMajorChannelStartAddress(channel), getRowMajorChannelStartAddress(channel));

		for (uint32_t elementIndex = 0; elementIndex < ElementsPerInvocation; elementIndex++)
		{
			const uint32_t index = WorkgroupSize * elementIndex | workgroup::SubgroupContiguousIndex();
			preloaded[elementIndex] = bothBuffersAccessor.get(colMajorOffset(index, glsl::gl_WorkGroupID().x));
		}
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed FFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		for (uint32_t elementIndex = 0; elementIndex < ElementsPerInvocation; elementIndex++)
		{
			const uint32_t index = WorkgroupSize * elementIndex | workgroup::SubgroupContiguousIndex();
			bothBuffersAccessor.set(rowMajorOffset(index, glsl::gl_WorkGroupID().x), preloaded[elementIndex]);
		}
	}

	DoubleLegacyBdaAccessor<complex_t<scalar_t> > bothBuffersAccessor;
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;

	PreloadedSecondAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}