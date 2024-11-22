#include "fft_mirror_common.hlsl"

[[vk::binding(3, 0)]] Texture2DArray<float32_t2> kernelChannels;
[[vk::binding(1, 0)]] SamplerState samplerState;

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * glsl::gl_NumWorkGroups().x | y; // can sum with | here because NumWorkGroups is still PoT (has to match half the TotalSize of previous pass)
}

uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * pushConstants.dataElementCount + x; // can no longer sum with | since there's no guarantees on row length
}

// Same as what was used to store in col-major after first axis FFT. This time we launch one workgroup per row so the height of the channel's (half) image is `glsl::gl_NumWorkGroups().x`,
// and the width (number of columns) is passed as a push constant
uint64_t getColMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * pushConstants.dataElementCount * sizeof(complex_t<scalar_t>);
}

// Image saved has the same size as image read 
uint64_t getRowMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * pushConstants.dataElementCount * sizeof(complex_t<scalar_t>);
}


// ------------------------------------------ SECOND AXIS FFT + CONVOLUTION + IFFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index TotalSize / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisAccessor : PreloadedAccessorMirrorTradeBase
{
	void preload(uint32_t channel)
	{
		// Set up accessor to point at channel offsets
		bothBuffersAccessor = DoubleLegacyBdaAccessor<complex_t<scalar_t> >::create(getColMajorChannelStartAddress(channel), getRowMajorChannelStartAddress(channel));
		const uint32_t padding = uint32_t(TotalSize - pushConstants.dataElementCount) >> 1;

		for (uint32_t elementIndex = 0; elementIndex < ElementsPerInvocation; elementIndex++)
		{
			const uint32_t index = WorkgroupSize * elementIndex | workgroup::SubgroupContiguousIndex();
			const int32_t paddedIndex = index - int32_t(padding);
			int32_t wrappedIndex = paddedIndex < 0 ? ~ paddedIndex : paddedIndex; // ~x = - x - 1 in two's complement (except maybe at the borders of representable range) 
			wrappedIndex = paddedIndex < pushConstants.dataElementCount ? wrappedIndex : 2 * pushConstants.dataElementCount - paddedIndex + 1;
			preloaded[elementIndex] = bothBuffersAccessor.get(colMajorOffset(wrappedIndex, glsl::gl_WorkGroupID().x));
		}
	}

	// Each element on this row is Nabla-ordered. So the element at `x' = index, y' = gl_WorkGroupID().x` that we're operating on is actually the element at
	// `x = F(index), y = bitreverse(gl_WorkGroupID().x)` (with the bitreversal done as an N-1 bit number, for `N = log2(TotalSize)` *of the first axist FFT*)
	template<typename sharedmem_adaptor_t>
	void convolve(uint32_t channel, sharedmem_adaptor_t adaptorForSharedMemory)
	{
		// Remember first row holds Z + iN
		if (!glsl::gl_WorkGroupID().x)
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
			{
				complex_t<scalar_t> zero = preloaded[localElementIndex];
				complex_t<scalar_t> nyquist = getDFTMirror<sharedmem_adaptor_t>(localElementIndex, adaptorForSharedMemory);

				fft::unpack<scalar_t>(zero, nyquist);

				// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getDFTIndex(index)` to get the actual index into the DFT
				const uint32_t globalElementIndex = WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
				const float32_t indexDFT = float32_t(FFTIndexingUtils::getDFTIndex(globalElementIndex));

				float32_t2 uv = float32_t2(indexDFT / float32_t(TotalSize), float32_t(0)) + pushConstants.kernelHalfPixelSize;
				const vector<scalar_t, 2> zeroKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const complex_t<scalar_t> zeroKernel = { zeroKernelVector.x, zeroKernelVector.y };
				zero = zero * zeroKernel;

				// Do the same for the nyquist coord
				uv.y += 0.5;
				const vector<scalar_t, 2> nyquistKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const complex_t<scalar_t> nyquistKernel = { nyquistKernelVector.x, nyquistKernelVector.y };
				nyquist = nyquist * nyquistKernel;

				// Since their IFFT is going to be real, we can pack them back as Z + iN, do a single IFFT and recover them afterwards
				preloaded[localElementIndex] = zero + rotateLeft<scalar_t>(nyquist);

				// We have set Z + iN for an even element (lower half of the DFT). We must now set conj(Z) + i * conj(N) for an odd element (upper half of DFT)
				// The logic here is basically the same as in getDFTMirror: we figure out which of our odd elements corresponds to the other thread's
				// current even element (current even element is `localElementIndex` and our local odd element that's the mirror of the other thread's even element is
				// `elementToTradeLocalIdx`. Then we get conj(Z) + i * conj(N) from that thread and send our own via a shuffle
				const complex_t<scalar_t> mirrored = conj(zero) + rotateLeft<scalar_t>(conj(nyquist));
				vector<scalar_t, 2> mirroredVector = { mirrored.real(), mirrored.imag() };
				const uint32_t otherElementIdx = FFTIndexingUtils::getNablaMirrorIndex(globalElementIndex);
				const uint32_t otherThreadID = otherElementIdx & (WorkgroupSize - 1);
				const uint32_t otherThreadGlobalElementIndex = WorkgroupSize * localElementIndex | otherThreadID;
				const uint32_t elementToTradeGlobalIdx = FFTIndexingUtils::getNablaMirrorIndex(otherThreadGlobalElementIndex);
				const uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / WorkgroupSize;
				workgroup::Shuffle<sharedmem_adaptor_t, vector<scalar_t, 2> >::__call(mirroredVector, otherThreadID, adaptorForSharedMemory);
				preloaded[elementToTradeLocalIdx].real(mirroredVector.x);
				preloaded[elementToTradeLocalIdx].imag(mirroredVector.y);
			}
		}
		else
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				const uint32_t globalElementIndex = WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
				const uint32_t indexDFT = FFTIndexingUtils::getDFTIndex(globalElementIndex);
				const uint32_t bits = pushConstants.numWorkgroupsLog2;
				const uint32_t y = glsl::bitfieldReverse<uint32_t>(glsl::gl_WorkGroupID().x) >> (32 - bits);
				const uint32_t2 texCoords = uint32_t2(indexDFT, y);
				const float32_t2 uv = texCoords / float32_t2(TotalSize, 2 * glsl::gl_NumWorkGroups().x) + pushConstants.kernelHalfPixelSize;
				const vector<scalar_t, 2> sampledKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const complex_t<scalar_t> sampledKernel = { sampledKernelVector.x, sampledKernelVector.y };
				preloaded[localElementIndex] = preloaded[localElementIndex] * sampledKernel;
			}
		}
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed IFFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		const uint64_t startAddress = getRowMajorChannelStartAddress(channel);

		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			const uint32_t globalElementIndex = WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t padding = uint32_t(TotalSize - pushConstants.dataElementCount) >> 1;
			const int32_t paddedIndex = globalElementIndex - int32_t(padding);
			if (paddedIndex >= 0 && paddedIndex < pushConstants.dataElementCount)
				bothBuffersAccessor.set(rowMajorOffset(paddedIndex, glsl::gl_WorkGroupID().x), preloaded[localElementIndex]);
		}
	}

	DoubleLegacyBdaAccessor<complex_t<scalar_t> > bothBuffersAccessor;
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using sharedmem_adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, FFTParameters::WorkgroupSize>;
	sharedmem_adaptor_t adaptorForSharedMemory;

	PreloadedSecondAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
		// Update state after FFT run
		adaptorForSharedMemory.accessor = sharedmemAccessor;
		preloadedAccessor.convolve(channel, adaptorForSharedMemory);
		// Remember to update the accessor's state
		sharedmemAccessor = adaptorForSharedMemory.accessor;
		workgroup::FFT<true, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}