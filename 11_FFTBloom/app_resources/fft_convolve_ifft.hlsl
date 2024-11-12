#include "common.hlsl"

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * KERNEL_SCALE
 * scalar_t
 * FORMAT
*/

[[vk::binding(0, 0)]] Texture2D<float32_t2> kernelChannels[CHANNELS];
[[vk::binding(0, 0)]] SamplerState samplerState[CHANNELS];

#define FFT_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

groupshared uint32_t sharedmem[workgroup::fft::SharedMemoryDWORDs<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

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

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * glsl::gl_NumWorkGroups().x | y; // can sum with | here because NumWorkGroups is still PoT (has to match half the FFT_LENGTH of previous pass)
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
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index FFT_LENGTH / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisAccessor : PreloadedAccessorBase<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>
{
	int32_t mirrorWrap(int32_t paddedCoordinate)
	{
		const int32_t negMask = paddedCoordinate >> 31u;
		const int32_t d = ((paddedCoordinate ^ negMask) / pushConstants.dataElementCount) ^ negMask;
		paddedCoordinate = paddedCoordinate - d * pushConstants.dataElementCount;
		const int32_t flip = d & 0x1;
		return (1 - flip) * paddedCoordinate + flip * (pushConstants.dataElementCount - 1 - paddedCoordinate); //lerping is a float op
	}

	void preload(uint32_t channel)
	{
		// Set up accessor to point at channel offsets
		bothBuffersAccessor = DoubleLegacyBdaAccessor<complex_t<scalar_t> >::create(getColMajorChannelStartAddress(channel), getRowMajorChannelStartAddress(channel));
		const uint32_t padding = uint32_t(FFT_LENGTH - pushConstants.dataElementCount) >> 1;

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			const int32_t paddedIndex = index - int32_t(padding);
			const int32_t wrappedIndex = mirrorWrap(paddedIndex);
			preloaded[elementIndex] = bothBuffersAccessor.get(colMajorOffset(wrappedIndex, glsl::gl_WorkGroupID().x));
		}
	}

	// Each element on this row is Nabla-ordered. So the element at `x' = index, y' = gl_WorkGroupID().x` that we're operating on is actually the element at
	// `x = F(index), y = bitreverse(gl_WorkGroupID().x)` (with the bitreversal done as an N-1 bit number, for `N = log2(FFT_LENGTH)` *of the first axist FFT*)
	template<typename SharedmemAdaptor>
	void convolve(uint32_t channel, SharedmemAdaptor sharedmemAdaptor)
	{
		// Remember first row holds Z + iN
		if (!glsl::gl_WorkGroupID().x)
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex += 2)
			{
				complex_t<scalar_t> zero = preloaded[localElementIndex];
				complex_t<scalar_t> nyquist = getDFTMirror<SharedmemAdaptor>(localElementIndex, sharedmemAdaptor);

				workgroup::fft::unpack<scalar_t>(zero, nyquist);

				// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getDFTIndex(index)` to get the actual index into the DFT
				const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
				const uint32_t indexDFT = workgroup::fft::FFTIndexingUtils<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>::getDFTIndex(globalElementIndex);

				float32_t2 uv = float32_t2(indexDFT / float32_t(FFT_LENGTH), float32_t(0)) + pushConstants.kernelHalfPixelSize;
				const vector<scalar_t, 2> zeroKernelVector = kernelChannels[channel].SampleLevel(samplerState[channel], uv, 0);
				const complex_t<scalar_t> zeroKernel = { zeroKernelVector.x, zeroKernelVector.y };
				zero = zero * zeroKernel;

				// Do the same for the nyquist coord
				uv.y += 0.5;
				const vector<scalar_t, 2> nyquistKernelVector = kernelChannels[channel].SampleLevel(samplerState[channel], uv, 0);
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
				const uint32_t otherElementIdx = workgroup::fft::FFTIndexingUtils<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>::getNablaMirrorIndex(globalElementIndex);
				const uint32_t otherThreadID = otherElementIdx & (_NBL_HLSL_WORKGROUP_SIZE_ - 1);
				const uint32_t otherThreadGlobalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | otherThreadID;
				const uint32_t elementToTradeGlobalIdx = workgroup::fft::FFTIndexingUtils<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>::getNablaMirrorIndex(otherThreadGlobalElementIndex);
				const uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / _NBL_HLSL_WORKGROUP_SIZE_;
				workgroup::Shuffle<SharedmemAdaptor, vector<scalar_t, 2> >::__call(mirroredVector, otherThreadID, sharedmemAdaptor);
				preloaded[elementToTradeLocalIdx].real(mirroredVector.x);
				preloaded[elementToTradeLocalIdx].imag(mirroredVector.y);
			}
		}
		else
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
			{
				const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
				const uint32_t indexDFT = workgroup::fft::FFTIndexingUtils<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>::getDFTIndex(globalElementIndex);
				const uint32_t bits = pushConstants.numWorkgroupsLog2;
				const uint32_t y = glsl::bitfieldReverse<uint32_t>(glsl::gl_WorkGroupID().x) >> (32 - bits);
				const uint32_t2 texCoords = uint32_t2(indexDFT, y);
				const float32_t2 uv = texCoords / float32_t2(FFT_LENGTH, 2 * glsl::gl_NumWorkGroups().x) + pushConstants.kernelHalfPixelSize;
				const vector<scalar_t, 2> sampledKernelVector = kernelChannels[channel].SampleLevel(samplerState[channel], uv, 0);
				const complex_t<scalar_t> sampledKernel = { sampledKernelVector.x, sampledKernelVector.y };
				preloaded[localElementIndex] = preloaded[localElementIndex] * sampledKernel;
			}
		}
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed IFFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		const uint64_t startAddress = getRowMajorChannelStartAddress(channel);

		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t padding = uint32_t(FFT_LENGTH - pushConstants.dataElementCount) >> 1;
			const int32_t paddedIndex = globalElementIndex - int32_t(padding);
			if (paddedIndex >= 0 && paddedIndex < pushConstants.dataElementCount)
				bothBuffersAccessor.set(rowMajorOffset(paddedIndex, glsl::gl_WorkGroupID().x), preloaded[localElementIndex]);
		}
	}

	DoubleLegacyBdaAccessor<complex_t<scalar_t> > bothBuffersAccessor;
};

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, _NBL_HLSL_WORKGROUP_SIZE_>;
	adaptor_t sharedmemAdaptor;

	PreloadedSecondAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload(channel);
		workgroup::FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		// Update state after FFT run
		sharedmemAdaptor.accessor = sharedmemAccessor;
		preloadedAccessor.convolve(channel, sharedmemAdaptor);
		// Remember to update the accessor's state
		sharedmemAccessor = sharedmemAdaptor.accessor;
		workgroup::FFT<ELEMENTS_PER_THREAD, true, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}