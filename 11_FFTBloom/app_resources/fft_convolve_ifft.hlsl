#include "common.hlsl"

// TODO: There's a lot of redundant stuff in every FFT file, I'd like to move that to another file that I can sourceFmt at runtime then include in all of them (something like 
// a runtime common.hlsl)

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] Texture2DArray kernelChannels;
[[vk::combinedImageSampler]] [[vk::binding(0, 0)]] SamplerState samplerState;

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#else
#define scalar_t float32_t
#endif

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
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * gl_NumWorkGroups().x | y; // can sum with | here because NumWorkGroups is still PoT (has to match half the FFT_LENGTH of previous pass)
}

uint32_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * pushContants.dataElementCount + x; // can no longer sum with | since there's no guarantees on row length
}

// Same as what was used to store in col-major after first axis FFT. This time we launch one workgroup per row so the height of the channel's image is `glsl::gl_NumWorkGroups().x`,
// and the width (number of columns) is passed as a push constant
uint32_t getColMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * pushContants.dataElementCount * sizeof(complex_t<scalar_t>);
}

// Image saved has the same size as image read 
uint32_t getRowMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * pushContants.dataElementCount * sizeof(complex_t<scalar_t>);
}

struct PreloadedAccessorBase {

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
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};


// ------------------------------------------ SECOND AXIS FFT + CONVOLUTION + IFFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index FFT_LENGTH / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisAccessor : PreloadedAccessorBase
{
	int32_t mirrorWrap(int32_t paddedCoordinate)
	{
		const int32_t negMask = paddedCoordinate >> 31u;
		const int32_t d = ((paddedCoordinate ^ negMask) / pushContants.dataElementCount) ^ negMask;
		paddedCoordinate = paddedCoordinate - d * pushContants.dataElementCount;
		const int32_t flip = d & 0x1;
		return (1 - flip) * paddedCoordinate + flip * (pushContants.dataElementCount - 1 - paddedCoordinate); //lerping is a float op
	}

	void preload(uint32_t channel)
	{
		const uint32_t startAddress = getColMajorChannelStartAddress(channel);
		const uint32_t padding = (FFT_LENGTH - pushContants.dataElementCount) >> 1;

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t paddedIndex = index - padding;
			const uint32_t wrappedIndex = mirrorWrap(paddedIndex);
			preloaded[elementIndex] = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + colMajorOffset(wrappedIndex, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>));
		}
	}

	template<typename Scalar, typename SharedmemAdaptor>
	complex_t<Scalar> trade(uint32_t localElementIdx, SharedmemAdaptor sharedmemAdaptor)
	{
		uint32_t globalElementIdx = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex();
		uint32_t otherElementIdx = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(globalElementIdx);
		uint32_t otherThreadID = otherElementIdx & (_NBL_HLSL_WORKGROUP_SIZE_ - 1);
		uint32_t otherThreadGlobalElementIdx = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | otherThreadID;
		uint32_t elementToTradeGlobalIdx = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(otherThreadGlobalElementIdx);
		uint32_t elementToTradeLocalIdx = elementToTradeGlobalIdx / _NBL_HLSL_WORKGROUP_SIZE_;
		complex_t<Scalar> toTrade = preloaded[elementToTradeLocalIdx];
		workgroup::Shuffle<SharedmemAdaptor, complex_t<Scalar> >::__call(toTrade, otherThreadID, sharedmemAdaptor);
		return toTrade;
	}

	// Each element on this row is Nabla-ordered. So the element at `x' = index, y' = gl_WorkGroupID().x` that we're operating on is actually the element at
	// `x = F(index), y = bitreverse(gl_WorkGroupID().x)` (with the bitreversal done as an N-1 bit number, for `N = log2(FFT_LENGTH)` *of the first axist FFT*)
	template<typename SharedmemAdaptor>
	void convolve(uint32_t channel, SharedmemAdaptor sharedmemAdaptor)
	{
		// Remember first row holds Z + iN
		if (!gl_WorkGroupID().x)
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
			{
				const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex();
				complex_t<scalar_t> zero = preloaded[localElementIndex];
				complex_t<scalar_t> nyquist = trade<scalar_t, SharedmemAdaptor>(localElementIndex, sharedmemAdaptor);

				workgroup::fft::unpack<scalar_t>(zero, nyquist);
				// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getFrequencyIndex(index)` to get the actual index into the DFT
				const uint32_t indexDFT = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(globalElementIndex);

				// We want IFFT of these to be strictly real so the packing stays consistent
				// To do so, we're going to load the values at the correponding row exactly, and lerp them ourselves to ensure interpolation happens across one dimension only.
				// That way, we make sure to interpolate only along a row corresponding to the DFT of a real signal, which makes its IFFT stay real
				// MAYBE this isn't necessary
				const float32_t zeroNyquistU = indexDFT / float32_t(FFT_LENGTH) + pushConstants.kernelHalfPixelSize.x;
				uint32_t3 kernelSize;
				kernelChannels.GetDimensions(kernelSize.x, kernelSize.y, kernelSize.z);
				const uint32_t left = floor(kernelSize.x * zeroNyquistU);
				const uint32_t right = left + 1;

				const complex_t<scalar_t> zeroKernelLeft = kernelChannels.Load(uint32_t4(left, 0, channel, 0));
				const complex_t<scalar_t> zeroKernelRight = kernelChannels.Load(uint32_t4(right, 0, channel, 0));
				// Very annoying that I can't directly lerp complex_t, or even bit_cast
				const float32_t2 zeroKernelLeftVector = { zeroKernelLeft.real(), zeroKernelLeft.imag() };
				const float32_t2 zeroKernelRightVector = { zeroKernelRight.real(), zeroKernelRight.imag() };
				const float32_t2 zeroKernelVector = lerp(zeroKernelLeftVector, zeroKernelRightVector, frac(kernelSize.x * zeroNyquistU));
				const complex_t<float32_t> zeroKernel = { zeroKernelVector.x, zeroKernelVector.y };
				zero = zero * zeroKernel;

				// Do the same for the nyquist coord
				const complex_t<scalar_t> nyquistKernelLeft = kernelChannels.Load(uint32_t4(left, glsl::gl_NumWorkGroups().x, channel, 0));
				const complex_t<scalar_t> nyquistKernelRight = kernelChannels.Load(uint32_t4(right, glsl::gl_NumWorkGroups().x, channel, 0));
				// Same annoyance
				const float32_t2 nyquistKernelLeftVector = { nyquistKernelLeft.real(), nyquistKernelLeft.imag() };
				const float32_t2 nyquistKernelRightVector = { nyquistKernelRight.real(), nyquistKernelRight.imag() };
				const float32_t2 nyquistKernelVector = lerp(nyquistKernelLeftVector, nyquistKernelRightVector, frac(kernelSize.x * zeroNyquistU));
				const complex_t<float32_t> nyquistKernel = { nyquistKernelVector.x, nyquistKernelVector.y };
				nyquist = nyquist * nyquistKernel;

				// Since their IFFT is going to be real, we can pack them back as Z + iN, do a single IFFT and recover them afterwards
				preloaded[localElementIndex] = zero + rotateLeft<scalar_t>(nyquist);
			}
		}
		else
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
			{
				const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIdx | workgroup::SubgroupContiguousIndex();
				const uint32_t indexDFT = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(globalElementIndex);
				const uint32_t bits = uint32_t(mpl::log2<glsl::gl_NumWorkGroups().x>::value);
				const uint32_t y = glsl::bitfieldReverse<uint32_t>(gl_WorkGroupID().x) >> (32 - bits);
				const uint32_t texCoords = uint32_t(indexDFT, y);
				const float32_t2 uv = texCoords / float32_t2(FFT_LENGTH, 2 * glsl::gl_NumWorkGroups().x) + pushConstants.kernelHalfPixelSize;
				preloaded[localElementIndex] = preloaded[localElementIndex] * kernelChannels.Sample(samplerState, float32_t3(uv, channel));
			}
		}
	}

	// Util to write values to output buffer in row major order
	void storeRowMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		vk::RawBufferStore<complex_t<scalar_t> >(startAddress + rowMajorOffset(index, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>), value);
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed IFFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		const uint32_t startAddress = getRowMajorChannelStartAddress(channel);

		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t globalElementIndex = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t padding = (FFT_LENGTH - pushContants.dataElementCount) >> 1;
			const uint32_t paddedIndex = index - padding;
			if (paddedIndex >= 0 && paddedIndex < pushContants.dataElementCount)
				storeRowMajor(startAddress, paddedIndex, preloaded[localElementIndex]);
		}
	}
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
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		// Update state after FFT run
		sharedmemAdaptor.accessor = sharedmemAccessor;
		preloadedAccessor.convolve(channel, sharedmemAdaptor);
		// Remember to update the accessor's state
		sharedmemAccessor = sharedmemAdaptor.accessor;
		FFT<ELEMENTS_PER_THREAD, true, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}