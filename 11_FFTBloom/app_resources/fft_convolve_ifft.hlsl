#include "fft_mirror_common.hlsl"

[[vk::binding(3, 0)]] Texture2DArray<float32_t2> kernelChannels;
[[vk::binding(1, 0)]] SamplerState samplerState;

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * (ConstevalParameters::NumWorkgroups << 1) | y; // can sum with | here because NumWorkGroups is still PoT (has to match half the TotalSize of previous pass)
}

uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * pushConstants.imageRowLength + x; // can no longer sum with | since there's no guarantees on row length
}

// Same as what was used to store in col-major after first axis FFT. This time we launch one workgroup per row so the height of the channel's (half) image is ConstevalParameters::NumWorkgroups,
// and the width (number of columns) is passed as a push constant
uint64_t getColMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.colMajorBufferAddress + channel * ConstevalParameters::NumWorkgroups * pushConstants.imageRowLength * sizeof(complex_t<scalar_t>);
}

// Image saved has the same size as image read 
uint64_t getRowMajorChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * ConstevalParameters::NumWorkgroups * pushConstants.imageRowLength * sizeof(complex_t<scalar_t>);
}


// ------------------------------------------ SECOND AXIS FFT + CONVOLUTION + IFFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index TotalSize / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

// Reordering for unpacking on load is used with info from the previous pass FFT
using PreviousPassFFTIndexingUtils = workgroup::fft::FFTIndexingUtils<ConstevalParameters::PreviousElementsPerInvocationLog2, ConstevalParameters::PreviousWorkgroupSizeLog2>;

struct PreloadedSecondAxisAccessor : PreloadedAccessorMirrorTradeBase
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t NumWorkgroupsLog2 = ConstevalParameters::NumWorkgroupsLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t NumWorkgroups = ConstevalParameters::NumWorkgroups;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t PreviousWorkgroupSize = uint32_t(ConstevalParameters::PreviousWorkgroupSize);
	NBL_CONSTEXPR_STATIC_INLINE float32_t TotalSizeReciprocal = ConstevalParameters::TotalSizeReciprocal;
	NBL_CONSTEXPR_STATIC_INLINE float32_t2 KernelHalfPixelSize;

	NBL_CONSTEXPR_STATIC_INLINE vector<scalar_t, 2> One;

	// The lower half of the DFT is retrieved (in bit-reversed order as an N/2 bit number, N being the length of the whole DFT) as the even elements of the Nabla-ordered FFT.
	// That is, for the threads in the previous pass you take all `preloaded[0]` elements in thread-ascending order (so preloaded[0] for the 0th thread, then 1st thread etc).
	// Then you do the same for the next even index of `preloaded` (`prealoded[2]`, `preloaded[4]`, etc).
	// Every two consecutive threads here have to get the value for the row given by `gl_WorkGroupID().x`, for two consecutive columns which were stored as one column holding the packed FFT.
	// Except for the special case `gl_WorkGroupID().x = 0`, to unpack the values for the current two columns at `y = NablaFFT[gl_WorkGroupID().x] = DFT[T]` we need the value at 
	// `NablaFFT[getDFTMirrorIndex(gl_WorkGroupID().x)] = DFT[-T]`. To achieve this, we make the thread with an even ID load the former element and the one with an odd ID load the latter.
	// Then, we do a subgroup shuffle (`xor`ing with 1) so each even thread shares its value with their corresponding odd thread. Then they perform a computation to retrieve the value
	// corresponding to the column they correspond to. 
	// The `gl_WorkGroupID().x = 0` case is special because instead of getting the mirror we need to get both zero and nyquist frequencies for the columns, which doesn't happen just by mirror
	// indexing. 
	void preload(uint32_t channel)
	{
		// Set up accessor to point at channel offsets
		bothBuffersAccessor = DoubleLegacyBdaAccessor<complex_t<scalar_t> >::create(getColMajorChannelStartAddress(channel), getRowMajorChannelStartAddress(channel));

		// This one shows up a lot so we give it a name
		const bool oddThread = glsl::gl_SubgroupInvocationID() & 1u;

		if (glsl::gl_WorkGroupID().x)
		{
			for (uint32_t elementIndex = 0; elementIndex < ElementsPerInvocation; elementIndex++)
			{
				// Since every two consecutive columns are stored as one packed column, we divide the index by 2 to get the index of that packed column
				const int32_t index = int32_t(WorkgroupSize * elementIndex | workgroup::SubgroupContiguousIndex()) / 2;
				const int32_t paddedIndex = index - pushConstants.halfPadding;
				int32_t wrappedIndex = paddedIndex < 0 ? ~paddedIndex : paddedIndex; // ~x = - x - 1 in two's complement (except maybe at the borders of representable range) 
				wrappedIndex = paddedIndex < pushConstants.imageHalfRowLength ? wrappedIndex : pushConstants.imageRowLength + ~paddedIndex;
				// If mirrored, we need to invert which thread is loading lo and which is loading hi
				bool invert = paddedIndex < 0 || paddedIndex >= pushConstants.imageHalfRowLength;
				// Even thread must index a y corresponding to an even element of the previous FFT pass, and the odd thread must index its DFT Mirror
				// The math here essentially ensues we enumerate all even elements in order: we alternate `PreviousWorkgroupSize` even elements (all `preloaded[0]` elements of
				// the previous pass' threads), then `PreviousWorkgroupSize` odd elements (`preloaded[1]`) and so on
				const uint32_t evenRow = glsl::gl_WorkGroupID().x + ((glsl::gl_WorkGroupID().x / PreviousWorkgroupSize) * PreviousWorkgroupSize);
				uint32_t y = oddThread ? PreviousPassFFTIndexingUtils::getNablaMirrorIndex(evenRow) : evenRow;
				const complex_t<scalar_t> loOrHi = bothBuffersAccessor.get(colMajorOffset(wrappedIndex, y));
				// Make it a vector so it can be subgroup-shuffled
				const vector <scalar_t, 2> loOrHiVector = vector <scalar_t, 2>(loOrHi.real(), loOrHi.imag());
				const vector <scalar_t, 2> otherThreadloOrHiVector = glsl::subgroupShuffleXor< vector <scalar_t, 2> >(loOrHiVector, 1u);
				const complex_t<scalar_t> otherThreadLoOrHi = { otherThreadloOrHiVector.x, otherThreadloOrHiVector.y };
				complex_t<scalar_t> lo = ternaryOperator(oddThread, otherThreadLoOrHi, loOrHi);
				complex_t<scalar_t> hi = ternaryOperator(oddThread, loOrHi, otherThreadLoOrHi);
				fft::unpack<scalar_t>(lo, hi);
				preloaded[elementIndex] = ternaryOperator(oddThread ^ invert, hi, lo);
			}
		}
		// Special case where we retrieve 0 and Nyquist
		else
		{
			for (uint32_t elementIndex = 0; elementIndex < ElementsPerInvocation; elementIndex++)
			{
				// Since every two consecutive columns are stored as one packed column, we divide the index by 2 to get the index of that packed column
				const int32_t index = int32_t(WorkgroupSize * elementIndex | workgroup::SubgroupContiguousIndex()) / 2;
				const int32_t paddedIndex = index - pushConstants.halfPadding;
				int32_t wrappedIndex = paddedIndex < 0 ? ~paddedIndex : paddedIndex; // ~x = - x - 1 in two's complement (except maybe at the borders of representable range) 
				wrappedIndex = paddedIndex < pushConstants.imageHalfRowLength ? wrappedIndex : pushConstants.imageRowLength + ~paddedIndex;
				// If mirrored, we need to invert which thread is loading lo and which is loading hi
				bool invert = paddedIndex < 0 || paddedIndex >= pushConstants.imageHalfRowLength;
				// Even thread retrieves Zero, odd thread retrieves Nyquist. Zero is always `preloaded[0]` of the previous FFT's 0th thread, while Nyquist is always `preloaded[1]` of that same thread.
				// Therefore we know Nyquist ends up exactly at y = PreviousWorkgroupSize
				uint32_t y = oddThread ? PreviousWorkgroupSize : 0;
				const complex_t<scalar_t> loOrHi = bothBuffersAccessor.get(colMajorOffset(wrappedIndex, y));
				// Make it a vector so it can be subgroup-shuffled
				const vector <scalar_t, 2> loOrHiVector = vector <scalar_t, 2>(loOrHi.real(), loOrHi.imag());
				const vector <scalar_t, 2> otherThreadloOrHiVector = glsl::subgroupShuffleXor< vector <scalar_t, 2> >(loOrHiVector, 1u);
				const complex_t<scalar_t> otherThreadLoOrHi = { otherThreadloOrHiVector.x, otherThreadloOrHiVector.y };

				// `lo` holds `Z0 + iZ1` and `hi` holds `N0 + iN1`. We want at the end for `lo` to hold the packed `Z0 + iN0` and `hi` to hold `Z1 + iN1`
				// For the even thread (`oddThread == false`) `lo = loOrHi` and `hi = otherThreadLoOrHi`. For the odd thread the opposite is true

				// Even thread writes `lo = Z0 + iN0`
				const complex_t<scalar_t> evenThreadLo = { loOrHi.real(), otherThreadLoOrHi.real() };
				// Odd thread writes `hi = Z1 + iN1`
				const complex_t<scalar_t> oddThreadHi = { otherThreadLoOrHi.imag(), loOrHi.imag() };
				preloaded[elementIndex] = ternaryOperator(oddThread ^ invert, oddThreadHi, evenThreadLo);
			}
		}
	}

	// Each element on this row is Nabla-ordered. So the element at `x' = index, y' = gl_WorkGroupID().x` that we're operating on is actually the element at
	// `x = F(index), y = bitreverse(gl_WorkGroupID().x)` (with the bitreversal done as an N-1 bit number, for `N = log2(TotalSize)` *of the first axist FFT*)
	template<typename sharedmem_adaptor_t>
	void convolve(uint32_t channel, sharedmem_adaptor_t adaptorForSharedMemory)
	{
		if (glsl::gl_WorkGroupID().x)
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				const uint32_t globalElementIndex = WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
				const uint32_t indexDFT = FFTIndexingUtils::getDFTIndex(globalElementIndex);
				const uint32_t y = fft::bitReverse<uint32_t, NumWorkgroupsLog2>(glsl::gl_WorkGroupID().x);
				const uint32_t2 texCoords = uint32_t2(indexDFT, y);
				const float32_t2 uv = texCoords * float32_t2(TotalSizeReciprocal, 1.f / (2 * NumWorkgroups)) + KernelHalfPixelSize;
				const vector<scalar_t, 2> sampledKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const vector<scalar_t, 2> sampledKernelInterpolatedVector = lerp(sampledKernelVector, One, pushConstants.interpolatingFactor);
				const complex_t<scalar_t> sampledKernelInterpolated = { sampledKernelInterpolatedVector.x, sampledKernelInterpolatedVector.y };
				preloaded[localElementIndex] = preloaded[localElementIndex] * sampledKernelInterpolated;
			}
		}
		// Remember first row holds Z + iN
		else
		{
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
			{
				complex_t<scalar_t> zero = preloaded[localElementIndex];
				complex_t<scalar_t> nyquist = getDFTMirror<sharedmem_adaptor_t>(localElementIndex, adaptorForSharedMemory);

				fft::unpack<scalar_t>(zero, nyquist);

				// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getDFTIndex(index)` to get the actual index into the DFT
				const uint32_t globalElementIndex = WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
				const float32_t indexDFT = float32_t(FFTIndexingUtils::getDFTIndex(globalElementIndex));

				float32_t2 uv = float32_t2(indexDFT * TotalSizeReciprocal, float32_t(0)) + KernelHalfPixelSize;
				const vector<scalar_t, 2> zeroKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const vector<scalar_t, 2> zeroKernelInterpolatedVector = lerp(zeroKernelVector, One, pushConstants.interpolatingFactor);
				const complex_t<scalar_t> zeroKernelInterpolated = { zeroKernelInterpolatedVector.x, zeroKernelInterpolatedVector.y };
				zero = zero * zeroKernelInterpolated;

				// Do the same for the nyquist coord
				uv.y += 0.5;
				const vector<scalar_t, 2> nyquistKernelVector = kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(channel)), 0);
				const vector<scalar_t, 2> nyquistKernelInterpolatedVector = lerp(nyquistKernelVector, One, pushConstants.interpolatingFactor);
				const complex_t<scalar_t> nyquistKernelInterpolated = { nyquistKernelInterpolatedVector.x, nyquistKernelInterpolatedVector.y };
				nyquist = nyquist * nyquistKernelInterpolated;

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
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed IFFT of Zero and Nyquist rows.
	void unload(uint32_t channel)
	{
		const uint64_t startAddress = getRowMajorChannelStartAddress(channel);

		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			const int32_t globalElementIndex = int32_t(WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex());
			const int32_t paddedIndex = globalElementIndex - pushConstants.padding;
			if (paddedIndex >= 0 && paddedIndex < pushConstants.imageRowLength)
				bothBuffersAccessor.set(rowMajorOffset(paddedIndex, glsl::gl_WorkGroupID().x), preloaded[localElementIndex]);
		}
	}

	DoubleLegacyBdaAccessor<complex_t<scalar_t> > bothBuffersAccessor;
};

NBL_CONSTEXPR_STATIC_INLINE float32_t2 PreloadedSecondAxisAccessor::KernelHalfPixelSize = ConstevalParameters::KernelHalfPixelSize;
NBL_CONSTEXPR_STATIC_INLINE vector<scalar_t, 2> PreloadedSecondAxisAccessor::One = {1.0f, 0.f};

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