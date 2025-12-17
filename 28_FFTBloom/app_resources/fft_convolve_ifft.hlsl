#include "fft_mirror_common.hlsl"

[[vk::binding(3, 0)]] Texture2DArray<float32_t2> kernelChannels;
[[vk::binding(1, 0)]] SamplerState samplerState;

// ------------------------------------------ SECOND AXIS FFT + CONVOLUTION + IFFT -------------------------------------------------------------

// This is done for the channel specified in pushConstants.

// This time each Workgroup will compute the FFT along a horizontal line (fixed y for the whole Workgroup). We get the y coordinate for the
// row a workgroup is working on via `gl_WorkGroupID().x`. We have to keep this in mind: What's stored as the first row is actually `Z + iN`, 
// where `Z` is the actual 0th row and `N` is the Nyquist row (the one with index TotalSize / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

// Reordering for unpacking on load is used with info from the previous pass FFT
using PreviousPassFFTIndexingUtils = workgroup::fft::FFTIndexingUtils<ShaderConstevalParameters::PreviousElementsPerInvocationLog2, ShaderConstevalParameters::PreviousWorkgroupSizeLog2>;

struct PreloadedSecondAxisAccessor : PreloadedAccessorMirrorTradeBase
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t NumWorkgroupsLog2 = ShaderConstevalParameters::NumWorkgroupsLog2;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t NumWorkgroups = ShaderConstevalParameters::NumWorkgroups;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t PreviousWorkgroupSize = uint32_t(ShaderConstevalParameters::PreviousWorkgroupSize);
	NBL_CONSTEXPR_STATIC_INLINE float32_t TotalSizeReciprocal = ShaderConstevalParameters::TotalSizeReciprocal;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t KernelSideLength = ShaderConstevalParameters::KernelSideLength;
	NBL_CONSTEXPR_STATIC_INLINE float32_t2 KernelHalfPixelSize;

	// When sampling u/v coordinates along the first axis we did an FFT on, workgroup `w` samples at normalized position `SampleSlope * w + KernelHalfPixelSize`
	NBL_CONSTEXPR_STATIC_INLINE float32_t SampleSlope = float32_t(KernelSideLength) / float32_t(NumWorkgroups * uint32_t(KernelSideLength + 2));

	NBL_CONSTEXPR_STATIC_INLINE vector<scalar_t, 2> One;

	// ---------------------------------------------------- Utils ---------------------------------------------------------
	uint32_t colMajorOffset(uint32_t x, uint32_t y)
	{
		return x * (NumWorkgroups << 1) | y; // can sum with | here because NumWorkGroups is still PoT (has to match half the TotalSize of previous pass)
	}

	uint32_t rowMajorOffset(uint32_t x, uint32_t y)
	{
		return y * pushConstants.imageRowLength + x; // can no longer sum with | since there's no guarantees on row length
	}

	// No channel as parameter since workgroup runs on a single channel
	uint64_t getChannelStartOffsetBytes()
	{
		return uint64_t(glsl::gl_WorkGroupID().y) * pushConstants.channelStrideBytes;
	}
	// ---------------------------------------------------- End Utils ---------------------------------------------------------

	// Unpacking on load: Has no workgroup shuffles (which become execution barriers) which would be necessary for unpacking on store

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
	void preload()
	{
		// Set up accessor to read in data
		const uint64_t channelStartOffsetBytes = getChannelStartOffsetBytes();
		const LegacyBdaAccessor<complex_t<scalar_t> > colMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(pushConstants.colMajorBufferAddress + channelStartOffsetBytes);

		// This one shows up a lot so we give it a name
		const bool oddThread = glsl::gl_SubgroupInvocationID() & 1u;

		// Since every two consecutive columns are stored as one packed column, we divide the index by 2 to get the index of that packed column
		const uint32_t firstIndex = workgroup::SubgroupContiguousIndex() / 2;
		int32_t paddedIndex = int32_t(firstIndex) - pushConstants.halfPadding;
		const uint32_t evenRow = glsl::gl_WorkGroupID().x + ((glsl::gl_WorkGroupID().x / PreviousWorkgroupSize) * PreviousWorkgroupSize);
		const uint32_t y = glsl::gl_WorkGroupID().x ? (oddThread ? PreviousPassFFTIndexingUtils::getNablaMirrorIndex(evenRow) : evenRow)
													: (oddThread ? PreviousWorkgroupSize : 0);

		[unroll]
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			// If mirrored, we need to invert which thread is loading lo and which is loading hi
			// If using zero-padding, useful to find out if we're outside of [0,1) bounds
			bool inPadding = paddedIndex < 0 || paddedIndex >= pushConstants.imageHalfRowLength;
			int32_t wrappedIndex = paddedIndex < 0 ? ~paddedIndex : paddedIndex; // ~x = - x - 1 in two's complement (except maybe at the borders of representable range) 
			wrappedIndex = paddedIndex < pushConstants.imageHalfRowLength ? wrappedIndex : pushConstants.imageRowLength + ~paddedIndex;
			const complex_t<scalar_t> loOrHi = colMajorAccessor.get(colMajorOffset(wrappedIndex, y));
			// Make it a vector so it can be subgroup-shuffled
			const vector <scalar_t, 2> loOrHiVector = vector <scalar_t, 2>(loOrHi.real(), loOrHi.imag());
			const vector <scalar_t, 2> otherThreadloOrHiVector = glsl::subgroupShuffleXor< vector <scalar_t, 2> >(loOrHiVector, 1u);
			const complex_t<scalar_t> otherThreadLoOrHi = { otherThreadloOrHiVector.x, otherThreadloOrHiVector.y };

			if (glsl::gl_WorkGroupID().x)
			{
				complex_t<scalar_t> lo = nbl::hlsl::select(oddThread, otherThreadLoOrHi, loOrHi);
				complex_t<scalar_t> hi = nbl::hlsl::select(oddThread, loOrHi, otherThreadLoOrHi);
				fft::unpack<scalar_t>(lo, hi);

				// --------------------------------------------------- MIRROR PADDING -------------------------------------------------------------------------------------------
				#ifdef MIRROR_PADDING
				preloaded[localElementIndex] = nbl::hlsl::select(oddThread != inPadding, hi, lo);
				// ----------------------------------------------------- ZERO PADDING -------------------------------------------------------------------------------------------
				#else
				const complex_t<scalar_t> Zero = { scalar_t(0), scalar_t(0) };
				preloaded[localElementIndex] = nbl::hlsl::select(inPadding, Zero, nbl::hlsl::select(oddThread, hi, lo));
				#endif
				// ------------------------------------------------ END PADDING DIVERGENCE ----------------------------------------------------------------------------------------
			}
			else
			{
				// `lo` holds `Z0 + iZ1` and `hi` holds `N0 + iN1`. We want at the end for `lo` to hold the packed `Z0 + iN0` and `hi` to hold `Z1 + iN1`
				// For the even thread (`oddThread == false`) `lo = loOrHi` and `hi = otherThreadLoOrHi`. For the odd thread the opposite is true

				// Even thread writes `lo = Z0 + iN0`
				const complex_t<scalar_t> evenThreadLo = { loOrHi.real(), otherThreadLoOrHi.real() };
				// Odd thread writes `hi = Z1 + iN1`
				const complex_t<scalar_t> oddThreadHi = { otherThreadLoOrHi.imag(), loOrHi.imag() };
				preloaded[localElementIndex] = nbl::hlsl::select(oddThread != inPadding, oddThreadHi, evenThreadLo);
			}
			paddedIndex += WorkgroupSize / 2;
		}

	}

	// Each element on this row is Nabla-ordered. So the element at `x' = index, y' = gl_WorkGroupID().x` that we're operating on is actually the element at
	// `x = F(index), y = bitreverse(gl_WorkGroupID().x)` (with the bitreversal done as an N-1 bit number, for `N = log2(TotalSize)` *of the first axist FFT*)
	template<typename sharedmem_adaptor_t>
	void convolve(NBL_REF_ARG(sharedmem_adaptor_t) adaptorForSharedMemory)
	{
		if (glsl::gl_WorkGroupID().x)
		{
			const uint32_t y = bitReverseAs<uint32_t>(glsl::gl_WorkGroupID().x, NumWorkgroupsLog2);
			uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
			[unroll]
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
			{
				const uint32_t indexDFT = FFTIndexingUtils::getDFTIndex(globalElementIndex);
				const uint32_t2 texCoords = uint32_t2(indexDFT, y);
				const float32_t2 uv = texCoords * float32_t2(TotalSizeReciprocal, SampleSlope) + KernelHalfPixelSize;
				const vector<scalar_t, 2> sampledKernelVector = vector<scalar_t, 2>(kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(glsl::gl_WorkGroupID().y)), 0));
				const vector<scalar_t, 2> sampledKernelInterpolatedVector = lerp(sampledKernelVector, One, promote<vector<scalar_t, 2>, float32_t>(pushConstants.interpolatingFactor));
				const complex_t<scalar_t> sampledKernelInterpolated = { sampledKernelInterpolatedVector.x, sampledKernelInterpolatedVector.y };
				preloaded[localElementIndex] = preloaded[localElementIndex] * sampledKernelInterpolated;

				globalElementIndex += WorkgroupSize;
			}
		}
		// Remember first row holds Z + iN
		else
		{
			uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
			[unroll]
			for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
			{
				complex_t<scalar_t> zero = preloaded[localElementIndex];
				// Either wait on FFT pass or previous iteration's workgroup shuffle
				adaptorForSharedMemory.workgroupExecutionAndMemoryBarrier();
				complex_t<scalar_t> nyquist = getDFTMirror<sharedmem_adaptor_t>(globalElementIndex, adaptorForSharedMemory);

				fft::unpack<scalar_t>(zero, nyquist);

				// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getDFTIndex(index)` to get the actual index into the DFT
				const float32_t indexDFT = float32_t(FFTIndexingUtils::getDFTIndex(globalElementIndex));

				float32_t2 uv = float32_t2(indexDFT * TotalSizeReciprocal, float32_t(0)) + KernelHalfPixelSize;
				const vector<scalar_t, 2> zeroKernelVector = vector<scalar_t, 2>(kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(glsl::gl_WorkGroupID().y)), 0));
				const vector<scalar_t, 2> zeroKernelInterpolatedVector = lerp(zeroKernelVector, One, promote<vector<scalar_t, 2>, float32_t>(pushConstants.interpolatingFactor));
				const complex_t<scalar_t> zeroKernelInterpolated = { zeroKernelInterpolatedVector.x, zeroKernelInterpolatedVector.y };
				zero = zero * zeroKernelInterpolated;

				// Do the same for the nyquist coord
				uv.y = 1.f - KernelHalfPixelSize.y;
				const vector<scalar_t, 2> nyquistKernelVector = vector<scalar_t, 2>(kernelChannels.SampleLevel(samplerState, float32_t3(uv, float32_t(glsl::gl_WorkGroupID().y)), 0));
				const vector<scalar_t, 2> nyquistKernelInterpolatedVector = lerp(nyquistKernelVector, One, promote<vector<scalar_t, 2>, float32_t>(pushConstants.interpolatingFactor));
				const complex_t<scalar_t> nyquistKernelInterpolated = { nyquistKernelInterpolatedVector.x, nyquistKernelInterpolatedVector.y };
				nyquist = nyquist * nyquistKernelInterpolated;

				// Since their IFFT is going to be real, we can pack them back as Z + iN, do a single IFFT and recover them afterwards
				preloaded[localElementIndex] = zero + rotateLeft<scalar_t>(nyquist);

				// We have set Z + iN for an even element (lower half of the DFT). We must now set conj(Z) + i * conj(N) for an odd element (upper half of DFT)
				// The logic here is very similar to that in `getDFTMirror`: we figure out which of our odd elements corresponds to the other thread's
				// current even element (current even element is `localElementIndex` and our local odd element that's the mirror of the other thread's even element is
				// `elementToTradeLocalIdx`. Then we get conj(Z) + i * conj(N) from that thread and send our own via a shuffle.
				// Unlike `getDFTMirror` however, the logic is inverted, in the sense that we don't send `preloaded[mirrorLocalIndex]` but rather we receive a value to store there
				const complex_t<scalar_t> mirrored = conj(zero) + rotateLeft<scalar_t>(conj(nyquist));
				vector<scalar_t, 2> mirroredVector = { mirrored.real(), mirrored.imag() };
				const FFTMirrorTradeUtils::NablaMirrorLocalInfo info = FFTMirrorTradeUtils::getNablaMirrorLocalInfo(globalElementIndex);
				const uint32_t mirrorLocalIndex = info.mirrorLocalIndex;
				const uint32_t otherThreadID = info.otherThreadID;
				// Make sure the `getDFTMirror` at the top is done
				adaptorForSharedMemory.workgroupExecutionAndMemoryBarrier();
				workgroup::Shuffle<sharedmem_adaptor_t, vector<scalar_t, 2> >::__call(mirroredVector, otherThreadID, adaptorForSharedMemory);
				preloaded[mirrorLocalIndex].real(mirroredVector.x);
				preloaded[mirrorLocalIndex].imag(mirroredVector.y);

				globalElementIndex += 2 * WorkgroupSize;
			}
		}
	}

	// Save a row back in row major order. Remember that the first row (one with `gl_WorkGroupID().x == 0`) will actually hold the packed IFFT of Zero and Nyquist rows.
	void unload()
	{
		// Set up accessor to write out data
		const uint64_t channelStartOffsetBytes = getChannelStartOffsetBytes();
		const LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(pushConstants.rowMajorBufferAddress + channelStartOffsetBytes);

		const uint32_t firstIndex = workgroup::SubgroupContiguousIndex();
		int32_t paddedIndex = int32_t(firstIndex) - pushConstants.padding;
		[unroll]
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			if (paddedIndex >= 0 && paddedIndex < pushConstants.imageRowLength)
				rowMajorAccessor.set(rowMajorOffset(paddedIndex, glsl::gl_WorkGroupID().x), preloaded[localElementIndex]);

			paddedIndex += WorkgroupSize;
		}
	}
};
NBL_CONSTEXPR_STATIC_INLINE float32_t2 PreloadedSecondAxisAccessor::KernelHalfPixelSize = ShaderConstevalParameters::KernelHalfPixelSize;
NBL_CONSTEXPR_STATIC_INLINE vector<scalar_t, 2> PreloadedSecondAxisAccessor::One = {1.0f, 0.f};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
[shader("compute")]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using sharedmem_adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, FFTParameters::WorkgroupSize>;
	sharedmem_adaptor_t adaptorForSharedMemory;

	PreloadedSecondAxisAccessor preloadedAccessor;

	preloadedAccessor.preload();
	workgroup::FFT<false, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	// Update state after FFT run
	adaptorForSharedMemory.accessor = sharedmemAccessor;
	preloadedAccessor.convolve(adaptorForSharedMemory);
	// Remember to update the accessor's state
	sharedmemAccessor = adaptorForSharedMemory.accessor;
	// Either wait on first FFT (all workgroups but 0) or convolution (only 0th workgroup actually uses sharedmem for convolution)
	sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
	workgroup::FFT<true, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	preloadedAccessor.unload();
}