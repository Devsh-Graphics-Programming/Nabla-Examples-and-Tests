#include "fft_mirror_common.hlsl"

[[vk::binding(2, 0)]] RWTexture2D<float32_t4> convolvedImage;



// -------------------------------------------- FIRST AXIS IFFT ------------------------------------------------------------------

// Previous shader stored results unpacked, and we re-pack them before IFFT here. Although re-packing on store achieves a much higher L2 cache hit ratio and depending on the implementation it can be done
// with no barriers (and reads that all over the place) OR half the amount of barriers done here with the same accesses to memory, the SM throughput inexplicably falls by about 5%.
// So re-pack on load it is.

struct PreloadedFirstAxisAccessor : MultiChannelPreloadedAccessorMirrorTradeBase
{
	// ---------------------------------------------------- Utils ---------------------------------------------------------
	uint32_t rowMajorOffset(uint32_t x, uint32_t y)
	{
		return y * pushConstants.imageRowLength + x; // can no longer sum with | since there's no guarantees on row length
	}

	// Same numbers as forward FFT
	uint64_t getChannelStartOffsetBytes(uint16_t channel)
	{
		return uint64_t(channel) * glsl::gl_NumWorkGroups().x * TotalSize * sizeof(complex_t<scalar_t>);
	}

	// ---------------------------------------------------- End Utils ---------------------------------------------------------

	// Each column of the data currently stored in the rowMajorBuffer corresponds to (half) a column of the DFT of a column of the convolved image. With this in mind, knowing that the IFFT will yield
	// a real result, we can pack two consecutive columns as Z = C1 + iC2 and by linearity of DFT we get IFFT(C1) = Re(IFFT(Z)), IFFT(C2) = Im(IFFT(Z)). This is the inverse of the packing trick
	// in the forward FFT, with a much easier expression.
	// When we wrote the columns after the forward FFT, each thread wrote its even elements to the buffer. So it stands to reason that if we load the elements from each column in the same way, 
	// we can load each thread's even elements. Their odd elements, however, are the conjugates of even elements of some other threads - which element of which thread follows the same logic we used to 
	// unpack the FFT in the forward step.
	// Since complex conjugation is not linear, we cannot simply store two columns and pass around their conjugates. We load one, trade, then load the other, trade again.
	template<typename sharedmem_adaptor_t>
	void preload(NBL_REF_ARG(sharedmem_adaptor_t) adaptorForSharedMemory)
	{
		for (uint16_t channel = 0; channel < Channels; channel++)
		{
			const uint64_t channelStartOffsetBytes = getChannelStartOffsetBytes(channel);
			// Set LegacyBdaAccessor for reading
			const LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(pushConstants.rowMajorBufferAddress + channelStartOffsetBytes);

			uint32_t globalElementIndex = workgroup::SubgroupContiguousIndex();
			// Load all even elements of first column
			for (uint32_t localElementIndex = 0; localElementIndex < (ElementsPerInvocation / 2); localElementIndex++)
			{
				preloaded[channel][localElementIndex << 1] = rowMajorAccessor.get(rowMajorOffset(2 * glsl::gl_WorkGroupID().x, globalElementIndex));
				globalElementIndex += WorkgroupSize;
			}
			// Get all odd elements by trading
			// Reset globalElementIndex - Add WorkgroupSize to account for `localElementIndex` starting at 1
			globalElementIndex = WorkgroupSize | workgroup::SubgroupContiguousIndex();
			for (uint32_t localElementIndex = 1; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
			{
				preloaded[channel][localElementIndex] = conj(getDFTMirror<sharedmem_adaptor_t>(globalElementIndex, channel, adaptorForSharedMemory));
				// Add 2 * WorkgroupSize since `localElementIndex` moves in strides of 2
				globalElementIndex += 2 * WorkgroupSize;
				adaptorForSharedMemory.workgroupExecutionAndMemoryBarrier();
			}
			// Load even elements of second column, multiply them by i and add them to even positions
			// This makes even positions hold C1 + iC2
			// Reset globalElementIndex
			globalElementIndex = workgroup::SubgroupContiguousIndex();
			for (uint32_t localElementIndex = 0; localElementIndex < (ElementsPerInvocation / 2); localElementIndex++)
			{
				preloaded[channel][localElementIndex << 1] = preloaded[channel][localElementIndex << 1] + rotateLeft<scalar_t>(rowMajorAccessor.get(rowMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, globalElementIndex)));
				globalElementIndex += WorkgroupSize;
			}
			// Finally, trade to get odd elements of second column. Note that by trading we receive an element of the form C1 + iC2 for an even position. The current odd position holds conj(C1) and we
			// want it to hold conj(C1) + i*conj(C2). So we first do conj(C1 + iC2) to yield conj(C1) - i*conj(C2). Then we subtract conj(C1) to get -i*conj(C2), negate that to get i * conj(C2), and finally
			// add conj(C1) back to have conj(C1) + i * conj(C2).
			// Reset globalElementIndex - Add WorkgroupSize to account for `localElementIndex` starting at 1
			globalElementIndex = WorkgroupSize | workgroup::SubgroupContiguousIndex();
			for (uint32_t localElementIndex = 1; localElementIndex < ElementsPerInvocation; localElementIndex += 2)
			{
				complex_t<scalar_t> otherThreadEven = conj(getDFTMirror<sharedmem_adaptor_t>(globalElementIndex, channel, adaptorForSharedMemory));
				if (workgroup::SubgroupContiguousIndex() || localElementIndex != 1)
				{
					otherThreadEven = otherThreadEven - preloaded[channel][localElementIndex];
					otherThreadEven = otherThreadEven * scalar_t(-1);
					preloaded[channel][localElementIndex] = preloaded[channel][localElementIndex] + otherThreadEven;
				}
				// Thread 0's first odd element is Nyquist, which was packed alongside Zero - this means that what was said above breaks in this particular case and needs special treatment
				else
				{
					// preloaded[channel][1] currently holds trash - this is because 0 and Nyquist are the only fixed points of T -> -T. 
					// preloaded[channel][0] currently holds (C1(Z) - C2(N)) + i * (C1(N) + C2(Z)). This is because of how we loaded the even elements of both columns.
					// We want preloaded[channel][0] to hold C1(Z) + i * C2(Z) and preloaded[channel][1] to hold C1(N) + i * C2(N).
					// We can re-load C2(Z) + i * C2(N) and use it to unpack the values
					complex_t<scalar_t> c2 = rowMajorAccessor.get(rowMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, 0));
					complex_t<scalar_t> p1 = { preloaded[channel][0].imag() - c2.real(), c2.imag() };
					preloaded[channel][1] = p1;
					complex_t<scalar_t> p0 = { preloaded[channel][0].real() + c2.imag() , c2.real() };
					preloaded[channel][0] = p0;
				}
				// Add 2 * WorkgroupSize since `localElementIndex` moves in strides of 2
				globalElementIndex += 2 * WorkgroupSize;
				adaptorForSharedMemory.workgroupExecutionAndMemoryBarrier();
			}
		}
	}

	void unload()
	{
		const uint32_t firstIndex = workgroup::SubgroupContiguousIndex();
		int32_t paddedIndex = int32_t(firstIndex) - pushConstants.padding;
		for (uint32_t localElementIndex = 0; localElementIndex < ElementsPerInvocation; localElementIndex++)
		{
			if (paddedIndex >= 0 && paddedIndex < pushConstants.imageColumnLength)
			{
				vector<scalar_t, 4> firstLineTexValue, secondLineTexValue;
				// In case we're not using alpha
				firstLineTexValue.a = 1.f; 
				secondLineTexValue.a = 1.f;

				for (uint16_t channel = 0; channel < Channels; channel++)
				{
					firstLineTexValue[channel] = scalar_t(preloaded[channel][localElementIndex].real());
					secondLineTexValue[channel] = scalar_t(preloaded[channel][localElementIndex].imag());
				}
				convolvedImage[uint32_t2(2 * glsl::gl_WorkGroupID().x, paddedIndex)] = firstLineTexValue;
				convolvedImage[uint32_t2(2 * glsl::gl_WorkGroupID().x + 1, paddedIndex)] = secondLineTexValue;
			}
			paddedIndex += WorkgroupSize;
		}
	}
};

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using sharedmem_adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, FFTParameters::WorkgroupSize>;
	sharedmem_adaptor_t adaptorForSharedMemory;
	adaptorForSharedMemory.accessor = sharedmemAccessor;

	PreloadedFirstAxisAccessor preloadedAccessor;
	preloadedAccessor.preload(adaptorForSharedMemory);
	// Update state after preload
	sharedmemAccessor = adaptorForSharedMemory.accessor;

	for (uint16_t channel = 0; channel < Channels; channel++)
	{
		preloadedAccessor.currentChannel = channel;
		if (channel)
			sharedmemAccessor.workgroupExecutionAndMemoryBarrier();
		workgroup::FFT<true, FFTParameters>::template __call(preloadedAccessor, sharedmemAccessor);
	}
	preloadedAccessor.unload();
}
