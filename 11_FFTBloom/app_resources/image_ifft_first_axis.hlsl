#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"

// TODO: There's a lot of redundant stuff in every FFT file, I'd like to move that to another file that I can sourceFmt at runtime then include in all of them (something like 
// a runtime common.hlsl)

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#define FORMAT "rgba16f"
#else
#define scalar_t float32_t
#define FORMAT "rgba32f"
#endif


#define FFT_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0, 0)]] [[vk::image_format( FORMAT )]] RWTexture2D<float32_t4> convolvedImage;

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
uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * pushConstants.dataElementCount + x; // can no longer sum with | since there's no guarantees on row length
}

// Same numbers as forward FFT
uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * glsl::gl_NumWorkGroups().x * FFT_LENGTH * sizeof(complex_t<scalar_t>);
}

struct PreloadedAccessorBase {

	void set(uint32_t idx, nbl::hlsl::complex_t<scalar_t> value)
	{
		preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_] = value;
	}

	void get(uint32_t idx, NBL_REF_ARG(nbl::hlsl::complex_t<scalar_t>) value)
	{
		value = preloaded[idx / _NBL_HLSL_WORKGROUP_SIZE_];
	}

	void memoryBarrier()
	{
		// only one workgroup is touching any memory it wishes to trade
		spirv::memoryBarrier(spv::ScopeWorkgroup, spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsUniformMemoryMask);
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};

// -------------------------------------------- FIRST AXIS IFFT ------------------------------------------------------------------
struct PreloadedFirstAxisAccessor : PreloadedAccessorBase {
	
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
		vector<Scalar, 2> toTradeVector = { toTrade.real(), toTrade.imag() };
		workgroup::Shuffle<SharedmemAdaptor, vector<Scalar, 2> >::__call(toTradeVector, otherThreadID, sharedmemAdaptor);
		toTrade.real(toTradeVector.x);
		toTrade.imag(toTradeVector.y);
		return toTrade;
	}

	// Each column of the data currently stored in the rowMajorBuffer corresponds to (half) a column of the DFT of a column of the convolved image. With this in mind, knowing that the IFFT will yield
	// a real result, we can pack two consecutive columns as Z = C1 + iC2 and by linearity of DFT we get IFFT(C1) = Re(IFFT(Z)), IFFT(C2) = Im(IFFT(Z)). This is the inverse of the packing trick
	// in the forward FFT, with a much easier expression.
	// When we wrote the columns after the forward FFT, each thread wrote its even elements to the buffer. So it stands to reason that if we load the elements from each column in the same way, 
	// we can load each thread's even elements. Their odd elements, however, are the conjugates of even elements of some other threads - which element of which thread follows the same logic we used to 
	// unpack the FFT in the forward step.
	// Since complex conjugation is not linear, we cannot simply store two columns and pass around their conjugates. We load one, trade, then load the other, trade again.
	template<typename SharedmemAdaptor>
	void preload(uint32_t channel, SharedmemAdaptor sharedmemAdaptor)
	{
		uint64_t startAddress = getChannelStartAddress(channel);
		// Load all even elements of first column
		for (uint32_t localElementIndex = 0; localElementIndex < (ELEMENTS_PER_THREAD / 2); localElementIndex ++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			preloaded[localElementIndex << 1] = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(2 * glsl::gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>));
		}
		// Get all odd elements by trading
		for (uint32_t localElementIndex = 1; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex += 2)
		{
			preloaded[localElementIndex] = conj(trade<scalar_t, SharedmemAdaptor>(localElementIndex, sharedmemAdaptor));
		}
		// Load even elements of second column, multiply them by i and add them to even positions
		// This makes even positions hold C1 + iC2
		for (uint32_t localElementIndex = 0; localElementIndex < (ELEMENTS_PER_THREAD / 2); localElementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			preloaded[localElementIndex << 1] = preloaded[localElementIndex << 1] + rotateLeft<scalar_t>(vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, index) * sizeof(complex_t<scalar_t>)));
		}
		// Finally, trade to get odd elements of second column. Note that by trading we receive an element of the form C1 + iC2 for an even position. The current odd position holds conj(C1) and we
		// want it to hold conj(C1) + i*conj(C2). So we first do conj(C1 + iC2) to yield conj(C1) - i*conj(C2). Then we subtract conj(C1) to get -i*conj(C2), negate that to get i * conj(C2), and finally
		// add conj(C1) back to have conj(C1) + i * conj(C2).
		for (uint32_t localElementIndex = 1; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex += 2)
		{
			// Thread 0's first odd element is Nyquist, which was packed alongside Zero - this means that what was said above breaks in this particular case and needs special treatment
			if (!workgroup::SubgroupContiguousIndex() && 1 == localElementIndex)
			{
				// preloaded[1] currently holds trash - this is because 0 and Nyquist are the only fixed points of T -> -T. 
				// preloaded[0] currently holds (C1(Z) - C2(N)) + i * (C1(N) + C2(Z)). This is because of how we loaded the even elements of both columns.
				// We want preloaded[0] to hold C1(Z) + i * C2(Z) and preloaded[1] to hold C1(N) + i * C2(N).
				// We can re-load C2(Z) + i * C2(N) and use it to unpack the values
				complex_t<scalar_t> c2 = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(2 * glsl::gl_WorkGroupID().x + 1, 0) * sizeof(complex_t<scalar_t>));
				complex_t<scalar_t> p1 = { preloaded[0].imag() - c2.real(), c2.imag() };
				preloaded[1] = p1;
				complex_t<scalar_t> p0 = { preloaded[0].real() + c2.imag() , c2.real()};
				preloaded[0] = p0;
			}
			else 
			{
				complex_t<scalar_t> otherThreadEven = conj(trade<scalar_t, SharedmemAdaptor>(localElementIndex, sharedmemAdaptor));
				otherThreadEven = otherThreadEven - preloaded[localElementIndex];
				otherThreadEven = otherThreadEven * scalar_t(-1);
				preloaded[localElementIndex] = preloaded[localElementIndex] + otherThreadEven;
			}
		}
	}

	void unload(uint32_t channel)
	{
		uint32_t2 imageDimensions;
		convolvedImage.GetDimensions(imageDimensions.x, imageDimensions.y);
		const uint32_t padding = uint32_t(FFT_LENGTH - imageDimensions.y) >> 1;
		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t paddedIndex = index - padding;
			if (paddedIndex >= 0 && paddedIndex < imageDimensions.y)
			{
				vector<scalar_t, 4> texValue = convolvedImage.Load(uint32_t2(2 * glsl::gl_WorkGroupID().x, paddedIndex));
				texValue[channel] = scalar_t(preloaded[localElementIndex].real());
				texValue.a = scalar_t(1);
				convolvedImage[uint32_t2(2 * glsl::gl_WorkGroupID().x, paddedIndex)] = texValue;

				texValue = convolvedImage.Load(uint32_t2(2 * glsl::gl_WorkGroupID().x + 1, paddedIndex));
				texValue[channel] = scalar_t(preloaded[localElementIndex].imag());
				texValue.a = scalar_t(1);
				convolvedImage[uint32_t2(2 * glsl::gl_WorkGroupID().x + 1, paddedIndex)] = texValue;
			}
		}
	}

};

void lastAxisFFT()
{
	SharedMemoryAccessor sharedmemAccessor;
	// Set up the memory adaptor
	using adaptor_t = accessor_adaptors::StructureOfArrays<SharedMemoryAccessor, uint32_t, uint32_t, 1, _NBL_HLSL_WORKGROUP_SIZE_>;
	adaptor_t sharedmemAdaptor;

	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		sharedmemAdaptor.accessor = sharedmemAccessor;
		preloadedAccessor.preload<adaptor_t>(channel, sharedmemAdaptor);
		// Update state after preload
		sharedmemAccessor = sharedmemAdaptor.accessor;
		workgroup::FFT<ELEMENTS_PER_THREAD, true, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload(channel);
	}
}

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	lastAxisFFT();
}
