#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/cpp_compat/promote.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"

using namespace nbl::hlsl;

/*
 * Remember we have these defines: 
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * (may be defined) USE_HALF_PRECISION
 * KERNEL_SCALE
*/

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] Texture2D texture;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] SamplerState samplerState;

#ifdef USE_HALF_PRECISION
#define scalar_t float16_t
#else
#define scalar_t float32_t
#endif

#define IMAGE_SIDE_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)
#define CHANNELS 3

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

// ---------------------- Utils -----------------------
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * IMAGE_SIDE_LENGTH + y;
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

	// Util to get and unpack elements from two different FFTs when having them packed as `z = x + iy`
	void unpack(uint32_t elementIndex, NBL_REF_ARG(complex_t<scalar_t>) x, NBL_REF_ARG(complex_t<scalar_t>) y)
	{
		x = (preloaded[elementIndex] + conj(preloaded[(ELEMENTS_PER_THREAD - elementIndex) & (ELEMENTS_PER_THREAD - 1)])) * scalar_t(0.5);
		y = rotateRight<scalar_t>(preloaded[elementIndex] - conj(preloaded[(ELEMENTS_PER_THREAD - elementIndex) & (ELEMENTS_PER_THREAD - 1)])) * 0.5;
	}

	complex_t<scalar_t> preloaded[ELEMENTS_PER_THREAD];
};

// -------------------------------------------- FFT Structs -----------------------------------------------------------------------------------------------------

// -------------------------------------------- FIRST AXIS FFT ---------------------------------------------

// Each Workgroup computes the FFT along two consecutive horizontal scanlines (fixed y for the whole Workgroup) so we use `2 * gl_WorkGroupID().x, 2 * gl_WorkGroupID().x + 1` 
// to get the y coordinates for each of the consecutive lines
// Since the image is square of size IMAGE_SIDE_LENGTH (defined above) we will be launching half that amount of workgroups

struct PreloadedFirstAxisAccessor : PreloadedAccessorBase {
	
	template<uint32_t Channel>
	void preload()
	{
		float32_t2 inputImageSize;
		texture.GetDimensions(inputImageSize.x, inputImageSize.y);
		float32_t2 normalizedCoordsFirstLine, normalizedCoordsSecondLine;
		normalizedCoordsFirstLine.y = (float32_t(2 * gl_WorkGroupID().x)+0.5f)/(inputImageSize*KERNEL_SCALE);
		normalizedCoordsSecondLine.y = (float32_t(2 * gl_WorkGroupID().x + 1)+0.5f)/(inputImageSize*KERNEL_SCALE);
		Promote<float32_t2, float32_t> promoter;

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
        {
        	// Index computation here is easier than FFT since the stride is fixed at _NBL_HLSL_WORKGROUP_SIZE_
            const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex + workgroup::SubgroupContiguousIndex();
			normalizedCoordsFirstLine.x = (float32_t(index)+0.5f)/(inputImageSize*KERNEL_SCALE);
			normalizedCoordsSecondLine.x = normalizedCoordsFirstLine.x;
			preloaded[elementIndex].real(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsFirstLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
			preloaded[elementIndex].imag(scalar_t(texture.SampleLevel(samplerState, normalizedCoordsSecondLine + promoter(0.5-0.5/KERNEL_SCALE), -log2(KERNEL_SCALE))[Channel]));
		}
	}

	// Util to write values to output buffer in column major order
	void storeColMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) firstValue, NBL_CONST_REF_ARG(complex_t<scalar_t>) secondValue)
	{
		vk::RawBufferStore<complex_t<scalar_t>>(startAddress + colMajorOffset(index, 2 * gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>), firstValue);
		vk::RawBufferStore<complex_t<scalar_t>>(startAddress + colMajorOffset(index, 2 * gl_WorkGroupID().x + 1) * sizeof(complex_t<scalar_t>), secondValue);
	}

	// Once the FFT is done, each thread should write its elements back. We want the storage to be in column-major order since the next FFT will be on y axis.
	// Channels will be contiguous in buffer memory. We only need to store outputs 0 through Nyquist, since the rest can be recovered via complex conjugation:
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#DFT_of_real_and_purely_imaginary_signals
	// FFT unpacking rules explained here: https://kovleventer.com/blog/fft_real/
	// Also, elements 0 and Nyquist fit into a single complex element since they're both real, and it's always thread 0 holding these values
	template<uint32_t Channel>
	void unload()
	{
		// Each channel will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
		// for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
		const uint32_t channelStride = Channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		const uint32_t channelStartAddress = pushConstants.outputAddress + channelStride;

		// Thread 0 has special unpacking and storage rules - no worries on this `if`, all but the first subgroup are coherent
		if (! workgroup::SubgroupContiguousIndex())
		{
			complex_t<scalar_t> packedFirstZeroNyquist =  {preloaded[0].real(), preloaded[ELEMENTS_PER_THREAD / 2].real()};
			complex_t<scalar_t> packedSecondZeroNyquist = {preloaded[0].imag(), preloaded[ELEMENTS_PER_THREAD / 2].imag()};
			storeColMajor(channelStartAddress, 0, packedFirstZeroNyquist, packedSecondZeroNyquist);
			
			for (uint32_t elementIndex = 1; elementIndex < ELEMENTS_PER_THREAD / 2; elementIndex++)
			{
				complex_t<scalar_t> firstLineElement, secondLineElement;
				unpack(elementIndex, firstLineElement, secondLineElement);
				storeColMajor(channelStartAddress, _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex, firstLineElement, secondLineElement);
			}
		}
		else
		{
			for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD / 2; elementIndex++)
			{
				complex_t<scalar_t> firstLineElement, secondLineElement;
				unpack(elementIndex, firstLineElement, secondLineElement);
				storeColMajor(channelStartAddress, _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex + workgroup::SubgroupContiguousIndex(), firstLineElement, secondLineElement);
			}
		}
	}
};

void firstAxisFFT()
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedFirstAxisAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload<channel>();
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		preloadedAccessor.unload<channel>();
	}
}


// ------------------------------------------ SECOND AXIS FFT -------------------------------------------------------------

// This time each Workgroup will compute the FFT along a vertical line (fixed x for the whole Workgroup). We get the x coordinate for the
// column a workgroup is working on via `gl_WorkGroupID().x`.We have launched IMAGE_SIDE_LENGTH / 2 workgroups, and there are exactly 
// that amount of columns in the buffer. We have to keep this in mind: What's stored as the first column is actually`Z + iN`, 
// where `Z` is the actual 0th column and `N` is the Nyquist column (the one with index IMAGE_SIDE_LENGTH / 2). Those are packed together
// so they need to be unpacked properly after FFT like we did earlier.

struct PreloadedSecondAxisNormalizeAccessor : PreloadedAccessorBase
{
	template<uint32_t Channel>
	void preload()
	{
		const uint32_t channelStride = Channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		const uint32_t channelStartAddress = pushConstants.outputAddress + channelStride;

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex + workgroup::SubgroupContiguousIndex();
			preloaded[elementIndex] = vk::RawBufferLoad<complex_t<scalar_t>>(channelStartAddress + colMajorOffset(gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>));
		}
	}

	// Util to write values to output buffer in column major order
	void storeColMajor(uint32_t startAddress, uint32_t index, NBL_CONST_REF_ARG(complex_t<scalar_t>) value)
	{
		vk::RawBufferStore<complex_t<scalar_t>>(startAddress + colMajorOffset(gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>), value);
	}

	// Put the image back in col-major order. Can write to same buffer after FFT. Will only save half the image again, and only get whole image when writing to
	// output images. Remember that the first column (one with `gl_WorkGroupID().x == 0`) will actually hold the FFT of Zero and Nyquist columns.
	template<uint32_t Channel>
	void unload()
	{
		const uint32_t channelStride = Channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		const uint32_t channelStartAddress = pushConstants.outputAddress + channelStride;

		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex + workgroup::SubgroupContiguousIndex();
			storeColMajor(channelStartAddress, index, preloaded[elementIndex]);
		}
	}

	// We want to avoid dumping the last channel FFT to memory since we'll have to pick it up again right afterwards to normalize. However, all threads do need to know the value
	// of the last channel's total sum to compute the luminance for normalization, so we only write this value back to memory
	void unloadLastChannelSum()
	{
		const uint32_t lastChannelStride = CHANNELS * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		const uint32_t lastChannelStartAddress = pushConstants.outputAddress + lastChannelStride;

		if (! gl_WorkGroupID().x && ! workgroup::SubgroupContiguousIndex())
		{
			// Remember that this workgroup actually contains Zeroth and Nyquist FFTs packed as `Z + iN`. But also it turns out that the sum of all elements turns out to just be
			// the real part of element `(0,0)` after the two FFTs. 
			vk::RawBufferStore<scalar_t>(lastChannelStartAddress, preloaded[0].real());
		}
	}
};

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT. So thread 0 has to broadcast this value by writing it 
	vector <scalar_t, 3> channelWiseSums;
	uint32_t channelStartAddress = pushConstants.outputAddress;
	
	for (uint32_t channel = 0u; channel < CHANNELS; channel++)
		uint32_t channelStride = channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
		channelWiseSums[i] = vk::RawBufferLoad<scalar_t>(channelStartAddress + channelStride);

	return (colorspace::scRGBtoXYZ * channelWiseSums).y;
}

void secondAxisFFTNormalize()
{
	SharedMemoryAccessor sharedmemAccessor;
	PreloadedSecondAxisNormalizeAccessor preloadedAccessor;
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		preloadedAccessor.preload<channel>();
		FFT<ELEMENTS_PER_THREAD, false, _NBL_HLSL_WORKGROUP_SIZE_, scalar_t>::template __call(preloadedAccessor, sharedmemAccessor);
		
		// Dump first two channels back to buffer, last channel can stay resident in registers, we first normalize the last channel with those so it's faster
		if (CHANNELS == channel)
		{
			unloadLastChannelSum();
		}
		else
		{
			preloadedAccessor.unload<channel>();
		}
	}

	const scalar_t power = getPower();

	// Remember that the first column has packed `Z + iN` so it has to unpack those. 
	if (! gl_WorkGroupID().x)
	{
		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex + workgroup::SubgroupContiguousIndex();

			complex_t<scalar_t> zero, nyquist; 
			unpack(elementIndex, zero, nyquist);
			const uint32_t bitReversedIndex = glsl::bitfieldReverse(index) >> BIT_REVERSE_SHIFT;
			
			// Store zeroth element
			const uint32_t2 zeroCoord = uint32_t2(0, bitReversedIndex);
			const nbl_glsl_complex shift = nbl_glsl_expImaginary(-nbl_glsl_PI*float(coord.x+coord.y));

			const uint32_t2 nyquistCoord = uint32_t2(IMAGE_SIDE_LENGTH / 2, bitReversedIndex);
		}
	}
	// The other columns have easier rules: They have to reflect their values along the Nyquist column via the conjugation in time rule,
	// see https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Conjugation_in_time
	else 
	{
	
	}
}