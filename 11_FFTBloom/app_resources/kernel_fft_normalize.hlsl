#include "common.hlsl"
#include "nbl/builtin/hlsl/workgroup/fft.hlsl"
#include "nbl/builtin/hlsl/workgroup/shuffle.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"
#include "nbl/builtin/hlsl/numbers.hlsl"

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
#else
#define scalar_t float32_t
#endif

#define IMAGE_SIDE_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

[[vk::push_constant]] PushConstantData pushConstants;
[[vk::binding(0, 0)]] RWTexture2DArray<complex_t<scalar_t> > kernelChannels;

groupshared uint32_t sharedmem[workgroup::fft::SharedMemoryDWORDs<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint32_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * IMAGE_SIDE_LENGTH | x;
}

uint32_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

// --------------------------------------------------- Normalization -----------------------------------------------------------------

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT.
	vector <scalar_t, 3> channelWiseSums;

	for (uint32_t channel = 0u; channel < CHANNELS; channel++) {
		channelWiseSums[i] = vk::RawBufferLoad<scalar_t>(getChannelStartAddress(channel));
	}
	return (colorspace::scRGBtoXYZ * channelWiseSums).y;
}

// Still launching IMAGE_SIDE_LENGTH / 2 workgroups
void normalize(uint32_t channel)
{
	const scalar_t power = getPower();
	const uint32_t startAddress = getChannelStartAddress(channel);

	// Remember that the first row has packed `Z + iN` so it has to unpack those
	if (!gl_WorkGroupID().x)
	{
		// FFT[Z + iN] was stored in the Nabla order, so we need to unpack it differently from what we did in the first axis FFT case - we're going to store it whole
		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t otherIndex = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);
			complex_t<scalar_t> zero = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(index, 0) * sizeof(complex_t<scalar_t>));
			complex_t<scalar_t> nyquist = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(otherIndex, 0) * sizeof(complex_t<scalar_t>));

			unpack<scalar_t>(zero, nyquist);
			// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getFrequencyIndex(index)` to get the actual index into the DFT
			const uint32_t indexDFT = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);

			// Store zeroth element
			const uint32_t2 zeroCoord = uint32_t2(indexDFT, 0);
			complex_t<scalar_t> shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(indexDFT));
			zero = (shift * zero) / power;
			kernelChannels[uint32_t3(zeroCoord, channel)] = zero;

			// Store nyquist element
			const uint32_t2 nyquistCoord = uint32_t2(indexDFT, IMAGE_SIDE_LENGTH / 2);
			shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(IMAGE_SIDE_LENGTH / 2 + indexDFT));
			nyquist = (shift * nyquist) / power;
			kernelChannels[uint32_t3(nyquistCoord, channel)] = nyquist;
		}
	}
	// The other rows have easier rules: They have to reflect their values along the Nyquist row
	// If you'll remember, however, the first axis FFT stored the lower half of the DFT, bit-reversed not as a `log2(FFTSize)` bit number but in the range of the lower half 
	// (so as a `log2(FFTSize / 2) bit number)`. Which means that whenever we get an element at `x' = index`, `y' = gl_WorkGroupID().x` we must get the actual coordinates of 
	// the element in the DFT with `x = F(x')` and `y = bitreverse(y')`
	else
	{
		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			// Get the element at `x' = index`, `y' = gl_WorkGroupID().x`
			complex_t<scalar_t> toStore = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + rowMajorOffset(index, gl_WorkGroupID().x) * sizeof(complex_t<scalar_t>));

			// Number of bits needed to represent the range of half the DFT
			NBL_CONSTEXPR uint32_t bits = uint32_t(mpl::log2<IMAGE_SIDE_LENGTH>::value - 1);
			uint32_t x = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);
			uint32_t y = glsl::bitfieldReverse<uint32_t>(gl_WorkGroupID().x) >> (32 - bits);

			// Store the element 
			const complex_t<scalar_t> shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(x + y));
			toStore = (shift * toStore) / power;
			kernelChannels[uint32_t3(x, y, channel)] = toStore;

			// Store the element at the column mirrored about the Nyquist column (so x'' = mirror(x))
			// https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Conjugation_in_time
			// Guess what? The above says the row index is also the mirror about Nyquist! Neat
			x = workgroup::fft::mirror<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(x);
			y = workgroup::fft::mirror<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(y);
			kernelChannels[uint32_t3(x, y, channel)] = conj(toStore);
		}
	}
}

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	for (uint32_t channel = 0; channel < CHANNELS; channel++)
		normalize(channel);
}