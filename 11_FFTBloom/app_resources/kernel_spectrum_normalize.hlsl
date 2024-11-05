#include "common.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"

/*
 * Remember we have these defines:
 * _NBL_HLSL_WORKGROUP_SIZE_
 * ELEMENTS_PER_THREAD
 * KERNEL_SCALE
 * scalar_t
 * FORMAT
*/

#define IMAGE_SIDE_LENGTH (_NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD)

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1); }

[[vk::binding(0, 0)]] [[vk::image_format( FORMAT )]] RWTexture2D<float32_t2> kernelChannels[CHANNELS];

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * IMAGE_SIDE_LENGTH | x;
}

uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

// --------------------------------------------------- Normalization -----------------------------------------------------------------

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT.
	vector <scalar_t, 3> channelWiseSums;

	for (uint32_t channel = 0u; channel < CHANNELS; channel++) {
		channelWiseSums[channel] = vk::RawBufferLoad<scalar_t>(getChannelStartAddress(channel));
	}
	return (mul(colorspace::scRGBtoXYZ, channelWiseSums)).y;
}

// Still launching IMAGE_SIDE_LENGTH / 2 workgroups
void normalizeChannel(uint32_t channel, scalar_t power, LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor)
{
	// Remember that the first row has packed `Z + iN` so it has to unpack those
	if (!glsl::gl_WorkGroupID().x)
	{
		// FFT[Z + iN] was stored in the Nabla order, so we need to unpack it differently from what we did in the first axis FFT case - we're going to store it whole
		for (uint32_t localElementIndex = 0; localElementIndex < ELEMENTS_PER_THREAD; localElementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t otherIndex = workgroup::fft::getNegativeIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);
			complex_t<scalar_t> zero = rowMajorAccessor.get(rowMajorOffset(index, 0));
			complex_t<scalar_t> nyquist = rowMajorAccessor.get(rowMajorOffset(otherIndex, 0));

			workgroup::fft::unpack<scalar_t>(zero, nyquist);
			// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getFrequencyIndex(index)` to get the actual index into the DFT
			const uint32_t indexDFT = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);

			// Store zeroth element
			const uint32_t2 zeroCoord = uint32_t2(indexDFT, 0);
			const scalar_t shift = indexDFT & 1 ? scalar_t(-1) : scalar_t(1);
			zero = (zero * shift) / power;
			vector<scalar_t, 2> zeroVector = { zero.real(), zero.imag() };
			kernelChannels[channel][zeroCoord] = zeroVector;

			// Store nyquist element
			const uint32_t2 nyquistCoord = uint32_t2(indexDFT, IMAGE_SIDE_LENGTH / 2);
			// IMAGE_SIDE_LENGTH / 2 is even, so indexDFT + IMAGE_SIDE_LENGTH / 2 is even iff indexDFT is even, which then means the shift factor stays the same
			nyquist = (nyquist * shift) / power;
			vector<scalar_t, 2> nyquistVector = { nyquist.real(), nyquist.imag() };
			kernelChannels[channel][nyquistCoord] = nyquistVector;
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
			complex_t<scalar_t> toStore = rowMajorAccessor.get(rowMajorOffset(index, glsl::gl_WorkGroupID().x));

			// Number of bits needed to represent the range of half the DFT
			NBL_CONSTEXPR uint32_t bits = uint32_t(mpl::log2<IMAGE_SIDE_LENGTH>::value - 1);
			uint32_t x = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);
			uint32_t y = glsl::bitfieldReverse<uint32_t>(glsl::gl_WorkGroupID().x) >> (32 - bits);

			// Store the element 
			const scalar_t shift = (x + y) & 1 ? scalar_t(-1) : scalar_t(1);
			toStore = (toStore * shift) / power;
			vector<scalar_t, 2> toStoreVector = { toStore.real(), toStore.imag() };
			kernelChannels[channel][uint32_t2(x, y)] = toStoreVector;

			// Store the element at the column mirrored about the Nyquist column (so x'' = mirror(x))
			// https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Conjugation_in_time
			// Guess what? The above says the row index is also the mirror about Nyquist! Neat
			x = workgroup::fft::mirror<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(x);
			y = workgroup::fft::mirror<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(y);
			const complex_t<scalar_t> conjugated = conj(toStore);
			toStoreVector.x = conjugated.real();
			toStoreVector.y = conjugated.imag();
			kernelChannels[channel][uint32_t2(x, y)] = toStoreVector;
		}
	}
}

[numthreads(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	scalar_t power = getPower();
	LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor;

	for (uint32_t channel = 0; channel < CHANNELS; channel++)
	{
		rowMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(getChannelStartAddress(channel));
		normalizeChannel(channel, power, rowMajorAccessor);
	}
}