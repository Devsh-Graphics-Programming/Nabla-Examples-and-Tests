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
RWTexture2DArray<complex_t<scalar_t> > [[vk::binding(0, 0)]] kernelChannels;

groupshared uint32_t sharedmem[workgroup::fft::SharedMemoryDWORDs<scalar_t, _NBL_HLSL_WORKGROUP_SIZE_>];

// Users MUST define this method for FFT to work
uint32_t3 glsl::gl_WorkGroupSize() { return uint32_t3(_NBL_HLSL_WORKGROUP_SIZE_, 1, 1); }

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint32_t colMajorOffset(uint32_t x, uint32_t y)
{
	return x * IMAGE_SIDE_LENGTH | y;
}

// Each channel after first FFT will be stored as half the image (cut along the x axis) in col-major order, and the whole size of the image is N^2, 
// for N = _NBL_HLSL_WORKGROUP_SIZE_ * ELEMENTS_PER_THREAD
uint32_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.outputAddress + channel * IMAGE_SIDE_LENGTH * IMAGE_SIDE_LENGTH / 2 * sizeof(complex_t<scalar_t>);
}

// Util to unpack two values from the packed FFT X + iY - get outputs in the same input arguments, storing x to lo and y to hi
void unpack(NBL_CONST_REF_ARG(complex_t<scalar_t>) lo, NBL_CONST_REF_ARG(complex_t<scalar_t>) hi)
{
	complex_t<scalar_t> x = (lo + conj(hi)) * scalar_t(0.5);
	hi = rotateRight<scalar_t>(lo - conj(hi)) * 0.5;
	lo = x;
}

// --------------------------------------------------- Normalization -----------------------------------------------------------------

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT.
	vector <scalar_t, 3> channelWiseSums;
	uint32_t channelStartAddress = pushConstants.outputAddress;

	for (uint32_t channel = 0u; channel < CHANNELS; channel++) {
		channelWiseSums[i] = vk::RawBufferLoad<scalar_t>(getChanneñStartAddress(channel));
	}
	return (colorspace::scRGBtoXYZ * channelWiseSums).y;
}

void normalize(uint32_t channel)
{
	const scalar_t power = getPower();
	const uint32_t startAddress = getChannelStartAddress(channel);

	// Remember that the first column has packed `Z + iN` so it has to unpack those. We avoid preloading + shuffling now because we'd be killing the occupancy so that a single 
	// workgroup runs a bit faster.
	if (!gl_WorkGroupID().x)
	{
		// FFT[Z + iN] was stored in the Nabla order, so we need to unpack it differently from what we did in the first axis FFT case - we're going to store it whole
		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t otherIndex = workgroup::fft::getOutputIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(-workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index));
			complex_t<scalar_t> zero = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + colMajorOffset(0, index) * sizeof(complex_t<scalar_t>));
			complex_t<scalar_t> nyquist = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + colMajorOffset(0, otherIndex) * sizeof(complex_t<scalar_t>));

			unpack(zero, nyquist);
			// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getFrequencyIndex(index)` to get the actual index into the DFT
			const uint32_t indexDFT = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);

			// Store zeroth element
			const uint32_t2 zeroCoord = uint32_t2(0, indexDFT);
			complex_t<scalar_t> shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(indexDFT));
			zero = (shift * zero) / power;
			kernelChannels[uint32_t3(zeroCoord, channel)] = zero;

			// Store nyquist element
			const uint32_t2 nyquistCoord = uint32_t2(IMAGE_SIDE_LENGTH / 2, indexDFT);
			shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(IMAGE_SIDE_LENGTH / 2 + indexDFT));
			nyquist = (shift * nyquist) / power;
			kernelChannels[uint32_t3(nyquistCoord, channel)] = nyquist;
		}
	}
	// The other columns have easier rules: They have to reflect their values along the Nyquist column 
	// If you'll remember, however, the first axis FFT stored the lower half of the DFT, bit-reversed not as a `log2(FFTSize)` bit number but in the range of the lower half 
	// (so as a `log2(FFTSize / 2) bit number)`. Which means that whenever we get an element at x' = `gl_WorkGroupID().x`, y' = `index` we must get the actual coordinates of 
	// the element in the DFT with x = `bitreverse(x')` and y = `F(y')`
	else
	{
		for (uint32_t elementIndex = 0; elementIndex < ELEMENTS_PER_THREAD; elementIndex++)
		{
			const uint32_t index = _NBL_HLSL_WORKGROUP_SIZE_ * elementIndex | workgroup::SubgroupContiguousIndex();
			// Get the element at x' = `gl_WorkGroupID().x`, y' = `index`
			complex_t<scalar_t> toStore = vk::RawBufferLoad<complex_t<scalar_t> >(startAddress + colMajorOffset(gl_WorkGroupID().x, index) * sizeof(complex_t<scalar_t>));

			const uint32_t bits = uint32_t(mpl::log2<IMAGE_SIDE_LENGTH>::value);
			// 33 because we're bitreversing a number that's one bit shorter than the full FFT
			uint32_t x = glsl::bitfieldReverse<uint32_t>(gl_WorkGroupID().x) >> (33 - bits);
			uint32_t y = workgroup::fft::getFrequencyIndex<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(index);

			// Store the element 
			const complex_t<scalar_t> shift = polar(scalar_t(1), - numbers::pi<scalar_t> * scalar_t(x + y));
			toStore = (shift * toStore) / power;
			kernelChannels[uint32_t3(x, y, channel)] = toStore;

			// Store the element at the column mirrored about the Nyquist column (so x'' = mirror(x))
			// https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Conjugation_in_time
			x = workgroup::fft::mirror<ELEMENTS_PER_THREAD, _NBL_HLSL_WORKGROUP_SIZE_>(x);
			y = (IMAGE_SIDE_LENGTH - y) & (IMAGE_SIDE_LENGTH - 1);
			kernelChannels[uint32_t3(x, y, channel)] = conj(toStore);
		}
	}
}