#include "common.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"

[[vk::binding(2, 0)]] RWTexture2DArray<float32_t2> kernelChannels;

// ---------------------------------------------------- Utils ---------------------------------------------------------
uint64_t rowMajorOffset(uint32_t x, uint32_t y)
{
	return y * FFTParameters::TotalSize | x;
}

uint64_t getChannelStartAddress(uint32_t channel)
{
	return pushConstants.rowMajorBufferAddress + channel * FFTParameters::TotalSize * FFTParameters::TotalSize / 2 * sizeof(complex_t<scalar_t>);
}

// --------------------------------------------------- Normalization -----------------------------------------------------------------

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT.
	vector <scalar_t, 3> channelWiseSums;

	for (uint32_t channel = 0u; channel < Channels; channel++) {
		channelWiseSums[channel] = vk::RawBufferLoad<scalar_t>(getChannelStartAddress(channel));
	}
	// Just need Y
	return mul(colorspace::scRGBtoXYZ._m10_m11_m12, channelWiseSums);
}

// Still launching FFTParameters::TotalSize / 2 workgroups
void normalizeChannel(uint32_t channel, scalar_t power, LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor)
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t NumWorkgroupsLog2 = ConstevalParameters::NumWorkgroupsLog2;
	
	// Most rows have just have to reflect their values along the Nyquist row.
	// If you'll remember, however, the first axis FFT stored the lower half of the DFT, bit-reversed not as a `log2(FFTSize)` bit number but in the range of the lower half 
	// (so as a `log2(FFTSize / 2) bit number)`. Which means that whenever we get an element at `x' = index`, `y' = gl_WorkGroupID().x` we must get the actual coordinates of 
	// the element in the DFT with `x = F(x')` and `y = bitreverse(y')`
	if (glsl::gl_WorkGroupID().x)
	{
		for (uint32_t localElementIndex = 0; localElementIndex < FFTParameters::ElementsPerInvocation; localElementIndex++)
		{
			const uint32_t index = FFTParameters::WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
			// Get the element at `x' = index`, `y' = gl_WorkGroupID().x`
			complex_t<scalar_t> toStore = rowMajorAccessor.get(rowMajorOffset(index, glsl::gl_WorkGroupID().x));

			uint32_t x = FFTIndexingUtils::getDFTIndex(index);
			uint32_t y = fft::bitReverse<uint32_t, NumWorkgroupsLog2>(glsl::gl_WorkGroupID().x);

			// Store the element 
			const scalar_t shift = (x + y) & 1 ? scalar_t(-1) : scalar_t(1);
			toStore = (toStore * shift) / power;
			vector<scalar_t, 2> toStoreVector = { toStore.real(), toStore.imag() };
			kernelChannels[uint32_t3(x, y, channel)] = toStoreVector;

			// Store the element at the column mirrored about the Nyquist column (so x'' = mirror(x))
			// https://en.wikipedia.org/wiki/Discrete_Fourier_transform#Conjugation_in_time
			// Guess what? The above says the row index is also the mirror about Nyquist! Neat
			x = FFTIndexingUtils::getDFTMirrorIndex(x);
			y = FFTIndexingUtils::getDFTMirrorIndex(y);
			const complex_t<scalar_t> conjugated = conj(toStore);
			toStoreVector.x = conjugated.real();
			toStoreVector.y = conjugated.imag();
			kernelChannels[uint32_t3(x, y, channel)] = toStoreVector;
		}
	}
	// Remember that the first row has packed `Z + iN` so it has to unpack those
	else
	{
		// FFT[Z + iN] was stored in the Nabla order, so we need to unpack it differently from what we did in the first axis FFT case - we're going to store it whole
		for (uint32_t localElementIndex = 0; localElementIndex < FFTParameters::ElementsPerInvocation; localElementIndex++)
		{
			const uint32_t index = FFTParameters::WorkgroupSize * localElementIndex | workgroup::SubgroupContiguousIndex();
			const uint32_t otherIndex = FFTIndexingUtils::getNablaMirrorIndex(index);
			complex_t<scalar_t> zero = rowMajorAccessor.get(rowMajorOffset(index, 0));
			complex_t<scalar_t> nyquist = rowMajorAccessor.get(rowMajorOffset(otherIndex, 0));

			fft::unpack<scalar_t>(zero, nyquist);
			// We now have zero and Nyquist frequencies at NFFT[index], so we must use `getDFTIndex(index)` to get the actual index into the DFT
			const uint32_t indexDFT = FFTIndexingUtils::getDFTIndex(index);

			// Store zeroth element
			const uint32_t2 zeroCoord = uint32_t2(indexDFT, 0);
			const scalar_t shift = indexDFT & 1 ? scalar_t(-1) : scalar_t(1);
			zero = (zero * shift) / power;
			vector<scalar_t, 2> zeroVector = { zero.real(), zero.imag() };
			kernelChannels[uint32_t3(zeroCoord, channel)] = zeroVector;

			// Store nyquist element
			const uint32_t2 nyquistCoord = uint32_t2(indexDFT, FFTParameters::TotalSize / 2);
			// FFTParameters::TotalSize / 2 is even, so indexDFT + FFTParameters::TotalSize / 2 is even iff indexDFT is even, which then means the shift factor stays the same
			nyquist = (nyquist * shift) / power;
			vector<scalar_t, 2> nyquistVector = { nyquist.real(), nyquist.imag() };
			kernelChannels[uint32_t3(nyquistCoord, channel)] = nyquistVector;
		}
	}
}

[numthreads(FFTParameters::WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	scalar_t power = getPower();
	LegacyBdaAccessor<complex_t<scalar_t> > rowMajorAccessor;

	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		rowMajorAccessor = LegacyBdaAccessor<complex_t<scalar_t> >::create(getChannelStartAddress(channel));
		normalizeChannel(channel, power, rowMajorAccessor);
	}
}