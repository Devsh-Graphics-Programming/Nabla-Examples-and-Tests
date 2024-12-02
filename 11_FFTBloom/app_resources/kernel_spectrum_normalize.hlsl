#include "common.hlsl"
#include "nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl"

[[vk::binding(2, 0)]] RWTexture2DArray<float32_t2> kernelChannels;

scalar_t getPower()
{
	// Retrieve channel-wise sums of the whole image, which turn out to be Zero element of FFT.
	vector <scalar_t, 3> channelWiseSums;

	for (uint32_t channel = 0u; channel < Channels; channel++) {
		channelWiseSums[channel] = kernelChannels[uint32_t3(0, 0, channel)].x;
	}
	// Just need Y
	return mul(colorspace::scRGBtoXYZ._m10_m11_m12, channelWiseSums);
}

// Harcoded 8x8 workgroup size
[numthreads(8, 8, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{	
	const uint32_t x = glsl::gl_GlobalInvocationID().x;
	const uint32_t y = glsl::gl_GlobalInvocationID().y;

	const scalar_t power = getPower();
	const scalar_t shift = (x + y) & 1 ? scalar_t(-1) : scalar_t(1);
	const scalar_t shiftOverPower = shift / power;

	for (uint32_t channel = 0; channel < Channels; channel++)
	{
		// Since kernel is required to be square, we infer its sidelength from the FFT's total size
		if (x < FFTParameters::TotalSize && y < FFTParameters::TotalSize)
			kernelChannels[uint32_t3(x, y, channel)] *= shiftOverPower;
	}
}