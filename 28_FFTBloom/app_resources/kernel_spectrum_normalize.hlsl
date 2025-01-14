#include "common.hlsl"
[[vk::binding(2, 0)]] RWTexture2DArray<float32_t2> kernelChannels;

[numthreads(8, 8, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	const scalar_t power = vk::RawBufferLoad<scalar_t>(pushConstants.rowMajorBufferAddress);
	const uint32_t2 texCoord = uint32_t2(glsl::gl_GlobalInvocationID().x, glsl::gl_GlobalInvocationID().y);
	const scalar_t shift = ((texCoord.x + texCoord.y) & 1) ? scalar_t(-1) : scalar_t(1);
	const scalar_t shiftOverPower = shift / power;
	// Kernel spectrum image conserves the original width but height is half + 1 (we don't need the other half since it's redundant)
	if (all(texCoord < uint32_t2(FFTParameters::TotalSize, FFTParameters::TotalSize / 2 + 1)))
	{
		for (uint32_t channel = 0; channel < Channels; channel++)
		{
			kernelChannels[uint32_t3(texCoord, channel)] *= shiftOverPower;
		}
	}
}