#pragma wave shader_stage(compute)

#include "../app_resources/common.hlsl"

static const uint32_t RED_OFFSET = 0u;
static const uint32_t GREEN_OFFSET = 256u;
static const uint32_t BLUE_OFFSET = 256u * 2u;

[[vk::combinedImageSampler]][[vk::binding(0,0)]] Texture2D texture;
[[vk::combinedImageSampler]][[vk::binding(0,0)]] SamplerState samplerState;
[[vk::binding(1,0)]] RWStructuredBuffer<uint> histogram;

[[vk::push_constant]]
PushConstants constants;

[numthreads(WorkgroupSizeX,WorkgroupSizeY,1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
	uint width;
	uint height;
	texture.GetDimensions(width, height);
	if(ID.x >= width || ID.y >= height)
		return;

    float4 texel = texture.SampleLevel(samplerState, ID.xy, 0.0);

    const uint32_t redVal = uint32_t(texel.r * 255u);
    const uint32_t greenVal = uint32_t(texel.g * 255u);
    const uint32_t blueVal = uint32_t(texel.b * 255u);

    InterlockedAdd(histogram[constants.histogramBufferOffset + RED_OFFSET + redVal], 1);
    InterlockedAdd(histogram[constants.histogramBufferOffset + GREEN_OFFSET + greenVal], 1);
    InterlockedAdd(histogram[constants.histogramBufferOffset + BLUE_OFFSET + blueVal], 1);
}