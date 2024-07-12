#pragma wave shader_stage(fragment)

[[vk::combinedImageSampler]][[vk::binding(0)]] Texture2D<float4> texture;
[[vk::combinedImageSampler]][[vk::binding(0)]] SamplerState samplerState;
[[vk::binding(1)]]RWTexture2D<float4> storageImgBuff;

tbuffer TextureBuffer
{
	float texBuffVal;
};

float4 main() : SV_TARGET
{
	static const float2 uv = float2(0.1f, 0.1f) * texBuffVal;
	static const float4 sampledVal = texture.SampleLevel(samplerState, uv, 0.0f);
	storageImgBuff[uint2(0, 0)] = sampledVal;
	return sampledVal;
}