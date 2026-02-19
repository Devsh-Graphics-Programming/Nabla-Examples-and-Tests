#pragma wave shader_stage(fragment)

// vertex shader is provided by the fullScreenTriangle extension
#include <nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl>
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::combinedImageSampler]][[vk::binding(0,2)]] Texture2D texture;
[[vk::combinedImageSampler]][[vk::binding(0,2)]] SamplerState samplerState;

[[vk::location(0)]] float32_t4 main(SVertexAttributes vxAttr) : SV_Target0
{
    return texture.Sample(samplerState,float32_t2(vxAttr.uv));
}