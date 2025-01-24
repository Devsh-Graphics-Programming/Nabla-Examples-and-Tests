#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("callable")]
void main(inout RayLight cLight)
{
    float32_t3 lDir = pc.light.position - cLight.inHitPosition;
    cLight.outLightDistance = length(lDir);
    cLight.outIntensity = pc.light.intensity / (cLight.outLightDistance * cLight.outLightDistance);
    cLight.outLightDir = normalize(lDir);
    float theta = dot(cLight.outLightDir, normalize(-pc.light.direction));
    float epsilon = pc.light.innerCutoff - pc.light.outerCutoff;
    float spotIntensity = clamp((theta - pc.light.outerCutoff) / epsilon, 0.0, 1.0);
    cLight.outIntensity *= spotIntensity;
}
