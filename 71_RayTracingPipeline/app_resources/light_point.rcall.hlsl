#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("callable")]
void main(inout RayLight cLight)
{
    float32_t3 lDir = pc.light.position - cLight.inHitPosition;
    float lightDistance = length(lDir);
    cLight.outIntensity = pc.light.intensity / (lightDistance * lightDistance);
    cLight.outLightDir = normalize(lDir);
    cLight.outLightDistance = lightDistance;
}