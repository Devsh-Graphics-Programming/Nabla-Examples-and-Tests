#include "common.hlsl"

[[vk::push_constant]] SPushConstants pc;

[shader("callable")]
void main(inout RayLight cLight)
{
    cLight.outLightDir = normalize(-pc.light.direction);
    cLight.outIntensity = 1.0;
    cLight.outLightDistance = 10000000;
}
