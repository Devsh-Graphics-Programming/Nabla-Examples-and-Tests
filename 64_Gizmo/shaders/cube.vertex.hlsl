#include "cube.common.hlsl"

[[vk::push_constant]] struct PushConstants pc;

// set 1, binding 0
[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SBasicViewParameters params;
};

PSInput VSMain(VSInput input)
{
    PSInput output;

    output.position = mul(params.MVP, float4(input.position, 1.0));
    output.color = float4(input.normal * 0.5 + 0.5, 1.0);

    return output;
}