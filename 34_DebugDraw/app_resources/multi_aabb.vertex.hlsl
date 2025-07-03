#pragma shader_stage(vertex)

#include "simple_common.hlsl"

struct VSInput
{
    [[vk::location(0)]] float32_t3 position : POSITION;
};

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[shader("vertex")]
PSInput main(VSInput input)
{
    PSInput output;

    output.position = mul(pc.MVP, float32_t4(input.position, 1));
    output.color = float32_t4(0, 1, 0, 1);

    return output;
}