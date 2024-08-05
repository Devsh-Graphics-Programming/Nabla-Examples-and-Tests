#pragma shader_stage(vertex)

#include "common.hlsl"
#include "render_common.hlsl"

// set 1, binding 0
[[vk::binding(0, 1)]]
cbuffer CameraData
{
    SMVPParams camParams;
};

[[vk::binding(1, 1)]] StructuredBuffer<Particle> particleBuffer;

GSInput main(uint vertexID : SV_VertexID)
{
    GSInput output;

    output.particle = particleBuffer[vertexID].position;

    return output;
}