#pragma shader_stage(vertex)

#include "common.hlsl"
#include "render_common.hlsl"

[[vk::binding(1, 1)]] RWStructuredBuffer<VertexInfo> particleVertexBuffer;

PSInput main(uint vertexID : SV_VertexID)
{
    PSInput output;

    output.position = particleVertexBuffer[vertexID].position;
    output.vsSpherePos = particleVertexBuffer[vertexID].vsSpherePos.xyz;

    output.radius = particleVertexBuffer[vertexID].radius;
    output.color = particleVertexBuffer[vertexID].color;
    output.uv = particleVertexBuffer[vertexID].uv;

    return output;
}