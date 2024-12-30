#include "common.hlsl"
#include "render_common.hlsl"

// TODO: move the Compute Shader that generates vertices into this!
// Also do an indexed draw
struct SPushConstants
{
    uint64_t particleVerticesAddress;
};

[[vk::push_constant]] SPushConstants pc;

PSInput main(uint vertexID : SV_VertexID)
{
    PSInput output;

    VertexInfo vertex = vk::RawBufferLoad<VertexInfo>(pc.particleVerticesAddress + sizeof(VertexInfo) * vertexID);

    output.position = vertex.position;
    output.vsSpherePos = vertex.vsSpherePos.xyz;

    output.radius = vertex.radius;
    output.color = vertex.color;
    output.uv = vertex.uv;

    return output;
}