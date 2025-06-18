#include "common.hlsl"
#include "render_common.hlsl"

// TODO: move the Compute Shader that generates vertices into this!
// Also do an indexed draw
struct SPushConstants
{
    uint64_t particleVerticesAddress;
};

[[vk::push_constant]] SPushConstants pc;


#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
using namespace nbl::hlsl;

[shader("vertex")]
PSInput main(uint vertexID : SV_VertexID)
{
    PSInput output;

    VertexInfo vertex = (bda::__ptr<VertexInfo>::create(pc.particleVerticesAddress)+vertexID).deref_restrict().load();

    output.position = vertex.position;
    output.vsSpherePos = vertex.vsSpherePos.xyz;

    output.radius = vertex.radius;
    output.color = vertex.color;
    output.uv = vertex.uv;

    return output;
}