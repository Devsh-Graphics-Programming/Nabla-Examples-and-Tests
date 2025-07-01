#pragma shader_stage(vertex)

#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "simple_common.hlsl"

using namespace nbl::hlsl;

[[vk::push_constant]] SPushConstants pc;

[shader("vertex")]
PSInput main(uint vertexID : SV_VertexID)
{
    PSInput output;

    float32_t4 vertex = (bda::__ptr<float32_t4>::create(pc.pVertices) + vertexID).deref_restrict().load();

    output.position = mul(pc.MVP, vertex);
    output.color = float32_t4(1, 0, 0, 1);

    return output;
}