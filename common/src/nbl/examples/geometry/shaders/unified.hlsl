//
#include "nbl/examples/geometry/SPushConstants.hlsl"
using namespace nbl::hlsl;
using namespace nbl::hlsl::examples::geometry_creator_scene;

// for dat sweet programmable pulling
[[vk::binding(0)]] Buffer<float32_t4> utbs[/*SPushConstants::DescriptorCount*/255];

//
[[vk::push_constant]] SPushConstants pc;

//
struct SInterpolants
{
	float32_t4 position : SV_Position;
	float32_t3 meta : COLOR0;
};
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

//
[shader("vertex")]
SInterpolants BasicVS(uint32_t VertexIndex : SV_VertexID)
{
    const float32_t3 position = utbs[pc.positionView][VertexIndex].xyz;

    SInterpolants output;
    output.position = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    output.meta = mul(pc.matrices.normal,utbs[pc.normalView][VertexIndex].xyz);
    return output;
}
[shader("pixel")]
float32_t4 BasicFS(SInterpolants input) : SV_Target0
{
    return float32_t4(normalize(input.meta)*0.5f+promote<float32_t3>(0.5f),1.f);
}

// TODO: do smooth normals on the cone
[shader("vertex")]
SInterpolants ConeVS(uint32_t VertexIndex : SV_VertexID)
{
    const float32_t3 position = utbs[pc.positionView][VertexIndex].xyz;

    SInterpolants output;
    output.position = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    output.meta = mul(inverse(transpose(pc.matrices.normal)),position);
    return output;
}
[shader("pixel")]
float32_t4 ConeFS(SInterpolants input) : SV_Target0
{
    const float32_t2x3 dViewPos_dScreen = float32_t2x3(
        ddx(input.meta),
        ddy(input.meta)
    );
    const float32_t3 normal = cross(dViewPos_dScreen[0],dViewPos_dScreen[1]);
    return float32_t4(normalize(normal)*0.5f+promote<float32_t3>(0.5f),1.f);
}