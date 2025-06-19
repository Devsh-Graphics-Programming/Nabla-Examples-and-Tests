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
	float32_t3 color : COLOR0;
};
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

//
SInterpolants BasicVS()
{
    const float32_t3 position = utbs[pc.positionView].xyz;

    SInterpolants output;
    output.position = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    output.color = mul(pc.matrices.normalMat,utbs[pc.normalView].xyz)*0.5+promote<float32_t3>(0.5f);
    return output;
}
float32_t4 BasicFS(SInterpolants input) : SV_Target0
{
    return float32_t4(input.color,1.f);
}

// TODO: do smooth normals on the cone
SInterpolants ConeVS()
{
    const float32_t3 position = utbs[pc.positionView].xyz;

    SInterpolants output;
    output.position = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    output.color = mul(inverse(transpose(pc.matrices.normalMat)),position);
    return output;
}
float32_t4 ConeFS(SInterpolants input) : SV_Target0
{
    const float32_t2x3 dViewPos_dScreen = float32_t2x3(
        ddx(input.color),
        ddy(input.color)
    );
    return float32_t4(normalize(cross(X,Y))*0.5f+promote<float32_t3>(0.5f),1.f);
}