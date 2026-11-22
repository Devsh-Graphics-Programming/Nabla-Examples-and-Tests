//
#include "nbl/examples/geometry/SPushConstants.hlsl"
using namespace nbl::hlsl;
using namespace nbl::hlsl::examples::geometry_creator_scene;

// for dat sweet programmable pulling
[[vk::binding(0)]] Buffer<float32_t4> utbs[SPushConstants::DescriptorCount];

//
[[vk::push_constant]] SPushConstants pc;

//
struct SInterpolants
{
	float32_t4 ndc : SV_Position;
	float32_t3 meta : COLOR1;
	float32_t2 gridUV : COLOR2;
};
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

float32_t3 reconstructGeometricNormal(float32_t3 pos)
{
    const float32_t2x3 dPos_dScreen = float32_t2x3(
        ddx(pos),
        ddy(pos)
    );
    return cross(dPos_dScreen[0],dPos_dScreen[1]);
}

//
[shader("vertex")]
SInterpolants BasicVS(uint32_t VertexIndex : SV_VertexID)
{
    const float32_t3 position = utbs[pc.positionView][VertexIndex].xyz;

    SInterpolants output;
    output.ndc = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    if (pc.normalView<SPushConstants::DescriptorCount)
        output.meta = mul(pc.matrices.normal,utbs[pc.normalView][VertexIndex].xyz);
    else
        output.meta = mul(inverse(transpose(pc.matrices.normal)),position);
    output.gridUV = position.xz;
    return output;
}
[shader("pixel")]
float32_t4 BasicFS(SInterpolants input) : SV_Target0
{
    const float32_t3 normal = pc.normalView<SPushConstants::DescriptorCount ? input.meta:reconstructGeometricNormal(input.meta);
    return float32_t4(normalize(normal)*0.5f+promote<float32_t3>(0.5f),1.f);
}

// Debug fragment shader for grid triangle-strips ("snake" order). It alternates
// triangle shading to visualize strip winding and connectivity.
[shader("pixel")]
float32_t4 BasicFSSnake(SInterpolants input) : SV_Target0
{
    float2 uv = input.gridUV * 32.0;
    float2 edge = min(frac(uv), 1.0 - frac(uv));
    float2 aa = max(fwidth(uv), 1e-4.xx);

    float minorX = 1.0 - smoothstep(0.0, aa.x * 1.6, edge.x);
    float minorY = 1.0 - smoothstep(0.0, aa.y * 1.6, edge.y);
    float minor = max(minorX, minorY);

    float2 uvMajor = uv * 0.25;
    float2 edgeMajor = min(frac(uvMajor), 1.0 - frac(uvMajor));
    float majorX = 1.0 - smoothstep(0.0, aa.x * 0.55, edgeMajor.x);
    float majorY = 1.0 - smoothstep(0.0, aa.y * 0.55, edgeMajor.y);
    float major = max(majorX, majorY);

    float lineMask = max(minor * 0.70, major);
    if (lineMask < 0.03)
        discard;

    float3 colMinor = float3(0.58, 0.66, 0.78);
    float3 colMajor = float3(0.76, 0.83, 0.92);
    float3 color = lerp(colMinor, colMajor, saturate(major));
    return float4(color, 1.0);
}

// TODO: do smooth normals on the cone
[shader("vertex")]
SInterpolants ConeVS(uint32_t VertexIndex : SV_VertexID)
{
    const float32_t3 position = utbs[pc.positionView][VertexIndex].xyz;

    SInterpolants output;
    output.ndc = math::linalg::promoted_mul(pc.matrices.worldViewProj,position);
    output.meta = mul(inverse(transpose(pc.matrices.normal)),position);
    output.gridUV = position.xz;
    return output;
}
[shader("pixel")]
float32_t4 ConeFS(SInterpolants input) : SV_Target0
{
    const float32_t3 normal = reconstructGeometricNormal(input.meta);
    return float32_t4(normalize(normal)*0.5f+promote<float32_t3>(0.5f),1.f);
}
