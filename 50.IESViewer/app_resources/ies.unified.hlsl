#include "ies.pcs.hlsl"
using namespace nbl::hlsl;
using namespace nbl::hlsl::examples::ies;

[[vk::binding(0)]] Buffer<float32_t4> utbs[SPushConstants::DescriptorCount];
[[vk::push_constant]] SPushConstants pc;

#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"

struct SInterpolants
{
    float32_t4 ndc : SV_Position;
    float32_t3 meta : COLOR1;
    float32_t2 uv : TEXCOORD0;
    nointerpolation uint triParity : TEXCOORD15;
};

// TODO: all of that for debugging currently, now use IES and project emission onto sphere
// later onto cube I will close my sphere into

static float32_t3 latLongDir(float32_t2 uv)
{
    const float32_t phi = 6.28318530718f * uv.x;
    const float32_t th = 3.14159265359f * uv.y;
    const float32_t s = sin(th), c = cos(th);
    return float32_t3(s * cos(phi), c, s * sin(phi));
}

[shader("vertex")]
SInterpolants SphereVS(uint32_t VertexIndex : SV_VertexID)
{
    const uint32_t W = pc.resX, H = pc.resY;
    const uint32_t i = VertexIndex % W, j = VertexIndex / W;

    // for sphere geometry created from our grid we need to make sure the surface is closed, aligned at U/V edges
    const float32_t2 uv = float32_t2(
        (float32_t(i)) / float32_t(W), 
        (float32_t(j)) / float32_t(H)
    );
    const float32_t vPos = (j == 0u) ? 0.0f : (j == H - 1u) ? 1.0f : uv.y;
    const float32_t uPos = (i == W - 1u) ? 1.0f : uv.x;
    const float32_t2 uvPos = float32_t2(uPos, vPos);

    const float32_t3 dir = latLongDir(uvPos);
    const float32_t3 pos = pc.radius * dir;

    SInterpolants o;
    o.ndc = math::linalg::promoted_mul(pc.matrices.worldViewProj, pos);
    o.meta = mul(pc.matrices.normal, dir);
    o.triParity = (VertexIndex & 1u);

    // but we want to sample centers
    o.uv = float32_t2(
        (float32_t(i) + 0.5f) / float32_t(W),
        (float32_t(j) + 0.5f) / float32_t(H)
    );
    return o;
}

[shader("pixel")]
float32_t4 SphereFS(SInterpolants input) : SV_Target0
{
    const float32_t2 uv = input.uv;
    const int32_t2 cell = int32_t2(floor(uv * float32_t2(pc.resX, pc.resY)));
    const int parity = (cell.x + cell.y) & 1;
    const float32_t3 base = parity == 0 ? float32_t3(0.88f,0.88f,0.88f) : float32_t3(0.68f,0.68f,0.68f);

    float32_t3 N = normalize(input.meta);
    float32_t nview = saturate(0.5f + 0.5f * N.z);
    float32_t grad = pow(nview, 0.5f);
    float32_t rim = pow(1.0f - nview, 2.0f) * 0.25f;

    float32_t3 col = base * (0.2f + 0.8f * grad) + rim;
    return float32_t4(col, 1.0f);
}