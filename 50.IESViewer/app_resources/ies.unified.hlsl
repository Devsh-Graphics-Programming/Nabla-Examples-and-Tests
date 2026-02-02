#include "common.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl"
#include "false_color.hlsl"

using namespace nbl::hlsl;
using namespace nbl::hlsl::this_example;
using namespace nbl::hlsl::this_example::ies;
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::binding(0, 0)]] Texture2D inIESCandelaImage[MaxIesImages];
[[vk::binding(0 + 10, 0)]] RWTexture2D<float32_t> outIESCandelaImage[MaxIesImages];
[[vk::binding(0 + 100, 0)]] SamplerState generalSampler;

[[vk::binding(0, 1)]] Buffer<float32_t4> utbs[SpherePC::DescriptorCount];
[[vk::push_constant]] PushConstants pc;

struct Accessor
{
    using angle_t = float32_t;
    using candela_t = float32_t;

    candela_t value(const uint32_t2 ij) { return (nbl::hlsl::bda::__ptr<candela_t>::create(pc.cdc.dataBDA) + pc.cdc.vAnglesCount * ij.x + ij.y).deref().load(); }
    angle_t vAngle(const uint32_t idx) { return (nbl::hlsl::bda::__ptr<angle_t>::create(pc.cdc.vAnglesBDA) + idx).deref().load(); }
    angle_t hAngle(const uint32_t idx) { return (nbl::hlsl::bda::__ptr<angle_t>::create(pc.cdc.hAnglesBDA) + idx).deref().load(); }
    uint32_t vAnglesCount() { return pc.cdc.vAnglesCount; }
    uint32_t hAnglesCount() { return pc.cdc.hAnglesCount; }

    nbl::hlsl::ies::ProfileProperties getProperties() { return pc.cdc.properties; }
};

#include "nbl/builtin/hlsl/ies/texture.hlsl"

struct SInterpolants
{
    float32_t4 ndc : SV_Position;
    float32_t3 latDir : COLOR1;
    float32_t2 uv : TEXCOORD0;
};

using octahedral_t = math::OctahedralTransform<float32_t>;
using texture_t = nbl::hlsl::ies::SProceduralTexture;

[shader("vertex")]
SInterpolants SphereVS(uint32_t vIx : SV_VertexID)
{
    uint32_t2 res;
    inIESCandelaImage[pc.sphere.texIx].GetDimensions(res.x, res.y);

    const float32_t2 resF = float32_t2(res);
    const float32_t2 uv = (float32_t2(vIx % res.x, vIx / res.x) + float32_t2(0.5f, 0.5f)) / resF;
    const float32_t2 halfMinusHalfPixel = float32_t2(0.5f, 0.5f) - float32_t2(0.5f, 0.5f) / resF;

    const float32_t3 dir = octahedral_t::uvToDir(uv, halfMinusHalfPixel);
    float32_t3 pos = dir;
    const bool useCube = (pc.sphere.mode & ESM_CUBE) != 0;
    if (useCube)
    {
        const float32_t3 ad = abs(dir);
        const float32_t maxAxis = max(ad.x, max(ad.y, ad.z));
        pos = dir / maxAxis;
    }
    pos *= pc.sphere.radius;

    SInterpolants o;
    o.ndc = math::linalg::promoted_mul(pc.sphere.matrices.worldViewProj, pos);
    o.latDir = dir;
    o.uv = uv;

    return o;
}

[shader("pixel")]
float32_t4 SpherePS(SInterpolants input) : SV_Target0
{
    uint32_t2 res;
    inIESCandelaImage[pc.sphere.texIx].GetDimensions(res.x, res.y);
    float32_t2 uv = input.uv;

    const bool dontInterpolateUV = (pc.sphere.mode & ESM_OCTAHEDRAL_UV_INTERPOLATE) == 0;
    if (dontInterpolateUV)
    {
        float32_t2 pixel = floor(uv * float32_t2(res));
        uv = (pixel + float32_t2(0.5f, 0.5f)) / float32_t2(res);
    }

    float32_t I = inIESCandelaImage[pc.sphere.texIx].SampleLevel(generalSampler, uv, 0.0f).r;
    const bool useFalseColor = (pc.sphere.mode & ESM_FALSE_COLOR) != 0;
    float32_t3 col = useFalseColor ? falseColor(I) : float32_t3(I, I, I);

    return float32_t4(col, 1.0f);
}

[numthreads(WorkgroupDimension, WorkgroupDimension, 1)]
[shader("compute")]
void CdcCS(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t2 destinationSize;
    outIESCandelaImage[pc.cdc.texIx].GetDimensions(destinationSize.x, destinationSize.y);
    const uint32_t2 pixelCoordinates = uint32_t2(glsl::gl_GlobalInvocationID().x, glsl::gl_GlobalInvocationID().y);
    if (all(pixelCoordinates < destinationSize))
    {
        Accessor accessor;
        texture_t txt;
        nbl::hlsl::ies::IESTextureInfo info = (nbl::hlsl::bda::__ptr<nbl::hlsl::ies::IESTextureInfo>::create(pc.cdc.txtInfoBDA) + pc.cdc.texIx).deref().load();
        txt.info = info;
        outIESCandelaImage[pc.cdc.texIx][pixelCoordinates] = txt.__call(accessor, pixelCoordinates);
    }
}

float32_t plot(float32_t cand, float32_t pct, float32_t bold)
{
    return smoothstep(pct - 0.005f * bold, pct, cand) - smoothstep(pct, pct + 0.005f * bold, cand);
}

// vertical cut of IES (i.e. cut by plane x = 0)
float32_t f(float32_t2 uv)
{
    float32_t3 dir = normalize(float32_t3(uv.x, 0.001f, uv.y));
    if (pc.cdc.zAngleDegreeRotation != 0.f)
    {
        float32_t rad = radians(pc.cdc.zAngleDegreeRotation);
        float32_t s = sin(rad);
        float32_t c = cos(rad);

        dir = float32_t3(
            c * dir.x - s * dir.y,
            s * dir.x + c * dir.y,
            dir.z
        );
    }

    uint32_t2 res;
    inIESCandelaImage[pc.cdc.texIx].GetDimensions(res.x, res.y);
    float32_t2 halfMinusHalfPixel = 0.5f - 0.5f / float32_t2(res);
    float32_t2 uvOcta = octahedral_t::dirToUV(dir, halfMinusHalfPixel);

    return inIESCandelaImage[pc.cdc.texIx].SampleLevel(generalSampler, uvOcta, 0u).x;
}

[shader("pixel")]
float32_t4 CdcPS(SVertexAttributes input) : SV_Target0
{
    switch (pc.cdc.mode)
    {
        case 0:
        {
            float32_t2 ndc = input.uv * 2.f - 1.f;
            float32_t dist = length(ndc) * 1.015625f;
            float32_t p = plot(dist, 1.0f, 0.75f);
            float32_t3 col = float32_t3(p, p, p);

            float32_t normalizedStrength = f(ndc);
            if (dist < normalizedStrength)
                col += float32_t3(1.0f, 0.0f, 0.0f);

            return float32_t4(col, 1.0f);
        }
        case 1:
            return float32_t4(inIESCandelaImage[pc.cdc.texIx].Sample(generalSampler, input.uv).x, 0.f, 0.f, 1.f);
        default:
            return float32_t4(0.f, 0.f, 0.f, 0.f);
    }
}
