#include "common.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/ies/texture.hlsl"
#include "nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl"

using namespace nbl::hlsl;
using namespace nbl::hlsl::this_example::ies;
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::binding(0, 0)]] Texture2D inIESCandelaImage[MAX_IES_IMAGES];
[[vk::binding(0 + 10, 0)]] RWTexture2D<float32_t> outIESCandelaImage[MAX_IES_IMAGES];
[[vk::binding(0 + 100, 0)]] SamplerState generalSampler;

[[vk::binding(0, 1)]] Buffer<float32_t4> utbs[SpherePC::DescriptorCount];
[[vk::push_constant]] PushConstants pc;

struct Accessor
{
    using key_t = uint32_t;
    using key_t2 = vector<key_t,2>;
    using value_t = float32_t;

    static key_t vAnglesCount() { return pc.cdc.vAnglesCount; }
    static key_t hAnglesCount() { return pc.cdc.hAnglesCount; }

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t>)
    static inline value_t vAngle(T j) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.cdc.vAnglesBDA) + j).deref().load(); }

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t>)
    static inline value_t hAngle(T i) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.cdc.hAnglesBDA) + i).deref().load(); } 

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t2>)
    static inline value_t value(T ij) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.cdc.dataBDA) + vAnglesCount() * ij.x + ij.y).deref().load(); }

	static inline nbl::hlsl::ies::ProfileProperties getProperties() { return pc.cdc.properties; }
};

struct SInterpolants
{
    float32_t4 ndc : SV_Position;
    float32_t3 latDir : COLOR1;
};

using octahedral_t = math::OctahedralTransform<float32_t>;
using texture_t = nbl::hlsl::ies::Texture<Accessor>;

float32_t3 latLongDir(float32_t2 uv)
{
    const float32_t phi = 6.28318530718f * uv.x;
    const float32_t th = 3.14159265359f * uv.y;
    const float32_t s = sin(th), c = cos(th);
    return float32_t3(s * cos(phi), c, s * sin(phi));
}

[shader("vertex")]
SInterpolants SphereVS(uint32_t VertexIndex : SV_VertexID)
{
    uint32_t2 resolution;
    inIESCandelaImage[pc.sphere.texIx].GetDimensions(resolution.x, resolution.y);

    const uint32_t W = resolution.x, H = resolution.y;
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
    const float32_t3 pos = pc.sphere.radius * dir;

    SInterpolants o;
    o.ndc = math::linalg::promoted_mul(pc.sphere.matrices.worldViewProj, pos);
    o.latDir = dir;

    return o;
}

[shader("pixel")]
float32_t4 SpherePS(SInterpolants input) : SV_Target0
{
    float32_t2 uv = 0.5f * octahedral_t::dirToNDC(input.latDir) + 0.5f;
    float32_t candela = inIESCandelaImage[pc.sphere.texIx].Sample(generalSampler, uv).r;
    float32_t v = 1.0f - exp(-candela);
    return float32_t4(v,v,v,1);
}

[numthreads(WORKGROUP_DIMENSION, WORKGROUP_DIMENSION, 1)]
[shader("compute")]
void CdcCS(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t2 destinationSize;
	outIESCandelaImage[pc.cdc.texIx].GetDimensions(destinationSize.x, destinationSize.y);
	const uint32_t2 pixelCoordinates = uint32_t2(glsl::gl_GlobalInvocationID().x, glsl::gl_GlobalInvocationID().y);
	if (all(pixelCoordinates < destinationSize))
	{
		Accessor accessor; texture_t txt;
		typename texture_t::SInfo info = (nbl::hlsl::bda::__ptr<typename texture_t::SInfo>::create(pc.cdc.txtInfoBDA) + pc.cdc.texIx).deref_restrict().load();
        outIESCandelaImage[pc.cdc.texIx][pixelCoordinates] = txt.eval(accessor, info, pixelCoordinates);
	}
}

float32_t plot(float32_t cand, float32_t pct, float32_t bold)
{
	return smoothstep(pct-0.005*bold, pct, cand) - smoothstep(pct, pct+0.005*bold, cand);
}

// vertical cut of IES (i.e. cut by plane x = 0)
float32_t f(float32_t2 uv) 
{
	float32_t3 dir = normalize(float32_t3(uv.x, 0.001, uv.y));
	if (pc.cdc.zAngleDegreeRotation != 0.f)
	{
		float32_t rad = radians(pc.cdc.zAngleDegreeRotation);
		float32_t s = sin(rad);
		float32_t c = cos(rad);

		// rotate around Z axis
		dir = float32_t3(
			c * dir.x - s * dir.y,
			s * dir.x + c * dir.y,
			dir.z
		);
	}
	return inIESCandelaImage[pc.cdc.texIx].Sample(generalSampler, (0.5f * octahedral_t::dirToNDC(dir) + 0.5f)).x;
}

#include "nbl/builtin/hlsl/ext/FullScreenTriangle/default.vert.hlsl"

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
