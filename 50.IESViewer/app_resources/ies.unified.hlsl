#include "common.hlsl"
#include "nbl/builtin/hlsl/bda/__ptr.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/math/octahedral.hlsl"
#include "nbl/builtin/hlsl/math/polar.hlsl"
#include "nbl/builtin/hlsl/ies/sampler.hlsl"
#include "nbl/builtin/hlsl/ext/FullScreenTriangle/SVertexAttributes.hlsl"

using namespace nbl::hlsl;
using namespace nbl::hlsl::this_example::ies;
using namespace nbl::hlsl::ext::FullScreenTriangle;

[[vk::binding(0, 0)]] Texture2D inIESCandelaImage[MAX_IES_IMAGES];
[[vk::binding(1, 0)]] Texture2D inSphericalCoordinatesImage[MAX_IES_IMAGES];
[[vk::binding(2, 0)]] Texture2D inOUVProjectionDirectionImage[MAX_IES_IMAGES];
[[vk::binding(3, 0)]] Texture2D inPassTMaskImage[MAX_IES_IMAGES];
[[vk::binding(0 + 10, 0)]] RWTexture2D<float32_t> outIESCandelaImage[MAX_IES_IMAGES];
[[vk::binding(1 + 10, 0)]] RWTexture2D<float32_t2> outSphericalCoordinatesImage[MAX_IES_IMAGES];
[[vk::binding(2 + 10, 0)]] RWTexture2D<float32_t3> outOUVProjectionDirectionImage[MAX_IES_IMAGES];
[[vk::binding(3 + 10, 0)]] RWTexture2D<float32_t2> outPassTMask[MAX_IES_IMAGES];
[[vk::binding(0 + 100, 0)]] SamplerState generalSampler;
[[vk::binding(0, 1)]] Buffer<float32_t4> utbs[PushConstants::DescriptorCount];
[[vk::push_constant]] PushConstants pc;

struct Accessor
{
    using key_t = uint32_t;
    using key_t2 = vector<key_t,2>;
    using value_t = float32_t;
    using symmetry_t = nbl::hlsl::ies::ProfileProperties::LuminairePlanesSymmetry;

    static key_t vAnglesCount() { return pc.vAnglesCount; }
    static key_t hAnglesCount() { return pc.hAnglesCount; }

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t>)
    static inline value_t vAngle(T j) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.vAnglesBDA) + j).deref().load(); }

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t>)
    static inline value_t hAngle(T i) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.hAnglesBDA) + i).deref().load(); } 

    template<typename T NBL_FUNC_REQUIRES(is_same_v<T, key_t2>)
    static inline value_t value(T ij) { return (nbl::hlsl::bda::__ptr<value_t>::create(pc.dataBDA) + vAnglesCount() * ij.x + ij.y).deref().load(); }

    static inline symmetry_t symmetry() { return (symmetry_t)pc.symmetry; }
};

struct SInterpolants
{
    float32_t4 ndc : SV_Position;
    float32_t3 latDir : COLOR1;
};

using Octahedral = math::OctahedralTransform<float32_t>;
using Polar = math::Polar<float32_t>;
using CSampler = nbl::hlsl::ies::CandelaSampler<Accessor>;

//! Checks if (x,y) /in [0,PI] x [-PI,PI] product
/*
	IES vertical range is [0, 180] degrees
	and horizontal range is [0, 360] degrees
	but for easier computations (MIRROR & MIRROW_REPEAT operations)
	we represent horizontal range as [-180, 180] given spherical coordinates
*/

bool domainPass(const float32_t2 p)
{
    NBL_CONSTEXPR float32_t M_PI = numbers::pi<float32_t>;
    const float32_t2 lb = float32_t2(0, -M_PI);
    const float32_t2 ub = float32_t2(M_PI, M_PI);

    return all(lb <= p) && all(p <= ub);
}

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
    outIESCandelaImage[pc.texIx].GetDimensions(resolution.x, resolution.y); // optimal IES texture size

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
    const float32_t3 pos = pc.sphereRadius * dir;

    SInterpolants o;
    o.ndc = math::linalg::promoted_mul(pc.matrices.worldViewProj, pos);
    o.latDir = dir;

    return o;
}

[shader("pixel")]
float32_t4 SphereFS(SInterpolants input) : SV_Target0
{
    float32_t2 uv = 0.5f * Octahedral::dirToNDC(input.latDir) + 0.5f;
    float32_t candela = inIESCandelaImage[pc.texIx].Sample(generalSampler, uv).r;
    float32_t v = 1.0f - exp(-candela);
    return float32_t4(v,v,v,1);
}

[numthreads(WORKGROUP_DIMENSION, WORKGROUP_DIMENSION, 1)]
[shader("compute")]
void CdcCS(uint32_t3 ID : SV_DispatchThreadID)
{
	uint32_t2 destinationSize;
	outIESCandelaImage[pc.texIx].GetDimensions(destinationSize.x, destinationSize.y);
	const uint32_t2 pixelCoordinates = uint32_t2(glsl::gl_GlobalInvocationID().x, glsl::gl_GlobalInvocationID().y);

	const float32_t VERTICAL_INVERSE = 1.0f / float32_t(destinationSize.x);
	const float32_t HORIZONTAL_INVERSE = 1.0f / float32_t(destinationSize.y);

	if (all(pixelCoordinates < destinationSize))
	{
		const float32_t2 uv = float32_t2((float32_t(pixelCoordinates.x) + 0.5) * VERTICAL_INVERSE, (float32_t(pixelCoordinates.y) + 0.5) * HORIZONTAL_INVERSE);
		const float32_t3 dir = Octahedral::uvToDir(uv);
        Polar polar = Polar::createFromCartesian(dir);

		const float32_t normD = length(dir);
		float32_t2 mask;

		if (1.0f - QUANT_ERROR_ADMISSIBLE <= normD && normD <= 1.0f + QUANT_ERROR_ADMISSIBLE)
			mask.x = 1.f; // pass
		else
			mask.x = 0.f;

        const float32_t2 sCoords = float32_t2(polar.phi, polar.theta);
        if (domainPass(sCoords))
			mask.y = 1.f; // pass
		else
			mask.y = 0.f;

        Accessor accessor;
        CSampler candelaSampler;
        outIESCandelaImage[pc.texIx][pixelCoordinates] = candelaSampler.sample(accessor, polar) / pc.maxIValue;
		outSphericalCoordinatesImage[pc.texIx][pixelCoordinates] = sCoords;
		outOUVProjectionDirectionImage[pc.texIx][pixelCoordinates] = dir;
		outPassTMask[pc.texIx][pixelCoordinates] = mask;
	}
}


float32_t plot(float32_t cand, float32_t pct, float32_t bold)
{
	return smoothstep(pct-0.005*bold, pct, cand) - smoothstep(pct, pct+0.005*bold, cand);
}

// vertical cut of IES (i.e. cut by plane x = 0)
float32_t f(float32_t2 uv) 
{
	return inIESCandelaImage[pc.texIx].Sample(generalSampler, (0.5f * Octahedral::dirToNDC(normalize(float32_t3(uv.x, 0.001, uv.y))) + 0.5f)).x;
}

#include "nbl/builtin/hlsl/ext/FullScreenTriangle/default.vert.hlsl"

[shader("pixel")]
float32_t4 CdcPS(SVertexAttributes input) : SV_Target0
{
	switch (pc.mode)
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
			return float32_t4(inIESCandelaImage[pc.texIx].Sample(generalSampler, input.uv).x, 0.f, 0.f, 1.f);
		case 2:
			return float32_t4(inSphericalCoordinatesImage[pc.texIx].Sample(generalSampler, input.uv).xy, 0.f, 1.f);
		case 3:
			return float32_t4(inOUVProjectionDirectionImage[pc.texIx].Sample(generalSampler, input.uv).xyz, 1.f);
		default:
			return float32_t4(inPassTMaskImage[pc.texIx].Sample(generalSampler, input.uv).xy, 0.f, 1.f);
	}
}
