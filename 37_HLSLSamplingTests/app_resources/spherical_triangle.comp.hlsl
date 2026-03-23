#pragma shader_stage(compute)

#include "common/spherical_triangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<SphericalTriangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<SphericalTriangleTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Hardcode an axis-aligned octant triangle (valid, non-degenerate, all cos_sides=0).
	// Use invID for baseU so u is a runtime value — prevents loop DCE after unrolling.
	const float32_t3 verts[3] = { float32_t3(1.0f, 0.0f, 0.0f), float32_t3(0.0f, 1.0f, 0.0f), float32_t3(0.0f, 0.0f, 1.0f) };
	shapes::SphericalTriangle<float32_t> shape = shapes::SphericalTriangle<float32_t>::createFromUnitSphereVertices(verts);
	sampling::SphericalTriangle<float32_t> sampler = sampling::SphericalTriangle<float32_t>::create(shape);

	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t3 accDir = (uint32_t3)0;
	uint32_t accPdf = 0u;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::SphericalTriangle<float32_t>::cache_type cache;
		float32_t3 generated = sampler.generate(u, cache);
		accDir ^= asuint(generated);
		accPdf ^= asuint(sampler.forwardPdf(cache));
	}
	SphericalTriangleTestResults result = (SphericalTriangleTestResults)0;
	result.generated = asfloat(accDir);
	result.forwardPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	SphericalTriangleTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
