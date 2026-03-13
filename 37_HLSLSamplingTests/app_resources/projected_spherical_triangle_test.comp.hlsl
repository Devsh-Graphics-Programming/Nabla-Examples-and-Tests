#pragma shader_stage(compute)

#include "common/projected_spherical_triangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedSphericalTriangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedSphericalTriangleTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Hardcode an axis-aligned octant triangle (valid, non-degenerate, cos_sides=0).
	shapes::SphericalTriangle<float32_t> shape;
	shape.vertices[0] = float32_t3(1.0f, 0.0f, 0.0f);
	shape.vertices[1] = float32_t3(0.0f, 1.0f, 0.0f);
	shape.vertices[2] = float32_t3(0.0f, 0.0f, 1.0f);
	shape.cos_sides = float32_t3(0.0f, 0.0f, 0.0f);
	shape.csc_sides = float32_t3(1.0f, 1.0f, 1.0f);
	sampling::SphericalTriangle<float32_t> sphtri = sampling::SphericalTriangle<float32_t>::create(shape);

	sampling::ProjectedSphericalTriangle<float32_t> sampler;
	sampler.sphtri = sphtri;
	sampler.receiverNormal = float32_t3(0.0f, 0.0f, 1.0f);
	sampler.receiverWasBSDF = false;

	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t3 acc = (uint32_t3)0;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
		acc ^= asuint(sampler.generate(u, cache));
		acc ^= asuint(sampler.forwardPdf(cache));
	}
	ProjectedSphericalTriangleTestResults result = (ProjectedSphericalTriangleTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	ProjectedSphericalTriangleTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
