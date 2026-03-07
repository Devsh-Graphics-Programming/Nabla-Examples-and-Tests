#pragma shader_stage(compute)

#include "common/spherical_triangle_jacobian.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<SphericalTriangleJacobianInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<SphericalTriangleJacobianTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Hardcode an axis-aligned octant triangle (valid, non-degenerate, all cos_sides=0).
	// Use invID for baseU so u is a runtime value — prevents loop DCE after unrolling.
	shapes::SphericalTriangle<float32_t> shape;
	shape.vertices[0] = float32_t3(1.0f, 0.0f, 0.0f);
	shape.vertices[1] = float32_t3(0.0f, 1.0f, 0.0f);
	shape.vertices[2] = float32_t3(0.0f, 0.0f, 1.0f);
	shape.cos_sides = float32_t3(0.0f, 0.0f, 0.0f);
	shape.csc_sides = float32_t3(1.0f, 1.0f, 1.0f);
	sampling::SphericalTriangle<float32_t> sampler = sampling::SphericalTriangle<float32_t>::create(shape);

	const float32_t2 baseU = frac(float32_t(invID) * float32_t2(0.6180339887f, 0.7548776662f));
	uint32_t3 accDir = (uint32_t3)0;
	uint32_t accPdf = 0u;
	[loop]
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = frac(baseU + float32_t(i) * float32_t2(0.6180339887f, 0.7548776662f));
		float32_t rcpPdf;
		float32_t3 generated = sampler.generate(rcpPdf, u);
		accDir ^= asuint(generated);
		accPdf ^= asuint(rcpPdf);
	}
	SphericalTriangleJacobianTestResults result = (SphericalTriangleJacobianTestResults)0;
	result.generated = asfloat(accDir);
	result.forwardRcpPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	SphericalTriangleJacobianTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
