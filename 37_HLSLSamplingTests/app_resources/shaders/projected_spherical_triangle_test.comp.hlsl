#pragma shader_stage(compute)

#include "../common/projected_spherical_triangle.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#ifdef BENCH_ITERS
[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedSphericalTriangleInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedSphericalTriangleTestResults> outputTestValues;
#endif

#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 64
#endif
[numthreads(WORKGROUP_SIZE, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Perturb vertices and normal by invID so the sampler is non-uniform across threads.
	const float32_t perturbation = float32_t(invID) * 1.0e-7f;
	const float32_t3 verts[3] = { normalize(float32_t3(1.0f, perturbation, 0.0f)), normalize(float32_t3(0.0f, 1.0f, perturbation)), normalize(float32_t3(perturbation, 0.0f, 1.0f)) };
	shapes::SphericalTriangle<float32_t> shape = shapes::SphericalTriangle<float32_t>::createFromUnitSphereVertices(verts);
	sampling::ProjectedSphericalTriangle<float32_t> sampler = sampling::ProjectedSphericalTriangle<float32_t>::create(shape, normalize(float32_t3(perturbation, perturbation, 1.0f)), false);

	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t acc = 0u;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
		float32_t3 generated = sampler.generate(u, cache);
		acc ^= asuint(generated.x) ^ asuint(generated.y) ^ asuint(generated.z);
		acc ^= asuint(sampler.forwardPdf(u, cache));
	}
	benchOutput.Store(invID * 4u, acc);
#else
	ProjectedSphericalTriangleTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
