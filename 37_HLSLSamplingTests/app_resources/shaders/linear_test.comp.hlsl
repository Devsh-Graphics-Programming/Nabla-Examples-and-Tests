#pragma shader_stage(compute)

#include "../common/linear.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#ifdef BENCH_ITERS
[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<LinearInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<LinearTestResults> outputTestValues;
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
	// Perturb coefficients by invID so the sampler is non-uniform across threads.
	const float32_t perturbation = float32_t(invID) * 1.0e-7f;
	const float32_t2 coeffs = float32_t2(0.2f, 0.8f) + perturbation;
	sampling::Linear<float32_t> sampler = sampling::Linear<float32_t>::create(coeffs);
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t acc = 0u;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t u = float32_t(rng()) * toFloat;
		sampling::Linear<float32_t>::cache_type cache;
		float32_t generated = sampler.generate(u, cache);
		acc ^= asuint(generated);
		acc ^= asuint(sampler.forwardPdf(u, cache));
	}
	benchOutput.Store(invID * 4u, acc);
#else
	LinearTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
