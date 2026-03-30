#pragma shader_stage(compute)

#include "common/linear.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<LinearInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<LinearTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Coefficients are hardcoded to a valid non-degenerate distribution.
	const float32_t2 coeffs = float32_t2(0.2f, 0.8f);
	sampling::Linear<float32_t> sampler = sampling::Linear<float32_t>::create(coeffs);
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t acc = 0u;
	uint32_t accPdf = 0u;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t u = float32_t(rng()) * toFloat;
		sampling::Linear<float32_t>::cache_type cache;
		float32_t generated = sampler.generate(u, cache);
		acc ^= asuint(generated);
		accPdf ^= asuint(sampler.forwardPdf(u, cache));
	}
	LinearTestResults result = (LinearTestResults)0;
	result.generated = asfloat(acc);
	result.forwardPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	LinearTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
