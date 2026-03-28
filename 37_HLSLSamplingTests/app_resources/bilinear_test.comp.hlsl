#pragma shader_stage(compute)

#include "common/bilinear.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<BilinearInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<BilinearTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	// Hardcode valid bilinear coefficients (all positive).
	const float32_t4 coeffs = float32_t4(0.25f, 0.5f, 0.75f, 1.0f);
	sampling::Bilinear<float32_t> sampler = sampling::Bilinear<float32_t>::create(coeffs);
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t2 acc = (uint32_t2)0;
	uint32_t accPdf = 0;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::Bilinear<float32_t>::cache_type cache;
		float32_t2 generated = sampler.generate(u, cache);
		acc ^= asuint(generated);
		accPdf ^= asuint(sampler.forwardPdf(generated, cache));
	}
	BilinearTestResults result = (BilinearTestResults)0;
	result.generated = asfloat(acc);
	result.forwardPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	BilinearTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
