#pragma shader_stage(compute)

#include "common/projected_hemisphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedHemisphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedHemisphereTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t3 acc = (uint32_t3)0;
	uint32_t accPdf = 0;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::ProjectedHemisphere<float32_t> sampler;
		sampling::ProjectedHemisphere<float32_t>::cache_type cache;
		float32_t3 generated = sampler.generate(u, cache);
		acc ^= asuint(generated);
		accPdf ^= asuint(sampler.forwardPdf(u, cache));
	}
	ProjectedHemisphereTestResults result = (ProjectedHemisphereTestResults)0;
	result.generated = asfloat(acc);
	result.forwardPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	ProjectedHemisphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
