#pragma shader_stage(compute)

#include "common/polar_mapping.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<PolarMappingInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<PolarMappingTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t2 acc = (uint32_t2)0;
	uint32_t accPdf = 0;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::PolarMapping<float32_t>::cache_type cache;
		acc ^= asuint(sampling::PolarMapping<float32_t>::generate(u, cache));
	   accPdf ^= asuint(sampling::PolarMapping<float32_t>::forwardPdf(cache));
	}
	PolarMappingTestResults result = (PolarMappingTestResults)0;
	result.mapped = asfloat(acc);
   result.forwardPdf = asfloat(accPdf);
	outputTestValues[invID] = result;
#else
	PolarMappingTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
