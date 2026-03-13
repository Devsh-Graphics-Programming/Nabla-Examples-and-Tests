#pragma shader_stage(compute)

#include "common/uniform_sphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

[[vk::binding(0, 0)]] RWStructuredBuffer<UniformSphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<UniformSphereTestResults> outputTestValues;

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t3 acc = (uint32_t3)0;
	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t2 u = float32_t2(rng(), rng()) * toFloat;
		sampling::UniformSphere<float32_t> sampler;
		sampling::UniformSphere<float32_t>::cache_type cache;
		acc ^= asuint(sampler.generate(u, cache));
		acc ^= asuint(sampler.forwardPdf(cache));
	}
	UniformSphereTestResults result = (UniformSphereTestResults)0;
	result.generated = asfloat(acc);
	outputTestValues[invID] = result;
#else
	UniformSphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
