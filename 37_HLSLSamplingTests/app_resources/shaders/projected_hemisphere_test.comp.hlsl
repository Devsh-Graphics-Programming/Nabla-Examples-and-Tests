#pragma shader_stage(compute)

#include "../common/projected_hemisphere.hlsl"
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

#ifdef BENCH_ITERS
[[vk::binding(1, 0)]] RWByteAddressBuffer benchOutput;
#else
[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedHemisphereInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<ProjectedHemisphereTestResults> outputTestValues;
#endif

#if !defined(BENCH_SAMPLES_PER_CREATE) && defined(BENCH_ITERS)
#define BENCH_SAMPLES_PER_CREATE (BENCH_ITERS)
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;
#ifdef BENCH_ITERS
	nbl::hlsl::Xoroshiro64Star rng = nbl::hlsl::Xoroshiro64Star::construct(uint32_t2(invID, 0u));
	const float32_t toFloat = asfloat(0x2f800004u);
	uint32_t acc = 0u;
	const uint32_t outerIters = uint32_t(BENCH_ITERS) / uint32_t(BENCH_SAMPLES_PER_CREATE);
	for (uint32_t j = 0u; j < outerIters; j++)
	{
		sampling::ProjectedHemisphere<float32_t> sampler;
		for (uint32_t k = 0u; k < uint32_t(BENCH_SAMPLES_PER_CREATE); k++)
		{
			float32_t2 u = float32_t2(rng(), rng()) * toFloat;
			sampling::ProjectedHemisphere<float32_t>::cache_type cache;
			float32_t3 generated = sampler.generate(u, cache);
			acc ^= asuint(generated.x) ^ asuint(generated.y) ^ asuint(generated.z);
			acc ^= asuint(sampler.forwardPdf(u, cache));
		}
	}
	benchOutput.Store(invID * 4u, acc);
#else
	ProjectedHemisphereTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
