#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

#ifdef BENCH_ITERS
#include "common/discrete_sampler_bench.hlsl"
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>

[[vk::push_constant]] CumProbPushConstants pc;

struct BdaCumProbAccessor
{
	using value_type = float32_t;
	value_type get(uint32_t i) { return vk::RawBufferLoad<value_type>(addr + uint64_t(sizeof(value_type)) * uint64_t(i)); }
	value_type operator[](uint32_t i) { return get(i); } // needed for upper_bound

	uint64_t addr;
};

using BenchCumProbSampler = sampling::CumulativeProbabilitySampler<float32_t, BdaCumProbAccessor>;
#else
#include "common/cumulative_probability.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<CumProbInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<CumProbTestResults> outputTestValues;
#endif

[numthreads(64, 1, 1)]
[shader("compute")]
void main()
{
	const uint32_t invID = nbl::hlsl::glsl::gl_GlobalInvocationID().x;

#ifdef BENCH_ITERS
	BdaCumProbAccessor cumProbAcc;
	cumProbAcc.addr = pc.cumProbAddress;
	BenchCumProbSampler sampler = BenchCumProbSampler::create(cumProbAcc, pc.tableSize);

	float32_t xi = float32_t(nbl::hlsl::glsl::bitfieldReverse(invID)) / float32_t(~0u);
	NBL_CONSTEXPR float32_t goldenRatio = 0.6180339887498949f;
	uint32_t acc = 0u;

	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t u = frac(xi + float32_t(i) * goldenRatio);
		BenchCumProbSampler::cache_type cache;
		acc ^= sampler.generate(u, cache);
		acc ^= asuint(sampler.forwardPdf(cache));
	}

	vk::RawBufferStore<uint32_t>(pc.outputAddress + uint64_t(sizeof(uint32_t)) * uint64_t(invID), acc);
#else
	CumProbTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
