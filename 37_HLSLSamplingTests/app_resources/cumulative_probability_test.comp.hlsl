#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

#ifdef BENCH_ITERS
#include "common/discrete_sampler_bench.hlsl"
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>

[[vk::push_constant]] CumProbPushConstants pc;

struct BdaCumProbAccessor
{
	using value_type = float32_t;
	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC { val = V(vk::RawBufferLoad<value_type>(addr + uint64_t(sizeof(value_type)) * uint64_t(i))); }
	value_type operator[](uint32_t i) NBL_CONST_MEMBER_FUNC { value_type v; get<value_type, uint32_t>(i, v); return v; }

	uint64_t addr;
};

using BenchCumProbSampler = sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, BdaCumProbAccessor>;
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
	uint32_t accPdf = 0u;

	for (uint32_t i = 0u; i < uint32_t(BENCH_ITERS); i++)
	{
		float32_t u = frac(xi + float32_t(i) * goldenRatio);
		BenchCumProbSampler::cache_type cache;
		uint32_t generated = sampler.generate(u, cache);
		acc ^= generated;
		accPdf ^= asuint(sampler.forwardPdf(generated, cache));
	}

	vk::RawBufferStore<uint32_t>(pc.outputAddress + uint64_t(sizeof(uint32_t)) * uint64_t(invID), acc + accPdf);
#else
	CumProbTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
