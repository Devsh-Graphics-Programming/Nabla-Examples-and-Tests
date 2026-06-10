#pragma shader_stage(compute)

#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>

#ifdef BENCH_ITERS
#include "../common/discrete_sampler_bench.hlsl"
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>

[[vk::push_constant]] CumProbPushConstants pc;

struct BdaCumProbAccessor
{
	using value_type = float32_t;
	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC { val = V(vk::RawBufferLoad<value_type>(addr + uint64_t(sizeof(value_type)) * uint64_t(i), sizeof(value_type))); }

	uint64_t addr;
};

#if defined(NBL_CUMPROB_EYTZINGER)
using BenchCumProbSampler = sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, BdaCumProbAccessor, sampling::CumulativeProbabilityMode::EYTZINGER>;
#elif defined(NBL_CUMPROB_YOLO_READS)
using BenchCumProbSampler = sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, BdaCumProbAccessor, sampling::CumulativeProbabilityMode::YOLO>;
#else
using BenchCumProbSampler = sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, BdaCumProbAccessor, sampling::CumulativeProbabilityMode::TRACKING>;
#endif
#else
#include "../common/cumulative_probability.hlsl"

[[vk::binding(0, 0)]] RWStructuredBuffer<CumProbInputValues> inputTestValues;
[[vk::binding(1, 0)]] RWStructuredBuffer<CumProbTestResults> outputTestValues;
#endif

[numthreads(WORKGROUP_SIZE, 1, 1)]
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
		xi = frac(xi + goldenRatio);
		BenchCumProbSampler::cache_type cache;
		uint32_t generated = sampler.generate(xi, cache);
		acc ^= generated ^ asuint(sampler.forwardPdf(xi, cache));
	}

	vk::RawBufferStore<uint32_t>(pc.outputAddress + uint64_t(sizeof(uint32_t)) * uint64_t(invID), acc);
#else
	CumProbTestExecutor executor;
	executor(inputTestValues[invID], outputTestValues[invID]);
#endif
}
