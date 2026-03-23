#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CUMULATIVE_PROBABILITY_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_CUMULATIVE_PROBABILITY_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t CumProbTestTableSize = 4;

// Fixed-size array accessors for HLSL/C++ dual compilation
// Needs value_type + operator[] for upper_bound compatibility
struct CumProbTestAccessor
{
	using value_type = float32_t;
	float32_t get(uint32_t i) { return data[i]; }
	float32_t operator[](uint32_t i) { return data[i]; }
	float32_t data[3]; // N-1 stored entries
};

using CumProbTestSampler = sampling::CumulativeProbabilitySampler<float32_t, CumProbTestAccessor>;

struct CumProbInputValues
{
	float32_t u;
};

struct CumProbTestResults
{
	uint32_t generatedIndex;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
};

// Pre-computed CDF table for weights {1, 2, 3, 4}:
//   pdf     = {0.1, 0.2, 0.3, 0.4}
//   cumProb = {0.1, 0.3, 0.6}  (N-1=3 entries, last bucket implicitly 1.0)
struct CumProbTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(CumProbInputValues) input, NBL_REF_ARG(CumProbTestResults) output)
	{
		CumProbTestAccessor cumProbAcc;
		cumProbAcc.data[0] = 0.1f;
		cumProbAcc.data[1] = 0.3f;
		cumProbAcc.data[2] = 0.6f;

		CumProbTestSampler sampler = CumProbTestSampler::create(cumProbAcc, CumProbTestTableSize);

		CumProbTestSampler::cache_type cache;
		output.generatedIndex = sampler.generate(input.u, cache);
		output.forwardPdf = sampler.forwardPdf(cache);
		output.backwardPdf = sampler.backwardPdf(output.generatedIndex);
		output.forwardWeight = sampler.forwardWeight(cache);
		output.backwardWeight = sampler.backwardWeight(output.generatedIndex);
	}
};

#endif
