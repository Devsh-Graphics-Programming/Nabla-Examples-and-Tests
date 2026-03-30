#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ALIAS_TABLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ALIAS_TABLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "array_accessor.hlsl"
#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t AliasTestTableSize = 4;

using AliasTestProbAccessor = ArrayAccessor<float32_t, AliasTestTableSize>;
using AliasTestAliasAccessor = ArrayAccessor<uint32_t, AliasTestTableSize>;
using AliasTestPdfAccessor = ArrayAccessor<float32_t, AliasTestTableSize>;

using AliasTestSampler = sampling::AliasTable<float32_t, float32_t, uint32_t, AliasTestProbAccessor, AliasTestAliasAccessor, AliasTestPdfAccessor>;

struct AliasTableInputValues
{
	float32_t u;
};

struct AliasTableTestResults
{
	uint32_t generatedIndex;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
};

// Pre-computed alias table for weights {1, 2, 3, 4}:
//   pdf  = {0.1, 0.2, 0.3, 0.4}
//   prob = {0.4, 0.8, 1.0, 0.8}
//   alias = {3, 3, 2, 2}
struct AliasTableTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(AliasTableInputValues) input, NBL_REF_ARG(AliasTableTestResults) output)
	{
		AliasTestProbAccessor probAcc;
		probAcc.data[0] = 0.4f;
		probAcc.data[1] = 0.8f;
		probAcc.data[2] = 1.0f;
		probAcc.data[3] = 0.8f;

		AliasTestAliasAccessor aliasAcc;
		aliasAcc.data[0] = 3u;
		aliasAcc.data[1] = 3u;
		aliasAcc.data[2] = 2u;
		aliasAcc.data[3] = 2u;

		AliasTestPdfAccessor pdfAcc;
		pdfAcc.data[0] = 0.1f;
		pdfAcc.data[1] = 0.2f;
		pdfAcc.data[2] = 0.3f;
		pdfAcc.data[3] = 0.4f;

		AliasTestSampler sampler = AliasTestSampler::create(probAcc, aliasAcc, pdfAcc, AliasTestTableSize);

		AliasTestSampler::cache_type cache;
		output.generatedIndex = sampler.generate(input.u, cache);
		output.forwardPdf = sampler.forwardPdf(input.u, cache);
		output.backwardPdf = sampler.backwardPdf(output.generatedIndex);
		output.forwardWeight = sampler.forwardWeight(input.u, cache);
		output.backwardWeight = sampler.backwardWeight(output.generatedIndex);
	}
};

#endif
