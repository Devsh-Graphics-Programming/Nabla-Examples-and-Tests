#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ALIAS_TABLE_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_COMMON_ALIAS_TABLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include "array_accessor.hlsl"
#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>

using namespace nbl::hlsl;

NBL_CONSTEXPR uint32_t AliasTestTableSize = 4;
// Log2N = ceil_log2(N) minimises quantisation drift on the stayProb unorm
// (here 30 unorm bits, essentially lossless).
NBL_CONSTEXPR uint32_t AliasTestLog2N     = 2;

using AliasTestPdfAccessor        = ArrayAccessor<float32_t, AliasTestTableSize>;
using AliasTestPackedWordAccessor = ArrayAccessor<uint32_t, AliasTestTableSize>;

// Dedicated struct-valued accessor for PackedAliasEntryB. Field-wise copy
// sidesteps HLSL's struct functional-cast ambiguity.
struct AliasTestEntryBAccessor
{
	using value_type = sampling::PackedAliasEntryB<float32_t>;

	template<typename V, typename I>
	void get(I i, NBL_REF_ARG(V) val) NBL_CONST_MEMBER_FUNC
	{
		val.packedWord = data[i].packedWord;
		val.ownPdf     = data[i].ownPdf;
	}

	value_type data[AliasTestTableSize];
};

struct AliasTableInputValues
{
	float32_t u;
};

struct AliasTableTestResults
{
	uint32_t  generatedIndex;
	float32_t forwardPdf;
	float32_t backwardPdf;
	float32_t forwardWeight;
	float32_t backwardWeight;
	float32_t jacobianProduct;
};

// Pre-computed alias table for weights {1, 2, 3, 4}:
//   pdf       = {0.1, 0.2, 0.3, 0.4}
//   stayProb  = {0.4, 0.8, 1.0, 0.8}
//   alias     = {3,   3,   2,   2}
//
// Log2N = 2 unorm encoding (30 bits for stayProb, 2 bits for alias):
//   packedWord = (alias & 0x3) | (round(stayProb * ((1u<<30) - 1)) << 2)
//   bin 0: (3) | (429496729  << 2) = 0x66666667
//   bin 1: (3) | (858993458  << 2) = 0xCCCCCCCB
//   bin 2: (2) | (1073741823 << 2) = 0xFFFFFFFE
//   bin 3: (2) | (858993458  << 2) = 0xCCCCCCCA

struct PackedAliasATestExecutor
{
	void operator()(NBL_CONST_REF_ARG(AliasTableInputValues) input, NBL_REF_ARG(AliasTableTestResults) output)
	{
		AliasTestPackedWordAccessor wordAcc;
		wordAcc.data[0] = 0x66666667u;
		wordAcc.data[1] = 0xCCCCCCCBu;
		wordAcc.data[2] = 0xFFFFFFFEu;
		wordAcc.data[3] = 0xCCCCCCCAu;

		AliasTestPdfAccessor pdfAcc;
		pdfAcc.data[0] = 0.1f;
		pdfAcc.data[1] = 0.2f;
		pdfAcc.data[2] = 0.3f;
		pdfAcc.data[3] = 0.4f;

		using Sampler = sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, AliasTestPackedWordAccessor, AliasTestPdfAccessor, AliasTestLog2N>;
		Sampler sampler = Sampler::create(wordAcc, pdfAcc, AliasTestTableSize);

		Sampler::cache_type cache;
		output.generatedIndex  = sampler.generate(input.u, cache);
		output.forwardPdf      = sampler.forwardPdf(input.u, cache);
		output.backwardPdf     = sampler.backwardPdf(output.generatedIndex);
		output.forwardWeight   = sampler.forwardWeight(input.u, cache);
		output.backwardWeight  = sampler.backwardWeight(output.generatedIndex);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

struct PackedAliasBTestExecutor
{
	void operator()(NBL_CONST_REF_ARG(AliasTableInputValues) input, NBL_REF_ARG(AliasTableTestResults) output)
	{
		AliasTestEntryBAccessor entryAcc;
		entryAcc.data[0].packedWord = 0x66666667u; entryAcc.data[0].ownPdf = 0.1f;
		entryAcc.data[1].packedWord = 0xCCCCCCCBu; entryAcc.data[1].ownPdf = 0.2f;
		entryAcc.data[2].packedWord = 0xFFFFFFFEu; entryAcc.data[2].ownPdf = 0.3f;
		entryAcc.data[3].packedWord = 0xCCCCCCCAu; entryAcc.data[3].ownPdf = 0.4f;

		AliasTestPdfAccessor pdfAcc;
		pdfAcc.data[0] = 0.1f;
		pdfAcc.data[1] = 0.2f;
		pdfAcc.data[2] = 0.3f;
		pdfAcc.data[3] = 0.4f;

		using Sampler = sampling::PackedAliasTableB<float32_t, float32_t, uint32_t, AliasTestEntryBAccessor, AliasTestPdfAccessor, AliasTestLog2N>;
		Sampler sampler = Sampler::create(entryAcc, pdfAcc, AliasTestTableSize);

		Sampler::cache_type cache;
		output.generatedIndex  = sampler.generate(input.u, cache);
		output.forwardPdf      = sampler.forwardPdf(input.u, cache);
		output.backwardPdf     = sampler.backwardPdf(output.generatedIndex);
		output.forwardWeight   = sampler.forwardWeight(input.u, cache);
		output.backwardWeight  = sampler.backwardWeight(output.generatedIndex);
		output.jacobianProduct = (float32_t(1.0) / output.forwardPdf) * output.backwardPdf;
	}
};

#endif
