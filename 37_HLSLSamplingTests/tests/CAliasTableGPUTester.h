#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/alias_table.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

class CAliasTableGPUTester final : public ITester<AliasTableInputValues, AliasTableTestResults, AliasTableTestExecutor>
{
	using base_t = ITester<AliasTableInputValues, AliasTableTestResults, AliasTableTestExecutor>;
	using R = AliasTableTestResults;

public:
	CAliasTableGPUTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	AliasTableInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		AliasTableInputValues input;
		input.u = uDist(getRandomEngine());
		return input;
	}

	AliasTableTestResults determineExpectedResults(const AliasTableInputValues& input) override
	{
		AliasTableTestResults expected;
		AliasTableTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const AliasTableTestResults& expected, const AliasTableTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;

		if (expected.generatedIndex != actual.generatedIndex)
		{
			pass = false;
			printTestFail("AliasTable::generatedIndex", float(expected.generatedIndex), float(actual.generatedIndex), iteration, seed, testType, 0.0, 0.0);
		}

		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"AliasTable::forwardPdf",     &R::forwardPdf,     1e-5, 1e-6},
			FieldCheck{"AliasTable::backwardPdf",    &R::backwardPdf,    1e-5, 1e-6},
			FieldCheck{"AliasTable::forwardWeight",  &R::forwardWeight,  1e-5, 1e-6},
			FieldCheck{"AliasTable::backwardWeight", &R::backwardWeight, 1e-5, 1e-6});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"AliasTable::forwardPdf",  &R::forwardPdf},
			PdfCheck{"AliasTable::backwardPdf", &R::backwardPdf});

		// Structural invariants
		pass &= verifyTestValue("AliasTable::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("AliasTable::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		return pass;
	}
};

#endif
