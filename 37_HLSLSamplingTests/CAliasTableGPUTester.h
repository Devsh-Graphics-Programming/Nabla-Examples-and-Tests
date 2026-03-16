#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_ALIAS_TABLE_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/alias_table.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CAliasTableGPUTester final : public ITester<AliasTableInputValues, AliasTableTestResults, AliasTableTestExecutor>
{
	using base_t = ITester<AliasTableInputValues, AliasTableTestResults, AliasTableTestExecutor>;

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

		pass &= verifyTestValue("AliasTable::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("AliasTable::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("AliasTable::forwardWeight", expected.forwardWeight, actual.forwardWeight, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("AliasTable::backwardWeight", expected.backwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-6);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("AliasTable::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("AliasTable::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		// forwardPdf and backwardPdf(generatedIndex) should be identical
		pass &= verifyTestValue("AliasTable::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);

		// forwardWeight == forwardPdf and backwardWeight == backwardPdf (structural invariant)
		pass &= verifyTestValue("AliasTable::forwardWeight == forwardPdf", actual.forwardPdf, actual.forwardWeight, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("AliasTable::backwardWeight == backwardPdf", actual.backwardPdf, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		return pass;
	}
};

#endif
