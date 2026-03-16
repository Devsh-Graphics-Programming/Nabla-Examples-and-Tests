#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CUMULATIVE_PROBABILITY_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CUMULATIVE_PROBABILITY_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/cumulative_probability.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CCumulativeProbabilityGPUTester final : public ITester<CumProbInputValues, CumProbTestResults, CumProbTestExecutor>
{
	using base_t = ITester<CumProbInputValues, CumProbTestResults, CumProbTestExecutor>;

public:
	CCumulativeProbabilityGPUTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	CumProbInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		CumProbInputValues input;
		input.u = uDist(getRandomEngine());
		return input;
	}

	CumProbTestResults determineExpectedResults(const CumProbInputValues& input) override
	{
		CumProbTestResults expected;
		CumProbTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const CumProbTestResults& expected, const CumProbTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;

		if (expected.generatedIndex != actual.generatedIndex)
		{
			pass = false;
			printTestFail("CumProb::generatedIndex", float(expected.generatedIndex), float(actual.generatedIndex), iteration, seed, testType, 0.0, 0.0);
		}

		pass &= verifyTestValue("CumProb::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("CumProb::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("CumProb::forwardWeight", expected.forwardWeight, actual.forwardWeight, iteration, seed, testType, 1e-5, 1e-6);
		pass &= verifyTestValue("CumProb::backwardWeight", expected.backwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-6);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("CumProb::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("CumProb::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		// forwardPdf and backwardPdf(generatedIndex) should be identical
		pass &= verifyTestValue("CumProb::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);

		// forwardWeight == forwardPdf and backwardWeight == backwardPdf (structural invariant)
		pass &= verifyTestValue("CumProb::forwardWeight == forwardPdf", actual.forwardPdf, actual.forwardWeight, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("CumProb::backwardWeight == backwardPdf", actual.backwardPdf, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		return pass;
	}
};

#endif
