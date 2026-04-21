#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CUMULATIVE_PROBABILITY_GPU_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CUMULATIVE_PROBABILITY_GPU_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/cumulative_probability.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

class CCumulativeProbabilityGPUTester final : public ITester<CumProbInputValues, CumProbTestResults, CumProbTestExecutor>
{
	using base_t = ITester<CumProbInputValues, CumProbTestResults, CumProbTestExecutor>;
	using R = CumProbTestResults;

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

		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"CumProb::forwardPdf",     &R::forwardPdf,     1e-5, 1e-6},
			FieldCheck{"CumProb::backwardPdf",    &R::backwardPdf,    1e-5, 1e-6},
			FieldCheck{"CumProb::forwardWeight",  &R::forwardWeight,  1e-5, 1e-6},
			FieldCheck{"CumProb::backwardWeight", &R::backwardWeight, 1e-5, 1e-6});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"CumProb::forwardPdf",  &R::forwardPdf},
			PdfCheck{"CumProb::backwardPdf", &R::backwardPdf});

		// Structural invariants
		pass &= verifyTestValue("CumProb::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue("CumProb::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("CumProb::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		return pass;
	}
};

#endif
