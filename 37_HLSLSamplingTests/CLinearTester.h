#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LINEAR_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LINEAR_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/linear.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CLinearTester final : public ITester<LinearInputValues, LinearTestResults, LinearTestExecutor>
{
	using base_t = ITester<LinearInputValues, LinearTestResults, LinearTestExecutor>;

public:
	CLinearTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	LinearInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> coeffDist(0.1f, 5.0f);
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		LinearInputValues input;
		input.coeffs = nbl::hlsl::float32_t2(coeffDist(getRandomEngine()), coeffDist(getRandomEngine()));
		input.u = uDist(getRandomEngine());
		return input;
	}

	LinearTestResults determineExpectedResults(const LinearInputValues& input) override
	{
		LinearTestResults expected;
		LinearTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const LinearTestResults& expected, const LinearTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("Linear::generate", expected.generated, actual.generated, iteration, seed, testType, 5e-2, 5e-5);
		pass &= verifyTestValue("Linear::generateInverse", expected.generateInversed, actual.generateInversed, iteration, seed, testType, 5e-2, 5e-5);
		pass &= verifyTestValue("Linear::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 5e-2, 1e-5);
		pass &= verifyTestValue("Linear::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 5e-2, 1e-5);
		pass &= verifyTestValue("Linear::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 1e-2, 5e-3);
		pass &= verifyTestValue("Linear::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("Linear::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("Linear::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
