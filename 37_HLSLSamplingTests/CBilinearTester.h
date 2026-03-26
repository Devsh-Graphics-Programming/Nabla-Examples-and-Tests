#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BILINEAR_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BILINEAR_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/bilinear.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CBilinearTester final : public ITester<BilinearInputValues, BilinearTestResults, BilinearTestExecutor>
{
	using base_t = ITester<BilinearInputValues, BilinearTestResults, BilinearTestExecutor>;

public:
	CBilinearTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	BilinearInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> coeffDist(0.1f, 5.0f);
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		BilinearInputValues input;
		input.bilinearCoeffs = nbl::hlsl::float32_t4(
			coeffDist(getRandomEngine()), coeffDist(getRandomEngine()),
			coeffDist(getRandomEngine()), coeffDist(getRandomEngine()));
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		return input;
	}

	BilinearTestResults determineExpectedResults(const BilinearInputValues& input) override
	{
		BilinearTestResults expected;
		BilinearTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const BilinearTestResults& expected, const BilinearTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("Bilinear::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-2, 1e-3);
		pass &= verifyTestValue("Bilinear::pdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 5e-3);
		pass &= verifyTestValue("Bilinear::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 5e-3);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("Bilinear::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("Bilinear::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
