#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BOX_MULLER_TRANSFORM_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BOX_MULLER_TRANSFORM_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/box_muller_transform.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CBoxMullerTransformTester final : public ITester<BoxMullerTransformInputValues, BoxMullerTransformTestResults, BoxMullerTransformTestExecutor>
{
	using base_t = ITester<BoxMullerTransformInputValues, BoxMullerTransformTestResults, BoxMullerTransformTestExecutor>;

public:
	CBoxMullerTransformTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	BoxMullerTransformInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> stddevDist(0.1f, 5.0f);
		// Avoid u.x near 0 to prevent log(0) = -inf
		std::uniform_real_distribution<float> uDist(1e-4f, 1.0f - 1e-4f);

		BoxMullerTransformInputValues input;
		input.stddev = stddevDist(getRandomEngine());
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		return input;
	}

	BoxMullerTransformTestResults determineExpectedResults(const BoxMullerTransformInputValues& input) override
	{
		BoxMullerTransformTestResults expected;
		BoxMullerTransformTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const BoxMullerTransformTestResults& expected, const BoxMullerTransformTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("BoxMullerTransform::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-5, 2e-3); // tolerated
		pass &= verifyTestValue("BoxMullerTransform::cache.pdf", expected.cachedPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-3);
		pass &= verifyTestValue("BoxMullerTransform::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-3);
		pass &= verifyTestValue("BoxMullerTransform::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-3);
		pass &= verifyTestValue("BoxMullerTransform::separateBackwardPdf", expected.separateBackwardPdf, actual.separateBackwardPdf, iteration, seed, testType, 1e-5, 1e-3);

		// Joint PDF == product of marginal PDFs (independent random variables)
		pass &= verifyTestValue("BoxMullerTransform::jointPdf == pdf product", actual.backwardPdf, actual.separateBackwardPdf.x * actual.separateBackwardPdf.y, iteration, seed, testType, 1e-5, 1e-5);

		// forwardPdf must return the same value stored in cache.pdf by generate
		pass &= verifyTestValue("BoxMullerTransform::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("BoxMullerTransform::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("BoxMullerTransform::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
