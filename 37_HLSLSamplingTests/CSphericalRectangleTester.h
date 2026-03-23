#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_RECTANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_RECTANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/spherical_rectangle.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CSphericalRectangleTester final : public ITester<SphericalRectangleInputValues, SphericalRectangleTestResults, SphericalRectangleTestExecutor>
{
	using base_t = ITester<SphericalRectangleInputValues, SphericalRectangleTestResults, SphericalRectangleTestExecutor>;

public:
	CSphericalRectangleTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	SphericalRectangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> sizeDist(0.5f, 3.0f);
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		SphericalRectangleInputValues input;
		// Observer at origin, rect placed in front (negative Z) so the solid angle is valid.
		input.observer = nbl::hlsl::float32_t3(0.0f, 0.0f, 0.0f);
		const float width = sizeDist(getRandomEngine());
		const float height = sizeDist(getRandomEngine());
		input.rectOrigin = nbl::hlsl::float32_t3(0.0f, 0.0f, -2.0f);
		input.right = nbl::hlsl::float32_t3(width, 0.0f, 0.0f);
		input.up = nbl::hlsl::float32_t3(0.0f, height, 0.0f);
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		return input;
	}

	SphericalRectangleTestResults determineExpectedResults(const SphericalRectangleInputValues& input) override
	{
		SphericalRectangleTestResults expected;
		SphericalRectangleTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const SphericalRectangleTestResults& expected, const SphericalRectangleTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("SphericalRectangle::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("SphericalRectangle::pdf", expected.pdf, actual.pdf, iteration, seed, testType, 1e-5, 5e-4);
		pass &= verifyTestValue("SphericalRectangle::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 5e-4);

		// PDF positivity and finiteness
		if (!(actual.pdf > 0.0f) || !std::isfinite(actual.pdf))
		{
			pass = false;
			printTestFail("SphericalRectangle::forwardPdf (positive & finite)", 1.0f, actual.pdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("SphericalRectangle::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
