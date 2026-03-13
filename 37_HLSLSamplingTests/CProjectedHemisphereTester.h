#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_HEMISPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_HEMISPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_hemisphere.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CProjectedHemisphereTester final : public ITester<ProjectedHemisphereInputValues, ProjectedHemisphereTestResults, ProjectedHemisphereTestExecutor>
{
	using base_t = ITester<ProjectedHemisphereInputValues, ProjectedHemisphereTestResults, ProjectedHemisphereTestExecutor>;

public:
	CProjectedHemisphereTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	ProjectedHemisphereInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		ProjectedHemisphereInputValues input;
		input.u = nbl::hlsl::float32_t2(dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	ProjectedHemisphereTestResults determineExpectedResults(const ProjectedHemisphereInputValues& input) override
	{
		ProjectedHemisphereTestResults expected;
		ProjectedHemisphereTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const ProjectedHemisphereTestResults& expected, const ProjectedHemisphereTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("ProjectedHemisphere::generate", expected.generated, actual.generated, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedHemisphere::cache.pdf", expected.cachedPdf, actual.cachedPdf, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedHemisphere::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedHemisphere::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedHemisphere::generateInverse", expected.inverted, actual.inverted, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedHemisphere::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-1, 1e-2);
		pass &= verifyTestValue("ProjectedHemisphere::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 5e-4, 1e-4);
		pass &= verifyTestValue("ProjectedHemisphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("ProjectedHemisphere::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("ProjectedHemisphere::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
