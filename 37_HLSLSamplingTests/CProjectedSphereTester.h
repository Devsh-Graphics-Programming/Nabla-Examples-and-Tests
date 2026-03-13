#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_sphere.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CProjectedSphereTester final : public ITester<ProjectedSphereInputValues, ProjectedSphereTestResults, ProjectedSphereTestExecutor>
{
	using base_t = ITester<ProjectedSphereInputValues, ProjectedSphereTestResults, ProjectedSphereTestExecutor>;

public:
	CProjectedSphereTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	ProjectedSphereInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		ProjectedSphereInputValues input;
		input.u = nbl::hlsl::float32_t3(dist(getRandomEngine()), dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	ProjectedSphereTestResults determineExpectedResults(const ProjectedSphereInputValues& input) override
	{
		ProjectedSphereTestResults expected;
		ProjectedSphereTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const ProjectedSphereTestResults& expected, const ProjectedSphereTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("ProjectedSphere::generate", expected.generated, actual.generated, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedSphere::cache.pdf", expected.cachedPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedSphere::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedSphere::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedSphere::modifiedU", expected.modifiedU, actual.modifiedU, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedSphere::generateInverse", expected.inverted, actual.inverted, iteration, seed, testType, 5e-5, 5e-5);
		pass &= verifyTestValue("ProjectedSphere::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		// roundtripError covers only xy (z is intentionally lossy in generateInverse)
		pass &= verifyTestValue("ProjectedSphere::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 0.0, 1e-4);
		pass &= verifyTestValue("ProjectedSphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("ProjectedSphere::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("ProjectedSphere::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
