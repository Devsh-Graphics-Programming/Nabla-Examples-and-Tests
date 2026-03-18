#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_SPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_SPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/uniform_sphere.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CUniformSphereTester final : public ITester<UniformSphereInputValues, UniformSphereTestResults, UniformSphereTestExecutor>
{
	using base_t = ITester<UniformSphereInputValues, UniformSphereTestResults, UniformSphereTestExecutor>;

public:
	CUniformSphereTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	UniformSphereInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		UniformSphereInputValues input;
		input.u = nbl::hlsl::float32_t2(dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	UniformSphereTestResults determineExpectedResults(const UniformSphereInputValues& input) override
	{
		UniformSphereTestResults expected;
		UniformSphereTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const UniformSphereTestResults& expected, const UniformSphereTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("UniformSphere::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformSphere::pdf", expected.pdf, actual.pdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformSphere::generateInverse", expected.inverted, actual.inverted, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformSphere::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformSphere::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformSphere::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 0.0, 1e-4);
		pass &= verifyTestValue("UniformSphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("UniformSphere::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("UniformSphere::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
