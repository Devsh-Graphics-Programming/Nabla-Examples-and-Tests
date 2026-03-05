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
		return pass;
	}
};

#endif
