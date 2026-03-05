#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_HEMISPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_HEMISPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/uniform_hemisphere.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CUniformHemisphereTester final : public ITester<UniformHemisphereInputValues, UniformHemisphereTestResults, UniformHemisphereTestExecutor>
{
	using base_t = ITester<UniformHemisphereInputValues, UniformHemisphereTestResults, UniformHemisphereTestExecutor>;

public:
	CUniformHemisphereTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	UniformHemisphereInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		UniformHemisphereInputValues input;
		input.u = nbl::hlsl::float32_t2(dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	UniformHemisphereTestResults determineExpectedResults(const UniformHemisphereInputValues& input) override
	{
		UniformHemisphereTestResults expected;
		UniformHemisphereTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const UniformHemisphereTestResults& expected, const UniformHemisphereTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("UniformHemisphere::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("UniformHemisphere::pdf", expected.pdf, actual.pdf, iteration, seed, testType, 1e-5, 1e-5);
		return pass;
	}
};

#endif
