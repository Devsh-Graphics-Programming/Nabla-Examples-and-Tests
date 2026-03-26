#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_POLAR_MAPPING_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_POLAR_MAPPING_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/polar_mapping.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CPolarMappingTester final : public ITester<PolarMappingInputValues, PolarMappingTestResults, PolarMappingTestExecutor>
{
	using base_t = ITester<PolarMappingInputValues, PolarMappingTestResults, PolarMappingTestExecutor>;

public:
	CPolarMappingTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	PolarMappingInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		PolarMappingInputValues input;
		input.u = nbl::hlsl::float32_t2(dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	PolarMappingTestResults determineExpectedResults(const PolarMappingInputValues& input) override
	{
		PolarMappingTestResults expected;
		PolarMappingTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const PolarMappingTestResults& expected, const PolarMappingTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("PolarMapping::mapped", expected.mapped, actual.mapped, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("PolarMapping::inverted", expected.inverted, actual.inverted, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("PolarMapping::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("PolarMapping::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("PolarMapping::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("PolarMapping::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-5, 1e-5);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("PolarMapping::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("PolarMapping::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
