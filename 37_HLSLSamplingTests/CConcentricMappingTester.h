#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CONCENTRIC_MAPPING_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CONCENTRIC_MAPPING_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/concentric_mapping.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CConcentricMappingTester final : public ITester<ConcentricMappingInputValues, ConcentricMappingTestResults, ConcentricMappingTestExecutor>
{
	using base_t = ITester<ConcentricMappingInputValues, ConcentricMappingTestResults, ConcentricMappingTestExecutor>;

public:
	CConcentricMappingTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	ConcentricMappingInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		ConcentricMappingInputValues input;
		input.u = nbl::hlsl::float32_t2(dist(getRandomEngine()), dist(getRandomEngine()));
		return input;
	}

	ConcentricMappingTestResults determineExpectedResults(const ConcentricMappingInputValues& input) override
	{
		ConcentricMappingTestResults expected;
		ConcentricMappingTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const ConcentricMappingTestResults& expected, const ConcentricMappingTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		pass &= verifyTestValue("ConcentricMapping::concentricMapping", expected.mapped, actual.mapped, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentricMapping::invertConcentricMapping", expected.inverted, actual.inverted, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentringMapping::roundtripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentricMapping::cache.pdf", expected.cachedPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentringMapping::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentricMapping::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentringMapping::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ConcentringMapping::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-5, 1e-5);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("ConcentricMapping::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("ConcentricMapping::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
