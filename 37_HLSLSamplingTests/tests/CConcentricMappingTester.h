#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CONCENTRIC_MAPPING_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_CONCENTRIC_MAPPING_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/concentric_mapping.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/concentric_mapping.hlsl>

class CConcentricMappingTester final : public ITester<ConcentricMappingInputValues, ConcentricMappingTestResults, ConcentricMappingTestExecutor>
{
	using base_t = ITester<ConcentricMappingInputValues, ConcentricMappingTestResults, ConcentricMappingTestExecutor>;
	using R = ConcentricMappingTestResults;

public:
	CConcentricMappingTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"ConcentricMapping::concentricMapping",      &R::mapped,      1e-5, 1e-5},
			FieldCheck{"ConcentricMapping::invertConcentricMapping", &R::inverted,    1e-5, 1e-5},
			FieldCheck{"ConcentricMapping::forwardPdf",            &R::forwardPdf,  1e-5, 1e-5},
			FieldCheck{"ConcentricMapping::backwardPdf",           &R::backwardPdf, 1e-5, 1e-5},
			FieldCheck{"ConcentricMapping::forwardWeight",  &R::forwardWeight,  1e-5, 1e-5},
			FieldCheck{"ConcentricMapping::backwardWeight", &R::backwardWeight, 1e-5, 1e-5});
		pass &= verifyTestValue("ConcentricMapping::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 1e-5, 1e-5);
		VERIFY_JACOBIAN_OR_SKIP(pass, "ConcentricMapping::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 4e-2, 4e-2);
		VERIFY_JACOBIAN_OR_SKIP(pass, "ConcentricMapping::inverseJacobianPdf", actual.backwardPdf, actual.inverseJacobianPdf, iteration, seed, testType, 4e-2, 4e-2);
		pass &= verifyTestValue("ConcentricMapping::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-5);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"ConcentricMapping::forwardPdf",  &R::forwardPdf},
			PdfCheck{"ConcentricMapping::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct ConcentricMappingPropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::ConcentricMapping<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.02;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "ConcentricMapping"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	// PDF = 1/pi on the unit disk, codomain measure = pi
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
