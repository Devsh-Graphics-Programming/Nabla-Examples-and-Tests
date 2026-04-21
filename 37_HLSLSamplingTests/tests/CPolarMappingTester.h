#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_POLAR_MAPPING_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_POLAR_MAPPING_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/polar_mapping.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/polar_mapping.hlsl>

class CPolarMappingTester final : public ITester<PolarMappingInputValues, PolarMappingTestResults, PolarMappingTestExecutor>
{
	using base_t = ITester<PolarMappingInputValues, PolarMappingTestResults, PolarMappingTestExecutor>;
	using R = PolarMappingTestResults;

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"PolarMapping::mapped",      &R::mapped,      1e-5, 1e-5},
			FieldCheck{"PolarMapping::inverted",    &R::inverted,    1e-5, 1e-5},
			FieldCheck{"PolarMapping::forwardPdf",  &R::forwardPdf,  1e-5, 1e-5},
			FieldCheck{"PolarMapping::backwardPdf", &R::backwardPdf, 1e-5, 1e-5},
			FieldCheck{"PolarMapping::forwardWeight",  &R::forwardWeight,  1e-5, 1e-5},
			FieldCheck{"PolarMapping::backwardWeight", &R::backwardWeight, 1e-5, 1e-5});
		pass &= verifyTestValue("PolarMapping::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 1e-5, 1e-5);
		VERIFY_JACOBIAN_OR_SKIP(pass, "PolarMapping::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 9e-2, 9e-2);
		VERIFY_JACOBIAN_OR_SKIP(pass, "PolarMapping::inverseJacobianPdf", actual.backwardPdf, actual.inverseJacobianPdf, iteration, seed, testType, 1e-2, 1e-2);
		pass &= verifyTestValue("PolarMapping::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-5);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"PolarMapping::forwardPdf",  &R::forwardPdf},
			PdfCheck{"PolarMapping::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct PolarMappingPropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::PolarMapping<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.02;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "PolarMapping"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	// PDF = 1/pi on the unit disk, codomain measure = pi
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
