#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_HEMISPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_HEMISPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/uniform_hemisphere.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

class CUniformHemisphereTester final : public ITester<UniformHemisphereInputValues, UniformHemisphereTestResults, UniformHemisphereTestExecutor>
{
	using base_t = ITester<UniformHemisphereInputValues, UniformHemisphereTestResults, UniformHemisphereTestExecutor>;
	using R = UniformHemisphereTestResults;

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"UniformHemisphere::generate",        &R::generated,   1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::pdf",             &R::pdf,         1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::generateInverse", &R::inverted,    1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::forwardPdf",      &R::forwardPdf,  1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::backwardPdf",     &R::backwardPdf, 1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::forwardWeight",  &R::forwardWeight,  1e-5, 1e-5},
			FieldCheck{"UniformHemisphere::backwardWeight", &R::backwardWeight, 1e-5, 1e-5});
		pass &= verifyTestValue("UniformHemisphere::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 0.0, 1e-4);
		VERIFY_JACOBIAN_OR_SKIP(pass, "UniformHemisphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 5e-2, 5e-2);
		VERIFY_JACOBIAN_OR_SKIP(pass, "UniformHemisphere::inverseJacobianPdf", actual.backwardPdf, actual.inverseJacobianPdf, iteration, seed, testType, 5e-2, 5e-2);
		pass &= verifyTestValue("UniformHemisphere::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("UniformHemisphere::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"UniformHemisphere::forwardPdf",  &R::forwardPdf},
			PdfCheck{"UniformHemisphere::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct UniformHemispherePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::UniformHemisphere<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.02;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "UniformHemisphere"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	// PDF = 1/(2*pi) over hemisphere, so E[1/pdf] = 2*pi
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 2.0 * nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
