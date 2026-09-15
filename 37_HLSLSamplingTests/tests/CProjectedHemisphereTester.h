#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_HEMISPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_HEMISPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_hemisphere.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>

class CProjectedHemisphereTester final : public ITester<ProjectedHemisphereInputValues, ProjectedHemisphereTestResults, ProjectedHemisphereTestExecutor>
{
	using base_t = ITester<ProjectedHemisphereInputValues, ProjectedHemisphereTestResults, ProjectedHemisphereTestExecutor>;
	using R = ProjectedHemisphereTestResults;

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"ProjectedHemisphere::generate",        &R::generated,   5e-5, 5e-5},
			FieldCheck{"ProjectedHemisphere::cache.pdf",       &R::cachedPdf,   5e-5, 5e-5},
			FieldCheck{"ProjectedHemisphere::forwardPdf",      &R::forwardPdf,  5e-5, 5e-5},
			FieldCheck{"ProjectedHemisphere::generateInverse", &R::inverted,    5e-5, 5e-5},
			FieldCheck{"ProjectedHemisphere::backwardPdf",     &R::backwardPdf, 1e-4, 1e-4},
			FieldCheck{"ProjectedHemisphere::forwardWeight",  &R::forwardWeight,  5e-5, 5e-5},
			FieldCheck{"ProjectedHemisphere::backwardWeight", &R::backwardWeight, 1e-4, 1e-4});
		pass &= verifyTestValue("ProjectedHemisphere::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedHemisphere::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 5e-4, 1e-4);
		pass &= verifyTestValue("ProjectedHemisphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue("ProjectedHemisphere::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue("ProjectedHemisphere::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-4, 1e-4);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"ProjectedHemisphere::forwardPdf",  &R::forwardPdf},
			PdfCheck{"ProjectedHemisphere::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct ProjectedHemispherePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::ProjectedHemisphere<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	// Higher tolerance: generate() clamps domain with u*0.99999+0.000005 to avoid
	// z=0 singularity, which systematically underestimates E[1/pdf] since 1/pdf=pi/z
	// diverges at the horizon. The ~6% bias is expected and intentional.
	static constexpr float64_t mcNormalizationRelTol = 0.08;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "ProjectedHemisphere"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	// E_u[1/pdf] = integral_{hemisphere} 1 d_omega = 2*pi
	// (systematically underestimated due to domain clamping near z=0)
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 2.0 * nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
