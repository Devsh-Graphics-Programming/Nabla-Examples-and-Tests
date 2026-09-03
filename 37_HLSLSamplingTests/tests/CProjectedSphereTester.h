#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_sphere.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl>

class CProjectedSphereTester final : public ITester<ProjectedSphereInputValues, ProjectedSphereTestResults, ProjectedSphereTestExecutor>
{
	using base_t = ITester<ProjectedSphereInputValues, ProjectedSphereTestResults, ProjectedSphereTestExecutor>;
	using R = ProjectedSphereTestResults;

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"ProjectedSphere::generate",    &R::generated,   5e-5, 5e-5},
			FieldCheck{"ProjectedSphere::cache.pdf",   &R::cachedPdf,   1e-5, 1e-5},
			FieldCheck{"ProjectedSphere::forwardPdf",  &R::forwardPdf,  1e-5, 1e-5},
			FieldCheck{"ProjectedSphere::modifiedU",   &R::modifiedU,   1e-5, 1e-5},
			FieldCheck{"ProjectedSphere::backwardPdf", &R::backwardPdf, 1e-5, 1e-5},
			FieldCheck{"ProjectedSphere::forwardWeight",  &R::forwardWeight,  1e-5, 1e-5},
			FieldCheck{"ProjectedSphere::backwardWeight", &R::backwardWeight, 1e-5, 1e-5});
		pass &= verifyTestValue("ProjectedSphere::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("ProjectedSphere::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-4);
		pass &= verifyTestValue("ProjectedSphere::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-4, 1e-4);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"ProjectedSphere::forwardPdf",  &R::forwardPdf},
			PdfCheck{"ProjectedSphere::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct ProjectedSpherePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::ProjectedSphere<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	// Same domain-clamping bias as ProjectedHemisphere (delegates to it)
	static constexpr float64_t mcNormalizationRelTol = 0.08;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "ProjectedSphere"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t3 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t3>(rng); }
	// PDF = |cos(theta)|/(2*pi), codomain = full sphere, measure = 4*pi
	// (systematically underestimated due to domain clamping near |z|=0)
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 4.0 * nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
