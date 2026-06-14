#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_SPHERE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_UNIFORM_SPHERE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/uniform_sphere.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

class CUniformSphereTester final : public ITester<UniformSphereInputValues, UniformSphereTestResults, UniformSphereTestExecutor>
{
	using base_t = ITester<UniformSphereInputValues, UniformSphereTestResults, UniformSphereTestExecutor>;
	using R = UniformSphereTestResults;

public:
	CUniformSphereTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

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
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"UniformSphere::generate",        &R::generated,   1e-5, 1e-5},
			FieldCheck{"UniformSphere::pdf",             &R::pdf,         1e-5, 1e-5},
			FieldCheck{"UniformSphere::generateInverse", &R::inverted,    1e-5, 1e-5},
			FieldCheck{"UniformSphere::forwardPdf",      &R::forwardPdf,  1e-5, 1e-5},
			FieldCheck{"UniformSphere::backwardPdf",     &R::backwardPdf, 1e-5, 1e-5},
			FieldCheck{"UniformSphere::forwardWeight",  &R::forwardWeight,  1e-5, 1e-5},
			FieldCheck{"UniformSphere::backwardWeight", &R::backwardWeight, 1e-5, 1e-5});
		pass &= verifyTestValue("UniformSphere::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 0.0, 1e-4);
		VERIFY_JACOBIAN_OR_SKIP(pass, "UniformSphere::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 5e-2, 5e-2);
		VERIFY_JACOBIAN_OR_SKIP(pass, "UniformSphere::inverseJacobianPdf", actual.backwardPdf, actual.inverseJacobianPdf, iteration, seed, testType, 5e-2, 5e-2);
		pass &= verifyTestValue("UniformSphere::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("UniformSphere::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"UniformSphere::forwardPdf",  &R::forwardPdf},
			PdfCheck{"UniformSphere::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct UniformSpherePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::UniformSphere<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 1;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.02;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "UniformSphere"; }
	static sampler_type createRandomSampler(std::mt19937& rng) { return sampler_type(); }
	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	// PDF = 1/(4*pi), so E[1/pdf] = 4*pi
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 4.0 * nbl::hlsl::numbers::pi<float64_t>; }
};

#endif
