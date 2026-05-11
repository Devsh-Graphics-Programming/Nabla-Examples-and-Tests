#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_TRIANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_TRIANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/spherical_triangle.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>

class CSphericalTriangleTester final : public ITester<SphericalTriangleInputValues, SphericalTriangleTestResults, SphericalTriangleTestExecutor>
{
	using base_t = ITester<SphericalTriangleInputValues, SphericalTriangleTestResults, SphericalTriangleTestExecutor>;
	using R = SphericalTriangleTestResults;

public:
	CSphericalTriangleTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

private:
	SphericalTriangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		SphericalTriangleInputValues input;

		// Generate well-separated unit vectors for a valid spherical triangle
		do
		{
			input.vertex0 = generateRandomUnitVector(getRandomEngine());
			input.vertex1 = generateRandomUnitVector(getRandomEngine());
			input.vertex2 = generateRandomUnitVector(getRandomEngine());
		} while (!isValidSphericalTriangle(input.vertex0, input.vertex1, input.vertex2));

		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		m_inputs.push_back(input);
		return input;
	}

	SphericalTriangleTestResults determineExpectedResults(const SphericalTriangleInputValues& input) override
	{
		SphericalTriangleTestResults expected;
		SphericalTriangleTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const SphericalTriangleTestResults& expected, const SphericalTriangleTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		// GPU vs CPU: CPU trig may use double precision internally, allow larger tolerance.
		// Near domain boundaries (u.x~0), sin/cos precision differences get amplified
		// through Arvo's sub-area interpolation, producing up to ~0.002 absolute error.
		// For small triangles (solidAngle~0.05), GPU/CPU acos precision differences
		// cause ~1.1e-4 relative error in rcpSolidAngle which is both forwardPdf and backwardPdf.
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"SphericalTriangle::generate",       &R::generated,      5e-2, 2e-3},
			FieldCheck{"SphericalTriangle::forwardPdf",     &R::forwardPdf,     2e-4, 1e-4},
			FieldCheck{"SphericalTriangle::backwardPdf",    &R::backwardPdf,    2e-4, 1e-4},
			FieldCheck{"SphericalTriangle::forwardWeight",  &R::forwardWeight,  2e-4, 1e-4},
			FieldCheck{"SphericalTriangle::backwardWeight", &R::backwardWeight, 2e-4, 1e-4},
			FieldCheck{"SphericalTriangle::inverted",       &R::inverted,       1e-4, 5e-3});
		pass &= verifyTestValue("SphericalTriangle::roundtripError", nbl::hlsl::float32_t2(0.0f, 0.0f), actual.roundtripError, iteration, seed, testType, 1e-4, 5e-3);
		// TODO: we're not chasing this further but we have sinZ ~= sqrt(u.y) parameterization in the
		// Arvo ST sampler, so O(h) forward diff has O(h/u.y) bias that no fixed eps can fully resolve.
		VERIFY_JACOBIAN_OR_SKIP(pass, "SphericalTriangle::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 2.0, 2.0);
		VERIFY_JACOBIAN_OR_SKIP(pass, "SphericalTriangle::inverseJacobianPdf", actual.backwardPdf, actual.inverseJacobianPdf, iteration, seed, testType, 3.0, 3.0);
		pass &= verifyTestValue("SphericalTriangle::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("SphericalTriangle::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"SphericalTriangle::forwardPdf",  &R::forwardPdf},
			PdfCheck{"SphericalTriangle::backwardPdf", &R::backwardPdf});

		// Domain preservation: samples must not escape the domain.
		constexpr float64_t domainTolerance = 1e-6;
		if (actual.generatedInside < -domainTolerance)
		{
			pass = false;
			printTestFail("SphericalTriangle::generatedInside", 0.0f, actual.generatedInside, iteration, seed, testType, 0.0, domainTolerance);
		}
		if (actual.invertedInDomain < -domainTolerance)
		{
			pass = false;
			printTestFail("SphericalTriangle::invertedInDomain", 0.0f, actual.invertedInDomain, iteration, seed, testType, 0.0, domainTolerance);
		}

		if (!pass && iteration < m_inputs.size())
			logFailedInput(m_logger.get(), m_inputs[iteration]);

		return pass;
	}

	core::vector<SphericalTriangleInputValues> m_inputs;
};

// --- Property test config ---
struct SphericalTrianglePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::SphericalTriangle<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 500;
	static constexpr uint32_t samplesPerConfig = 20000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.05;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "SphericalTriangle"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		nbl::hlsl::float32_t3 v0, v1, v2;
		generateRandomTriangleVertices(rng, v0, v1, v2);
		return sampler_type::create(createSphericalTriangleShape(v0, v1, v2));
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	// E[1/pdf] = E[solid_angle] = solid_angle (constant pdf sampler)
	static float64_t expectedCodomainMeasure(const sampler_type& s)
	{
		return 1.0 / static_cast<float64_t>(s.rcpSolidAngle);
	}

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		logTriangleInfo(logger, s.tri_vertices[0], s.tri_vertices[1], s.APlusC - s.tri_vertices[0]);
	}
};

// Stress-test generateInverse with ill-conditioned triangle geometries:
// thin/elongated triangles, nearly coplanar vertices, one very short edge.
// These stress the C_s great-circle intersection and v-recovery in generateInverse.
struct SphericalTriangleStressConfig
{
	using sampler_type = nbl::hlsl::sampling::SphericalTriangle<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 500;
	static constexpr uint32_t samplesPerConfig = 20000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.05;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "SphericalTriangle(stress)"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		using namespace nbl::hlsl;
		std::uniform_int_distribution<int> caseDist(0, 2);
		std::uniform_real_distribution<float> angleDist(0.0f, 1.0f);

		float32_t3 v0, v1, v2;
		switch (caseDist(rng))
		{
			case 0:
			{
				// Thin/elongated: two vertices separated by ~20 degrees, third ~90 degrees away
				// Stresses C_s great-circle intersection (nearly parallel great circles)
				float32_t3 base = generateRandomUnitVector(rng);
				float32_t3 t1, t2;
				buildTangentFrame(base, t1, t2);

				// Two vertices close together (~20 deg apart)
				float spreadAngle = 0.15f + angleDist(rng) * 0.2f; // 0.15-0.35 rad
				v0 = normalize(base + t1 * spreadAngle);
				v1 = normalize(base - t1 * spreadAngle);
				// Third vertex far away
				float farAngle = 0.8f + angleDist(rng) * 0.8f; // 0.8-1.6 rad
				v2 = normalize(base * std::cos(farAngle) + t2 * std::sin(farAngle));
				break;
			}
			case 1:
			{
				// Nearly coplanar: all three vertices close to a great circle
				// Small solid angle, precision-sensitive solid angle computation
				float32_t3 pole = generateRandomUnitVector(rng);
				float32_t3 t1, t2;
				buildTangentFrame(pole, t1, t2);

				// Three points on a great circle, perturbed slightly off it
				float offset = 0.05f + angleDist(rng) * 0.1f; // small off-plane component
				float a1 = angleDist(rng) * 2.0f * 3.14159f;
				float a2 = a1 + 0.8f + angleDist(rng) * 1.0f;
				float a3 = a2 + 0.8f + angleDist(rng) * 1.0f;
				v0 = normalize(t1 * std::cos(a1) + t2 * std::sin(a1) + pole * offset);
				v1 = normalize(t1 * std::cos(a2) + t2 * std::sin(a2) - pole * offset * 0.5f);
				v2 = normalize(t1 * std::cos(a3) + t2 * std::sin(a3) + pole * offset * 0.3f);
				break;
			}
			default:
			{
				// One short edge: v0 and v1 separated by ~18-25 degrees,
				// v2 well separated. Short arc B->C_s stresses v-recovery.
				float32_t3 base = generateRandomUnitVector(rng);
				float32_t3 t1, t2;
				buildTangentFrame(base, t1, t2);

				float shortAngle = 0.32f + angleDist(rng) * 0.1f; // ~18-24 deg
				v0 = normalize(base + t1 * shortAngle * 0.5f);
				v1 = normalize(base - t1 * shortAngle * 0.5f);
				// v2 well separated
				v2 = normalize(t2 + base * (0.3f + angleDist(rng) * 0.5f));
				break;
			}
		}

		// Validate — if degenerate (shouldn't be common), fall back to random valid triangle
		if (!isValidSphericalTriangle(v0, v1, v2))
			generateRandomTriangleVertices(rng, v0, v1, v2);

		return sampler_type::create(createSphericalTriangleShape(v0, v1, v2));
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	static float64_t expectedCodomainMeasure(const sampler_type& s)
	{
		return 1.0 / static_cast<float64_t>(s.rcpSolidAngle);
	}

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		logTriangleInfo(logger, s.tri_vertices[0], s.tri_vertices[1], s.APlusC - s.tri_vertices[0]);
	}
};

#endif
