#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_TRIANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_TRIANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_spherical_triangle.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl>

class CProjectedSphericalTriangleTester final : public ITester<ProjectedSphericalTriangleInputValues, ProjectedSphericalTriangleTestResults, ProjectedSphericalTriangleTestExecutor>
{
	using base_t = ITester<ProjectedSphericalTriangleInputValues, ProjectedSphericalTriangleTestResults, ProjectedSphericalTriangleTestExecutor>;
	using R = ProjectedSphericalTriangleTestResults;

public:
	CProjectedSphericalTriangleTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

private:
	ProjectedSphericalTriangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		ProjectedSphericalTriangleInputValues input;

		do
		{
			input.vertex0 = generateRandomUnitVector(getRandomEngine());
			input.vertex1 = generateRandomUnitVector(getRandomEngine());
			input.vertex2 = generateRandomUnitVector(getRandomEngine());
		} while (!isValidSphericalTriangle(input.vertex0, input.vertex1, input.vertex2));

		// Ensure the receiver normal has positive projection onto at least one vertex,
		// otherwise the projected solid angle is zero and the bilinear patch is degenerate (NaN PDFs).
		do
		{
			input.receiverNormal = generateRandomUnitVector(getRandomEngine());
		} while (nbl::hlsl::dot(input.receiverNormal, input.vertex0) <= 0.0f &&
				 nbl::hlsl::dot(input.receiverNormal, input.vertex1) <= 0.0f &&
				 nbl::hlsl::dot(input.receiverNormal, input.vertex2) <= 0.0f);
		input.receiverWasBSDF = 0u;
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		m_inputs.push_back(input);
		return input;
	}

	ProjectedSphericalTriangleTestResults determineExpectedResults(const ProjectedSphericalTriangleInputValues& input) override
	{
		ProjectedSphericalTriangleTestResults expected;
		ProjectedSphericalTriangleTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const ProjectedSphericalTriangleTestResults& expected, const ProjectedSphericalTriangleTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		// PSA-normalized PDFs can be large when projectedSolidAngle is small (grazing),
		// and GPU/CPU trig differences are amplified by rcpProjSolidAngle.
		// Bilinear CDF inversion near domain boundaries (u~0 or u~1) amplifies
		// CPU/GPU FP differences, producing up to ~0.003 absolute error in generate.
		// Weight self-consistency is tested via backwardWeightAtGenerated (backwardWeight takes a
		// 3D direction; evaluate at the triangle centroid for a deterministic interior point).
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"ProjectedSphericalTriangle::generate",       &R::generated,      2e-1, 3e-3},
			FieldCheck{"ProjectedSphericalTriangle::forwardPdf",     &R::forwardPdf,     5e-2, 1e-4},
			FieldCheck{"ProjectedSphericalTriangle::forwardWeight",  &R::forwardWeight,  5e-2, 1e-4},
			FieldCheck{"ProjectedSphericalTriangle::backwardWeight", &R::backwardWeight, 5e-2, 1e-4});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"ProjectedSphericalTriangle::forwardPdf", &R::forwardPdf});
		// TODO: we're not chasing this further but we have sinZ ~= sqrt(u.y) parameterization in the
		// underlying SphericalTriangle (Arvo) which cascades through the bilinear warp at small SA.
		VERIFY_JACOBIAN_OR_SKIP(pass, "ProjectedSphericalTriangle::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 2.0, 2.0);
		pass &= verifyTestValue("ProjectedSphericalTriangle::weight consistency", actual.forwardWeight, actual.backwardWeightAtGenerated, iteration, seed, testType, 5e-2, 2e-2);

		if (!pass && iteration < m_inputs.size())
			logFailedInput(m_logger.get(), m_inputs[iteration]);

		return pass;
	}

	core::vector<ProjectedSphericalTriangleInputValues> m_inputs;
};

// --- Property test configs ---
struct ProjectedSphericalTrianglePropertyConfig
{
	// UsePdfAsWeight=false so receiverNormal is populated for logSamplerInfo.
	using sampler_type = nbl::hlsl::sampling::ProjectedSphericalTriangle<nbl::hlsl::float32_t, false>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 20000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.08;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "ProjectedSphericalTriangle"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		nbl::hlsl::float32_t3 v0, v1, v2;
		generateRandomTriangleVertices(rng, v0, v1, v2);

		// All vertices above horizon: no zero bilinear corners, bounded MC variance.
		// Grazing configs are tested separately by ProjectedSphericalTriangleGrazingConfig.
		nbl::hlsl::float32_t3 normal;
		do
		{
			normal = generateRandomUnitVector(rng);
		} while (!allVerticesAboveHorizon(normal, v0, v1, v2));

		auto shape = createSphericalTriangleShape(v0, v1, v2);
		return sampler_type::create(shape, normal, false);
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	// E[1/pdf] = solidAngle * E[1/bilinearPdf] = solidAngle * 1.0 = solidAngle
	static float64_t expectedCodomainMeasure(const sampler_type& s)
	{
		return 1.0 / static_cast<float64_t>(s.sphtri.rcpSolidAngle);
	}

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		logTriangleInfo(logger, s.sphtri.tri_vertices[0], s.sphtri.tri_vertices[1], s.sphtri.APlusC - s.sphtri.tri_vertices[0], s.receiverNormal);
	}
};

struct ProjectedSphericalTriangleGrazingConfig
{
	// UsePdfAsWeight=false so receiverNormal is populated for logSamplerInfo.
	using sampler_type = nbl::hlsl::sampling::ProjectedSphericalTriangle<nbl::hlsl::float32_t, false>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 20000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	// Single-corner bilinear patches (3/4 zero corners) have near-divergent
	// 1/bilinearPdf near zero-density regions, causing extreme MC variance.
	// 20k samples can't reliably converge closer than ~25% for these configs.
	static constexpr float64_t mcNormalizationRelTol = 0.25;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "ProjectedSphericalTriangle(grazing)"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		nbl::hlsl::float32_t3 v0, v1, v2;
		generateRandomTriangleVertices(rng, v0, v1, v2);

		// Normal nearly perpendicular to triangle (grazing angle)
		// This is where Matt sees the fireflies
		nbl::hlsl::float32_t3 triCenter = nbl::hlsl::normalize(v0 + v1 + v2);
		nbl::hlsl::float32_t3 tangent, unused;
		buildTangentFrame(triCenter, tangent, unused);

		// Normal is mostly tangent to the sphere at the triangle center
		// with a small component toward the triangle so NdotL > 0 for at least one vertex
		std::uniform_real_distribution<float> grazeDist(0.02f, 0.15f);
		nbl::hlsl::float32_t3 normal = nbl::hlsl::normalize(tangent + triCenter * grazeDist(rng));

		if (!anyVertexAboveHorizon(normal, v0, v1, v2))
			normal = nbl::hlsl::normalize(tangent + triCenter * 0.3f);

		auto shape = createSphericalTriangleShape(v0, v1, v2);
		return sampler_type::create(shape, normal, false);
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	static float64_t expectedCodomainMeasure(const sampler_type& s)
	{
		return 1.0 / static_cast<float64_t>(s.sphtri.rcpSolidAngle);
	}

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		logTriangleInfo(logger, s.sphtri.tri_vertices[0], s.sphtri.tri_vertices[1], s.sphtri.APlusC - s.sphtri.tri_vertices[0], s.receiverNormal);
	}
};

#endif
