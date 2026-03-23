#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_TRIANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_TRIANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CProjectedSphericalTriangleTester final : public ITester<ProjectedSphericalTriangleInputValues, ProjectedSphericalTriangleTestResults, ProjectedSphericalTriangleTestExecutor>
{
	using base_t = ITester<ProjectedSphericalTriangleInputValues, ProjectedSphericalTriangleTestResults, ProjectedSphericalTriangleTestExecutor>;

public:
	CProjectedSphericalTriangleTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	nbl::hlsl::float32_t3 generateRandomUnitVector()
	{
		std::uniform_real_distribution<float> dist(0.0f, 1.0f);
		nbl::hlsl::float32_t2 u(dist(getRandomEngine()), dist(getRandomEngine()));
		nbl::hlsl::sampling::UniformSphere<float>::cache_type cache;
		return nbl::hlsl::sampling::UniformSphere<float>::generate(u, cache);
	}

	static bool isValidSphericalTriangle(nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
	{
		using namespace nbl::hlsl;
		constexpr float sinSqThreshold = 0.09f; // sin(theta) > 0.3
		const float d01 = dot(v0, v1);
		const float d12 = dot(v1, v2);
		const float d20 = dot(v2, v0);
		if ((1.f - d01 * d01) < sinSqThreshold) return false;
		if ((1.f - d12 * d12) < sinSqThreshold) return false;
		if ((1.f - d20 * d20) < sinSqThreshold) return false;
		constexpr float tripleThreshold = 0.1f;
		return abs(dot(v0, cross(v1, v2))) > tripleThreshold;
	}

	ProjectedSphericalTriangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		ProjectedSphericalTriangleInputValues input;

		do
		{
			input.vertex0 = generateRandomUnitVector();
			input.vertex1 = generateRandomUnitVector();
			input.vertex2 = generateRandomUnitVector();
		} while (!isValidSphericalTriangle(input.vertex0, input.vertex1, input.vertex2));

		// Ensure the receiver normal has positive projection onto at least one vertex,
		// otherwise the projected solid angle is zero and the bilinear patch is degenerate (NaN PDFs).
		do
		{
			input.receiverNormal = generateRandomUnitVector();
		} while (nbl::hlsl::dot(input.receiverNormal, input.vertex0) <= 0.0f &&
				 nbl::hlsl::dot(input.receiverNormal, input.vertex1) <= 0.0f &&
				 nbl::hlsl::dot(input.receiverNormal, input.vertex2) <= 0.0f);
		input.receiverWasBSDF = 0u;
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
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
		// SphericalTriangle::generate has a known precision issue (see TODO in spherical_triangle.hlsl)
		// due to catastrophic cancellation in the cosAngleAlongAC formula; CPU/GPU rounding diverges
		// by up to ~0.002 in direction components for certain triangle geometries
		pass &= verifyTestValue("ProjectedSphericalTriangle::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-4, 3e-3);
		pass &= verifyTestValue("ProjectedSphericalTriangle::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-4, 1e-3);
		pass &= verifyTestValue("ProjectedSphericalTriangle::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-3);

		// PDF positivity and finiteness
		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("ProjectedSphericalTriangle::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("ProjectedSphericalTriangle::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		return pass;
	}
};

#endif
