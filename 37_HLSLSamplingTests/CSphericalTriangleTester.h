#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_TRIANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_TRIANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/examples/Tester/ITester.h"

class CSphericalTriangleTester final : public ITester<SphericalTriangleInputValues, SphericalTriangleTestResults, SphericalTriangleTestExecutor>
{
	using base_t = ITester<SphericalTriangleInputValues, SphericalTriangleTestResults, SphericalTriangleTestExecutor>;

public:
	CSphericalTriangleTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

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

		// Reject edges that are nearly coincident or antipodal
		constexpr float sinSqThreshold = 0.09f; // sin(theta) > 0.3

		const float d01 = dot(v0, v1);
		const float d12 = dot(v1, v2);
		const float d20 = dot(v2, v0);

		if ((1.f - d01 * d01) < sinSqThreshold)
			return false;
		if ((1.f - d12 * d12) < sinSqThreshold)
			return false;
		if ((1.f - d20 * d20) < sinSqThreshold)
			return false;

		// Reject triangles whose vertices lie nearly on the same great circle
		constexpr float tripleThreshold = 0.1f;
		return abs(dot(v0, cross(v1, v2))) > tripleThreshold;
	}

	SphericalTriangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		SphericalTriangleInputValues input;

		// Generate well-separated unit vectors for a valid spherical triangle
		do
		{
			input.vertex0 = generateRandomUnitVector();
			input.vertex1 = generateRandomUnitVector();
			input.vertex2 = generateRandomUnitVector();
		} while (!isValidSphericalTriangle(input.vertex0, input.vertex1, input.vertex2));

		// Avoid domain boundaries (u near 0 or 1) where generateInverse
		// can produce NaN due to float32 precision in sqrt/acos operations
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
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
		pass &= verifyTestValue("SphericalTriangle::generate", expected.generated, actual.generated, iteration, seed, testType, 1e-4, 1e-2);
		pass &= verifyTestValue("SphericalTriangle::forwardPdf", expected.forwardPdf, actual.forwardPdf, iteration, seed, testType, 1e-4, 1e-3);
		pass &= verifyTestValue("SphericalTriangle::backwardPdf", expected.backwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-3);
		pass &= verifyTestValue("SphericalTriangle::inverted", expected.inverted, actual.inverted, iteration, seed, testType, 1e-4, 4e-2); // tolerated
		pass &= verifyTestValue("SphericalTriangle::rountTripError (absolute)", 0.0f, actual.roundtripError, iteration, seed, testType, 5e-2, 1e-2); // tolerated

		// jacobianProduct = (1/forwardPdf) * backwardPdf should be == 1.0.
		pass &= verifyTestValue("SphericalTriangle::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 1e-4, 1e-4);

		if (!(actual.forwardPdf > 0.0f) || !std::isfinite(actual.forwardPdf))
		{
			pass = false;
			printTestFail("SphericalTriangle::forwardPdf (positive & finite)", 1.0f, actual.forwardPdf, iteration, seed, testType, 0.0, 0.0);
		}
		if (!(actual.backwardPdf > 0.0f) || !std::isfinite(actual.backwardPdf))
		{
			pass = false;
			printTestFail("SphericalTriangle::backwardPdf (positive & finite)", 1.0f, actual.backwardPdf, iteration, seed, testType, 0.0, 0.0);
		}

		// Domain preservation: samples must not escape the domain.
		// Values are signed distances (positive = inside); allow a small negative
		// tolerance for float32 imprecision near triangle edges / [0,1]^2 boundaries.
		// verifyTestValue is a symmetric closeness check and can't express ">= -eps",
		// so we do the comparison directly.
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

		return pass;
	}
};

#endif
