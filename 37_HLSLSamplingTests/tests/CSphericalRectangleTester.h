#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_RECTANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SPHERICAL_RECTANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/spherical_rectangle.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

class CSphericalRectangleTester final : public ITester<SphericalRectangleInputValues, SphericalRectangleTestResults, SphericalRectangleTestExecutor>
{
	using base_t = ITester<SphericalRectangleInputValues, SphericalRectangleTestResults, SphericalRectangleTestExecutor>;
	using R = SphericalRectangleTestResults;

public:
	CSphericalRectangleTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	SphericalRectangleInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t> compressed;
		nbl::hlsl::float32_t3 observer;
		generateRandomRectangle(getRandomEngine(), compressed, observer);

		SphericalRectangleInputValues input;
		input.observer = observer;
		input.rectOrigin = compressed.origin;
		input.right = compressed.right;
		input.up = compressed.up;
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		m_inputs.push_back(input);
		return input;
	}

	SphericalRectangleTestResults determineExpectedResults(const SphericalRectangleInputValues& input) override
	{
		SphericalRectangleTestResults expected;
		SphericalRectangleTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const SphericalRectangleTestResults& expected, const SphericalRectangleTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		// Tolerances reflect GPU-vs-CPU fp32 divergence on an identical algorithm: `solidAngle` is
		// built from basis dot products, 4 rsqrts, and one acos; GPU fuses these into FMA chains
		// while CPU doesn't, so small-angle cases (large 1/solidAngle) drift by a few ulps on the
		// divisor, amplified in the reciprocal.
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"SphericalRectangle::generate",              &R::generated,      5e-4, 2e-2},
			FieldCheck{"SphericalRectangle::generateSurfaceOffset", &R::surfaceOffset,  5e-4, 2e-2},
			FieldCheck{"SphericalRectangle::forwardPdf",            &R::forwardPdf,     2e-3, 1e-1},
			FieldCheck{"SphericalRectangle::backwardPdf",           &R::backwardPdf,    2e-3, 1e-1},
			FieldCheck{"SphericalRectangle::forwardWeight",         &R::forwardWeight,  2e-3, 1e-1},
			FieldCheck{"SphericalRectangle::backwardWeight",        &R::backwardWeight, 2e-3, 1e-1});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"SphericalRectangle::forwardPdf",  &R::forwardPdf},
			PdfCheck{"SphericalRectangle::backwardPdf", &R::backwardPdf});
		VERIFY_JACOBIAN_OR_SKIP(pass, "SphericalRectangle::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 4e-2, 4e-2);
		pass &= verifyTestValue("SphericalRectangle::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-7, 1e-7);
		pass &= verifyTestValue("SphericalRectangle::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-7, 1e-7);

		// surfaceOffset must land inside the rectangle
		if (actual.surfaceOffset.x < 0.0f || actual.surfaceOffset.x > actual.extents.x ||
			actual.surfaceOffset.y < 0.0f || actual.surfaceOffset.y > actual.extents.y)
		{
			pass = false;
			printTestFail("SphericalRectangle::generateSurfaceOffset (inside rect bounds)", actual.extents, actual.surfaceOffset, iteration, seed, testType, 0.0, 0.0);
		}

		// generate must be unit length
		{
			const float dirLen = nbl::hlsl::length(actual.generated);
			pass &= verifyTestValue("SphericalRectangle::generate (unit length)", dirLen, 1.0f, iteration, seed, testType, 1e-5, 1e-4);
		}

		// generate must agree with generateSurfaceOffset (reference direction from normalized local point)
		pass &= verifyTestValue("SphericalRectangle::generate vs generateSurfaceOffset", actual.generated, actual.referenceDirection, iteration, seed, testType, 5e-5, 5e-3);

		if (!pass && iteration < m_inputs.size())
			logFailedInput(m_logger.get(), m_inputs[iteration]);

		return pass;
	}

	core::vector<SphericalRectangleInputValues> m_inputs;
};

// --- Property test config ---
struct SphericalRectanglePropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::SphericalRectangle<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 1000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.05;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "SphericalRectangle"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> sizeDist(0.5f, 3.0f);
		const float width = sizeDist(rng);
		const float height = sizeDist(rng);

		nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t> compressed;
		compressed.origin = nbl::hlsl::float32_t3(0.0f, 0.0f, -2.0f);
		compressed.right = nbl::hlsl::float32_t3(width, 0.0f, 0.0f);
		compressed.up = nbl::hlsl::float32_t3(0.0f, height, 0.0f);

		nbl::hlsl::shapes::SphericalRectangle<nbl::hlsl::float32_t> rect = nbl::hlsl::shapes::SphericalRectangle<nbl::hlsl::float32_t>::create(compressed);
		nbl::hlsl::float32_t3 observer(0.0f, 0.0f, 0.0f);
		return sampler_type::create(rect, observer);
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	// E[1/pdf] = solid_angle (constant pdf sampler)
	static float64_t expectedCodomainMeasure(const sampler_type& s)
	{
		return static_cast<float64_t>(s.solidAngle);
	}

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    solidAngle=%s k=%s", nbl::system::ILogger::ELL_ERROR,
			to_string(s.solidAngle).c_str(), to_string(s.k).c_str());
	}
};

#endif
