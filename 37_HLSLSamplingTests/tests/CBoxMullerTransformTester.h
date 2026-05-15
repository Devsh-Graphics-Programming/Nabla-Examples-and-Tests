#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BOX_MULLER_TRANSFORM_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BOX_MULLER_TRANSFORM_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/box_muller_transform.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/box_muller_transform.hlsl>

class CBoxMullerTransformTester final : public ITester<BoxMullerTransformInputValues, BoxMullerTransformTestResults, BoxMullerTransformTestExecutor>
{
	using base_t = ITester<BoxMullerTransformInputValues, BoxMullerTransformTestResults, BoxMullerTransformTestExecutor>;
	using R = BoxMullerTransformTestResults;

public:
	CBoxMullerTransformTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	BoxMullerTransformInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> stddevDist(0.1f, 5.0f);
		// Avoid u.x near 0 to prevent log(0) = -inf
		std::uniform_real_distribution<float> uDist(1e-4f, 1.0f - 1e-4f);

		BoxMullerTransformInputValues input;
		input.stddev = stddevDist(getRandomEngine());
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		return input;
	}

	BoxMullerTransformTestResults determineExpectedResults(const BoxMullerTransformInputValues& input) override
	{
		BoxMullerTransformTestResults expected;
		BoxMullerTransformTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const BoxMullerTransformTestResults& expected, const BoxMullerTransformTestResults& actual,
		const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"BoxMullerTransform::generate",           &R::generated,          1e-5, 2e-3},
			FieldCheck{"BoxMullerTransform::cache.pdf",          &R::cachedPdf,          1e-5, 1e-3},
			FieldCheck{"BoxMullerTransform::forwardPdf",         &R::forwardPdf,         1e-5, 1e-3},
			FieldCheck{"BoxMullerTransform::backwardPdf",        &R::backwardPdf,        1e-5, 1e-3},
			FieldCheck{"BoxMullerTransform::separateBackwardPdf", &R::separateBackwardPdf, 1e-5, 1e-3},
			FieldCheck{"BoxMullerTransform::forwardWeight",  &R::forwardWeight,  1e-5, 1e-3},
			FieldCheck{"BoxMullerTransform::backwardWeight", &R::backwardWeight, 1e-5, 1e-3});
		// Joint PDF == product of marginal PDFs (independent random variables)
		pass &= verifyTestValue("BoxMullerTransform::jointPdf == pdf product", actual.backwardPdf, actual.separateBackwardPdf.x * actual.separateBackwardPdf.y, iteration, seed, testType, 1e-5, 1e-5);
		// forwardPdf must return the same value stored in cache.pdf by generate
		pass &= verifyTestValue("BoxMullerTransform::forwardPdf == cache.pdf", actual.forwardPdf, actual.cachedPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("BoxMullerTransform::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-4, 1e-3);
		pass &= verifyTestValue("BoxMullerTransform::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-4, 1e-3);
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"BoxMullerTransform::forwardPdf",  &R::forwardPdf},
			PdfCheck{"BoxMullerTransform::backwardPdf", &R::backwardPdf});
		return pass;
	}
};

// --- Property test config ---
struct BoxMullerTransformPropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::BoxMullerTransform<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 50;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = false; // codomain is unbounded R^2, E[1/pdf] diverges
	static constexpr bool hasGridIntegration = false;
	static constexpr float64_t mcNormalizationRelTol = 0.0;
	static constexpr float64_t gridNormalizationAbsTol = 0.0;

	static const char* name() { return "BoxMullerTransform"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> d(0.1f, 5.0f);
		return sampler_type::create(d(rng));
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    stddev=%s", nbl::system::ILogger::ELL_ERROR, to_string(s.stddev).c_str());
	}
};

#endif
