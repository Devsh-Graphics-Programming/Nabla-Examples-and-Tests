#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LINEAR_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_LINEAR_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/linear.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/linear.hlsl>

class CLinearTester final : public ITester<LinearInputValues, LinearTestResults, LinearTestExecutor>
{
	using base_t = ITester<LinearInputValues, LinearTestResults, LinearTestExecutor>;
	using R = LinearTestResults;

public:
	CLinearTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

private:
	LinearInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> coeffDist(0.1f, 5.0f);
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		LinearInputValues input;
		input.coeffs = nbl::hlsl::float32_t2(coeffDist(getRandomEngine()), coeffDist(getRandomEngine()));
		input.u = uDist(getRandomEngine());
		m_inputs.push_back(input);
		return input;
	}

	LinearTestResults determineExpectedResults(const LinearInputValues& input) override
	{
		LinearTestResults expected;
		LinearTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const LinearTestResults& expected, const LinearTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"Linear::generate",        &R::generated,       1e-4, 5e-5},
			FieldCheck{"Linear::forwardPdf",       &R::forwardPdf,      1e-4, 1e-5},
			FieldCheck{"Linear::backwardPdf",      &R::backwardPdf,     1e-4, 1e-5},
			FieldCheck{"Linear::forwardWeight",  &R::forwardWeight,  1e-4, 1e-5},
			FieldCheck{"Linear::backwardWeight", &R::backwardWeight, 1e-4, 1e-5});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"Linear::forwardPdf",  &R::forwardPdf},
			PdfCheck{"Linear::backwardPdf", &R::backwardPdf});
		VERIFY_JACOBIAN_OR_SKIP(pass, "Linear::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 6e-2, 6e-2);
		pass &= verifyTestValue("Linear::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("Linear::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-5);

		if (!pass && iteration < m_inputs.size())
			logFailedInput(m_logger.get(), m_inputs[iteration]);

		return pass;
	}

	core::vector<LinearInputValues> m_inputs;
};

// --- Property test configs ---
struct LinearPropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::Linear<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 50000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = true;
	static constexpr float64_t mcNormalizationRelTol = 0.03;
	static constexpr float64_t gridNormalizationAbsTol = 1e-4;

	static const char* name() { return "Linear"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> d(0.1f, 5.0f);
		return sampler_type::create(nbl::hlsl::float32_t2(d(rng), d(rng)));
	}

	static nbl::hlsl::float32_t randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t>(rng); }
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 1.0; }
	static float64_t gridIntegratePdf(const sampler_type& s) { return gridIntegratePdf1D(s); }
	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    coeffStart=%s coeffEnd=%s", nbl::system::ILogger::ELL_ERROR,
			to_string(s.normalizedCoeffStart).c_str(), to_string(s.normalizedCoeffEnd).c_str());
	}
};

struct LinearStressConfig
{
	using sampler_type = nbl::hlsl::sampling::Linear<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = true;
	// High variance in 1/pdf for extreme ratio configs; grid integration is the
	// authoritative normalization check (passes). MC serves as a secondary signal.
	static constexpr float64_t mcNormalizationRelTol = 0.1;
	static constexpr float64_t gridNormalizationAbsTol = 1e-3;

	static const char* name() { return "Linear(stress)"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_int_distribution<int> caseDist(0, 3);
		// Ratio after normalization determines the min PDF and thus MC variance.
		// Keeping max ratio ~ 10000:1 ensures E[(1/pdf)^2] stays bounded enough
		// for 100k samples to converge within tolerance.
		std::uniform_real_distribution<float> smallDist(1e-2f, 1e-1f);
		std::uniform_real_distribution<float> largeDist(10.0f, 100.0f);
		std::uniform_real_distribution<float> normalDist(0.1f, 5.0f);

		switch (caseDist(rng))
		{
			case 0: // one tiny, one large
				return sampler_type::create(nbl::hlsl::float32_t2(smallDist(rng), largeDist(rng)));
			case 1: // both tiny
				return sampler_type::create(nbl::hlsl::float32_t2(smallDist(rng), smallDist(rng)));
			case 2: // both large
				return sampler_type::create(nbl::hlsl::float32_t2(largeDist(rng), largeDist(rng)));
			default: // nearly equal (PDF ~constant)
				{
					float v = normalDist(rng);
					return sampler_type::create(nbl::hlsl::float32_t2(v, v + smallDist(rng)));
				}
		}
	}

	static nbl::hlsl::float32_t randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t>(rng); }
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 1.0; }
	static float64_t gridIntegratePdf(const sampler_type& s) { return gridIntegratePdf1D(s); }
	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    coeffStart=%s coeffEnd=%s", nbl::system::ILogger::ELL_ERROR,
			to_string(s.normalizedCoeffStart).c_str(), to_string(s.normalizedCoeffEnd).c_str());
	}
};

#endif
