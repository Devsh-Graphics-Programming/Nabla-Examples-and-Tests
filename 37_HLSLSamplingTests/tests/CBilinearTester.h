#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BILINEAR_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_BILINEAR_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/bilinear.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>

class CBilinearTester final : public ITester<BilinearInputValues, BilinearTestResults, BilinearTestExecutor>
{
	using base_t = ITester<BilinearInputValues, BilinearTestResults, BilinearTestExecutor>;
	using R = BilinearTestResults;

public:
	CBilinearTester(const uint32_t testBatchCount, const uint32_t workgroupSize) : base_t(testBatchCount, workgroupSize) {}

private:
	BilinearInputValues generateInputTestValues() override
	{
		std::uniform_real_distribution<float> coeffDist(0.1f, 5.0f);
		std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

		BilinearInputValues input;
		input.bilinearCoeffs = nbl::hlsl::float32_t4(
			coeffDist(getRandomEngine()), coeffDist(getRandomEngine()),
			coeffDist(getRandomEngine()), coeffDist(getRandomEngine()));
		input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
		m_inputs.push_back(input);
		return input;
	}

	BilinearTestResults determineExpectedResults(const BilinearInputValues& input) override
	{
		BilinearTestResults expected;
		BilinearTestExecutor executor;
		executor(input, expected);
		return expected;
	}

	bool verifyTestResults(const BilinearTestResults& expected, const BilinearTestResults& actual, const size_t iteration, const uint32_t seed, TestType testType) override
	{
		bool pass = true;
		VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
			FieldCheck{"Bilinear::generate",       &R::generated,      1e-4, 1e-4},
			FieldCheck{"Bilinear::pdf",            &R::backwardPdf,    1e-5, 1e-4},
			FieldCheck{"Bilinear::forwardPdf",     &R::forwardPdf,     1e-5, 1e-4},
			FieldCheck{"Bilinear::forwardWeight",  &R::forwardWeight,  1e-5, 1e-4},
			FieldCheck{"Bilinear::backwardWeight", &R::backwardWeight, 1e-5, 1e-4});
		VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
			PdfCheck{"Bilinear::forwardPdf",  &R::forwardPdf},
			PdfCheck{"Bilinear::backwardPdf", &R::backwardPdf});
		VERIFY_JACOBIAN_OR_SKIP(pass, "Bilinear::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 5e-2, 5e-2);
		pass &= verifyTestValue("Bilinear::pdf consistency", actual.forwardPdf, actual.backwardPdf, iteration, seed, testType, 1e-5, 1e-5);
		pass &= verifyTestValue("Bilinear::weight consistency", actual.forwardWeight, actual.backwardWeight, iteration, seed, testType, 1e-5, 1e-5);

		if (!pass && iteration < m_inputs.size())
			logFailedInput(m_logger.get(), m_inputs[iteration]);

		return pass;
	}

	core::vector<BilinearInputValues> m_inputs;
};

// --- Property test configs ---
struct BilinearPropertyConfig
{
	using sampler_type = nbl::hlsl::sampling::Bilinear<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 50000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = true;
	static constexpr float64_t mcNormalizationRelTol = 0.02;
	static constexpr float64_t gridNormalizationAbsTol = 1e-3;

	static const char* name() { return "Bilinear"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_real_distribution<float> d(0.1f, 5.0f);
		return sampler_type::create(nbl::hlsl::float32_t4(d(rng), d(rng), d(rng), d(rng)));
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 1.0; }
	static float64_t gridIntegratePdf(const sampler_type& s) { return gridIntegratePdf2D(s); }
	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    yStarts=%s yDiffs=%s normFactor=%s",
			nbl::system::ILogger::ELL_ERROR, to_string(s.yStarts).c_str(), to_string(s.yDiffs).c_str(), to_string(s.normFactor).c_str());
	}
};

struct BilinearStressConfig
{
	using sampler_type = nbl::hlsl::sampling::Bilinear<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 200;
	static constexpr uint32_t samplesPerConfig = 100000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = true;
	// Same as Linear stress: grid integration is authoritative
	static constexpr float64_t mcNormalizationRelTol = 0.1;
	static constexpr float64_t gridNormalizationAbsTol = 1e-3;

	static const char* name() { return "Bilinear(stress)"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_int_distribution<int> caseDist(0, 4);
		std::uniform_real_distribution<float> smallDist(1e-4f, 1e-2f);
		std::uniform_real_distribution<float> largeDist(10.0f, 100.0f);
		std::uniform_real_distribution<float> normalDist(0.1f, 5.0f);

		switch (caseDist(rng))
		{
			case 0: // one corner dominates (simulates grazing angle)
				return sampler_type::create(nbl::hlsl::float32_t4(largeDist(rng), smallDist(rng), smallDist(rng), smallDist(rng)));
			case 1: // two adjacent corners large, two small (edge-weighted)
				return sampler_type::create(nbl::hlsl::float32_t4(largeDist(rng), largeDist(rng), smallDist(rng), smallDist(rng)));
			case 2: // diagonal dominance
				return sampler_type::create(nbl::hlsl::float32_t4(largeDist(rng), smallDist(rng), smallDist(rng), largeDist(rng)));
			case 3: // nearly uniform (all similar)
				{
					float v = normalDist(rng);
					return sampler_type::create(nbl::hlsl::float32_t4(v, v + smallDist(rng), v + smallDist(rng), v + smallDist(rng)));
				}
			default: // one corner zero-ish (degenerate horizon)
				return sampler_type::create(nbl::hlsl::float32_t4(normalDist(rng), normalDist(rng), normalDist(rng), smallDist(rng)));
		}
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 1.0; }
	static float64_t gridIntegratePdf(const sampler_type& s) { return gridIntegratePdf2D(s); }
	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    yStarts=%s yDiffs=%s normFactor=%s",
			nbl::system::ILogger::ELL_ERROR, to_string(s.yStarts).c_str(), to_string(s.yDiffs).c_str(), to_string(s.normFactor).c_str());
	}
};

// Mirrors ProjectedSphericalTriangle's bilinear usage: corners = (NdotL_v1, NdotL_v1, NdotL_v0, NdotL_v2)
// with max(NdotL, 0) clamping. Two corners always identical, 0-2 corners can be zero.
struct BilinearPSTPatternConfig
{
	using sampler_type = nbl::hlsl::sampling::Bilinear<nbl::hlsl::float32_t>;

	static constexpr uint32_t numConfigurations = 500;
	static constexpr uint32_t samplesPerConfig = 500000;
	static constexpr bool hasMCNormalization = true;
	static constexpr bool hasGridIntegration = true;
	static constexpr float64_t mcNormalizationRelTol = 0.05;
	static constexpr float64_t gridNormalizationAbsTol = 1e-3;

	static const char* name() { return "Bilinear(PST-pattern)"; }

	static sampler_type createRandomSampler(std::mt19937& rng)
	{
		std::uniform_int_distribution<int> caseDist(0, 5);
		std::uniform_real_distribution<float> valDist(0.01f, 1.0f);
		std::uniform_real_distribution<float> smallDist(1e-6f, 1e-2f);

		// PST corner layout: (v1_NdotL, v1_NdotL, v0_NdotL, v2_NdotL)
		float a, b, c; // v0, v1, v2 NdotL values before clamping
		switch (caseDist(rng))
		{
			case 0: // all positive (non-grazing)
				a = valDist(rng); b = valDist(rng); c = valDist(rng);
				break;
			case 1: // one vertex below horizon (one zero corner)
				a = 0.0f; b = valDist(rng); c = valDist(rng);
				break;
			case 2: // v1 below horizon (two zero corners since v1 appears twice)
				a = valDist(rng); b = 0.0f; c = valDist(rng);
				break;
			case 3: // two vertices below horizon (three zero corners: v1 twice + one other)
				a = 0.0f; b = 0.0f; c = valDist(rng);
				break;
			case 4: // near-grazing: one vertex barely above horizon
				a = smallDist(rng); b = valDist(rng); c = valDist(rng);
				break;
			default: // near-grazing: v1 barely above horizon (two near-zero corners)
				a = valDist(rng); b = smallDist(rng); c = valDist(rng);
				break;
		}
		// PST pattern: .yyxz = (b, b, a, c)
		return sampler_type::create(nbl::hlsl::float32_t4(b, b, a, c));
	}

	static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }
	static float64_t expectedCodomainMeasure(const sampler_type& s) { return 1.0; }
	static float64_t gridIntegratePdf(const sampler_type& s) { return gridIntegratePdf2D(s); }
	static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
	{
		using nbl::system::to_string;
		logger->log("    yStarts=%s yDiffs=%s normFactor=%s",
			nbl::system::ILogger::ELL_ERROR, to_string(s.yStarts).c_str(), to_string(s.yDiffs).c_str(), to_string(s.normFactor).c_str());
	}
};

#endif
