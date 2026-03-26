#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_DISCRETE_TABLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_DISCRETE_TABLE_TESTER_INCLUDED_

#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
#include <vector>
#include <random>
#include <cmath>

// Generic ReadOnly accessor wrapping a raw pointer
template<typename T>
struct ReadOnlyAccessor
{
	using value_type = T;
	template<typename V, std::integral I> requires std::is_arithmetic_v<V>
	void get(I i, V& val) const { val = V(data[i]); }
	T operator[](uint32_t i) const { return data[i]; }

	const T* data;
};

using ProbabilityAccessor = ReadOnlyAccessor<float32_t>;
using AliasIndexAccessor = ReadOnlyAccessor<uint32_t>;
using PdfAccessor = ReadOnlyAccessor<float>;

using TestAliasTable = nbl::hlsl::sampling::AliasTable<float32_t, float32_t, uint32_t, ProbabilityAccessor, AliasIndexAccessor, PdfAccessor>;
using TestCumulativeProbabilitySampler = nbl::hlsl::sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>>;

// Tests table construction for both alias method and cumulative probability.
// Sampler generate/pdf correctness is verified by GPU testers (CAliasTableGPUTester, CCumulativeProbabilityGPUTester).
class CDiscreteTableTester
{
public:
	CDiscreteTableTester(system::ILogger* logger) : m_logger(logger) {}

	bool run()
	{
		bool pass = true;
		auto cases = createTestCases();

		m_logger->log("AliasTableBuilder tests:", system::ILogger::ELL_INFO);
		for (const auto& tc : cases)
			pass &= testAliasTable(tc.name, tc.weights);

		m_logger->log("CumulativeProbability tests:", system::ILogger::ELL_INFO);
		for (const auto& tc : cases)
			pass &= testCumulativeProbability(tc.name, tc.weights);

		return pass;
	}

private:
	struct TestCase
	{
		const char* name;
		std::vector<float> weights;
	};

	static std::vector<TestCase> createTestCases()
	{
		std::vector<TestCase> cases;
		cases.push_back({"Uniform(4)", {1.0f, 1.0f, 1.0f, 1.0f}});
		cases.push_back({"NonUniform(1,2,3,4)", {1.0f, 2.0f, 3.0f, 4.0f}});

		{
			std::vector<float> w(32, 1.0f);
			w[31] = 97.0f;
			cases.push_back({"SingleDominant(32)", std::move(w)});
		}
		{
			std::vector<float> w(64);
			for (uint32_t i = 0; i < 64; i++)
				w[i] = 1.0f / float(i + 1);
			cases.push_back({"PowerLaw(64)", std::move(w)});
		}

		cases.push_back({"SingleNonZero(4)", {0.0f, 0.0f, 5.0f, 0.0f}});

		{
			std::vector<float> w(1024);
			std::mt19937 rng(42);
			std::uniform_real_distribution<float> dist(0.001f, 100.0f);
			for (uint32_t i = 0; i < 1024; i++)
				w[i] = dist(rng);
			cases.push_back({"Random(1024)", std::move(w)});
		}

		return cases;
	}

	// Verify all values in array are in [0, 1]
	bool verifyRange01(const char* prefix, const char* name, const char* arrayName, const float* data, uint32_t count) const
	{
		bool pass = true;
		for (uint32_t i = 0; i < count; i++)
		{
			if (data[i] < 0.0f || data[i] > 1.0f + 1e-6f)
			{
				m_logger->log("%s[%s] %s[%u] = %f out of range [0, 1]",
					system::ILogger::ELL_ERROR, prefix, name, arrayName, i, data[i]);
				pass = false;
			}
		}
		return pass;
	}

	// Shared: verify PDFs sum to 1 and each matches weight/totalWeight
	bool verifyPdf(const char* prefix, const char* name, const float* pdf, const std::vector<float>& weights) const
	{
		const uint32_t N = static_cast<uint32_t>(weights.size());
		float totalWeight = 0.0f;
		for (uint32_t i = 0; i < N; i++)
			totalWeight += weights[i];

		bool pass = true;

		float pdfSum = 0.0f;
		for (uint32_t i = 0; i < N; i++)
			pdfSum += pdf[i];

		if (std::abs(pdfSum - 1.0f) > 1e-5f)
		{
			m_logger->log("%s[%s] PDF sum: expected 1.0, got %f", system::ILogger::ELL_ERROR, prefix, name, pdfSum);
			pass = false;
		}

		for (uint32_t i = 0; i < N; i++)
		{
			const float expected = weights[i] / totalWeight;
			const float err = std::abs(expected - pdf[i]);
			if (err > 1e-6f)
			{
				m_logger->log("%s[%s] pdf[%u]: expected %f, got %f (err=%e)", system::ILogger::ELL_ERROR, prefix, name, i, expected, pdf[i], err);
				pass = false;
			}
		}

		return pass;
	}

	// Verify alias table builder output:
	//   - bucket contributions reconstruct correct probabilities
	//   - PDFs sum to 1 and match weight/totalWeight
	//   - alias indices in range, probabilities in [0, 1]
	bool testAliasTable(const char* name, const std::vector<float>& weights) const
	{
		const uint32_t N = static_cast<uint32_t>(weights.size());

		std::vector<float> outProbability(N);
		std::vector<uint32_t> outAlias(N);
		std::vector<float> outPdf(N);
		std::vector<uint32_t> workspace(N);

		nbl::hlsl::sampling::AliasTableBuilder<float>::build({ weights },outProbability.data(), outAlias.data(), outPdf.data(), workspace.data());

		// Accumulate bucket contributions
		std::vector<float> dest(N, 0.0f);
		for (uint32_t i = 0; i < N; i++)
		{
			dest[i] += outProbability[i];
			dest[outAlias[i]] += (1.0f - outProbability[i]);
		}

		bool pass = true;

		float totalWeight = 0.0f;
		for (uint32_t i = 0; i < N; i++)
			totalWeight += weights[i];

		for (uint32_t i = 0; i < N; i++)
		{
			const float expected = weights[i] / totalWeight * float(N);
			const float err = std::abs(expected - dest[i]);
			const float tolerance = std::max(1e-5f * float(N), 1e-4f);

			if (err > tolerance)
			{
				m_logger->log("AliasTable[%s] bucket %u: expected %f, got %f (err=%e)",
					system::ILogger::ELL_ERROR, name, i, expected, dest[i], err);
				pass = false;
			}
		}

		// Alias indices in range
		for (uint32_t i = 0; i < N; i++)
		{
			if (outAlias[i] >= N)
			{
				m_logger->log("AliasTable[%s] alias[%u] = %u out of range [0, %u)",
					system::ILogger::ELL_ERROR, name, i, outAlias[i], N);
				pass = false;
			}
		}

		pass &= verifyPdf("AliasTable", name, outPdf.data(), weights);
		pass &= verifyRange01("AliasTable", name, "probability", outProbability.data(), N);

		if (pass)
			m_logger->log("  [%s] PASSED", system::ILogger::ELL_PERFORMANCE, name);

		return pass;
	}

	// Verify CDF table construction:
	//   - cumulative probabilities are monotonically non-decreasing
	//   - PDFs match weight/totalWeight
	//   - PDFs sum to 1
	bool testCumulativeProbability(const char* name, const std::vector<float>& weights) const
	{
		const uint32_t N = static_cast<uint32_t>(weights.size());

		std::vector<float> cumProb(N - 1);

		nbl::hlsl::sampling::computeNormalizedCumulativeHistogram<float>(
			std::span<const float>(weights),
			cumProb.data());

		bool pass = true;

		// Monotonically non-decreasing
		for (uint32_t i = 1; i < N - 1; i++)
		{
			if (cumProb[i] < cumProb[i - 1] - 1e-7f)
			{
				m_logger->log("CumProb[%s] non-monotonic at %u: cumProb[%u]=%f > cumProb[%u]=%f",
					system::ILogger::ELL_ERROR, name, i, i - 1, cumProb[i - 1], i, cumProb[i]);
				pass = false;
			}
		}

		// Last stored entry should be < 1.0 (the Nth bucket is implicitly 1.0)
		if (N > 1 && cumProb[N - 2] >= 1.0f + 1e-6f)
		{
			m_logger->log("CumProb[%s] last stored entry %f >= 1.0",
				system::ILogger::ELL_ERROR, name, cumProb[N - 2]);
			pass = false;
		}

		// Derive PDF from CDF for verification
		std::vector<float> pdf(N);
		for (uint32_t i = 0; i < N; i++)
		{
			const float cur = (i < N - 1) ? cumProb[i] : 1.0f;
			const float prev = (i > 0) ? cumProb[i - 1] : 0.0f;
			pdf[i] = cur - prev;
		}

		pass &= verifyPdf("CumProb", name, pdf.data(), weights);
		pass &= verifyRange01("CumProb", name, "cumProb", cumProb.data(), N - 1);

		if (pass)
			m_logger->log("  [%s] PASSED", system::ILogger::ELL_PERFORMANCE, name);

		return pass;
	}

	system::ILogger* m_logger;
};

#endif
