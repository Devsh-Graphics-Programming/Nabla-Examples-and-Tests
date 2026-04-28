#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_DISCRETE_TABLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_DISCRETE_TABLE_TESTER_INCLUDED_

#include <nbl/builtin/hlsl/sampling/alias_table.hlsl>
#include <nbl/builtin/hlsl/sampling/cumulative_probability.hlsl>
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

// Generic ReadOnly accessor wrapping a raw pointer
template<typename T>
   requires std::is_arithmetic_v<T>
struct ReadOnlyAccessor
{
   using value_type = T;
   template<typename V, std::integral I>
      requires std::is_arithmetic_v<V>
   void get(I i, V& val) const { val = V(data[i]); }

   const T* data;
};

// Tests table construction for both alias method and cumulative probability.
// Sampler generate/pdf correctness is verified by GPU testers (CAliasTableGPUTester, CCumulativeProbabilityGPUTester).
class CDiscreteTableTester
{
   public:
   CDiscreteTableTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass  = true;
      auto cases = createTestCases();

      m_logger->log("AliasTableBuilder tests:", system::ILogger::ELL_INFO);
      for (const auto& tc : cases)
         pass &= testAliasTable(tc.name, tc.weights);

      m_logger->log("CumulativeProbability tests:", system::ILogger::ELL_INFO);
      for (const auto& tc : cases)
         pass &= testCumulativeProbability(tc.name, tc.weights);

      m_logger->log("CumulativeProbabilitySampler tests (TRACKING / YOLO / EYTZINGER):", system::ILogger::ELL_INFO);
      for (const auto& tc : cases)
         pass &= testSamplers(tc.name, tc.weights);

      return pass;
   }

   private:
   struct TestCase
   {
      const char*        name;
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
         std::vector<float>                    w(1024);
         std::mt19937                          rng(42);
         std::uniform_real_distribution<float> dist(0.001f, 100.0f);
         for (uint32_t i = 0; i < 1024; i++)
            w[i] = dist(rng);
         cases.push_back({"Random(1024)", std::move(w)});
      }

      // NPoT cases exercise EYTZINGER padded-leaf territory (P > N).
      cases.push_back({"NonPot(7)", {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}});
      {
         std::vector<float>                    w(1000);
         std::mt19937                          rng(4242);
         std::uniform_real_distribution<float> dist(0.001f, 100.0f);
         for (uint32_t i = 0; i < 1000; i++)
            w[i] = dist(rng);
         cases.push_back({"Random(1000)", std::move(w)});
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
      const uint32_t N           = static_cast<uint32_t>(weights.size());
      float          totalWeight = 0.0f;
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
         const float err      = std::abs(expected - pdf[i]);
         if (err > 1e-6f)
         {
            m_logger->log("%s[%s] pdf[%u]: expected %f, got %f (err=%e)", system::ILogger::ELL_ERROR, prefix, name, i, expected, pdf[i], err);
            pass = false;
         }
      }

      return pass;
   }

   // Verify alias table builder output:
   //   - bucket contributions reconstruct correct scaled probabilities
   //   - PDFs sum to 1 and match weight/totalWeight
   //   - alias indices in range, probabilities in [0, 1]
   // Builder transparently pads PoT N to N+1; actual table size comes back
   // as `tableN` and is what gets compared against.
   bool testAliasTable(const char* name, const std::vector<float>& weights) const
   {
      const uint32_t userN = static_cast<uint32_t>(weights.size());

      std::vector<float>    outProbability;
      std::vector<uint32_t> outAlias;
      std::vector<float>    outPdf;
      const uint32_t        tableN = nbl::hlsl::sampling::AliasTableBuilder<float>::build({weights}, outProbability, outAlias, outPdf);

      // Accumulate bucket contributions over the full (possibly padded) table
      std::vector<float> dest(tableN, 0.0f);
      for (uint32_t i = 0; i < tableN; i++)
      {
         dest[i] += outProbability[i];
         dest[outAlias[i]] += (1.0f - outProbability[i]);
      }

      bool pass = true;

      float totalWeight = 0.0f;
      for (uint32_t i = 0; i < userN; i++)
         totalWeight += weights[i];

      // Real buckets: expected scaled prob = weight/total * tableN
      for (uint32_t i = 0; i < userN; i++)
      {
         const float expected  = weights[i] / totalWeight * float(tableN);
         const float err       = std::abs(expected - dest[i]);
         const float tolerance = std::max(1e-5f * float(tableN), 1e-4f);

         if (err > tolerance)
         {
            m_logger->log("AliasTable[%s] bucket %u: expected %f, got %f (err=%e)",
               system::ILogger::ELL_ERROR, name, i, expected, dest[i], err);
            pass = false;
         }
      }

      // Dummy bucket (only when padded): no real bucket aliases to it -> dest[userN] should be 0.
      if (tableN != userN && std::abs(dest[userN]) > 1e-4f)
      {
         m_logger->log("AliasTable[%s] dummy bucket %u has non-zero reconstructed probability %f",
            system::ILogger::ELL_ERROR, name, userN, dest[userN]);
         pass = false;
      }

      // Alias indices in range [0, tableN)
      for (uint32_t i = 0; i < tableN; i++)
      {
         if (outAlias[i] >= tableN)
         {
            m_logger->log("AliasTable[%s] alias[%u] = %u out of range [0, %u)",
               system::ILogger::ELL_ERROR, name, i, outAlias[i], tableN);
            pass = false;
         }
      }

      pass &= verifyPdf("AliasTable", name, outPdf.data(), weights);
      pass &= verifyRange01("AliasTable", name, "probability", outProbability.data(), tableN);

      if (pass)
         m_logger->log("  [%s] PASSED", system::ILogger::ELL_PERFORMANCE, name);

      return pass;
   }

   // Verify CDF table construction: monotonicity, implicit-1.0 invariant, and
   // stored entries in [0, 1]. PDF-from-CDF correctness is covered by the
   // TRACKING sampler test below (same cdf[i] - cdf[i-1] derivation via
   // sampler.backwardPdf), so it's not repeated here.
   bool testCumulativeProbability(const char* name, const std::vector<float>& weights) const
   {
      const uint32_t N = static_cast<uint32_t>(weights.size());

      std::vector<float> cumProb(N - 1);

      nbl::hlsl::sampling::computeNormalizedCumulativeHistogram<float>(std::span<const float>(weights), cumProb.data());

      bool pass = true;

      // Monotonically non-decreasing
      for (uint32_t i = 1; i < N - 1; i++)
      {
         if (cumProb[i] < cumProb[i - 1] - 1e-7f)
         {
            m_logger->log("CumProb[%s] non-monotonic at %u: cumProb[%u]=%f < cumProb[%u]=%f",
               system::ILogger::ELL_ERROR, name, i, i, cumProb[i], i - 1, cumProb[i - 1]);
            pass = false;
         }
      }

      // Last stored entry should be < 1.0 (the Nth bucket is implicitly 1.0)
      if (N > 1 && cumProb[N - 2] >= 1.0f + 1e-6f)
      {
         m_logger->log("CumProb[%s] last stored entry %f >= 1.0", system::ILogger::ELL_ERROR, name, cumProb[N - 2]);
         pass = false;
      }

      pass &= verifyRange01("CumProb", name, "cumProb", cumProb.data(), N - 1);

      if (pass)
         m_logger->log("  [%s] PASSED", system::ILogger::ELL_PERFORMANCE, name);

      return pass;
   }

   // Reference binary search over the full N-entry CDF (last entry == 1.0).
   static uint32_t referenceUpperBound(const std::vector<float>& fullCdf, float u)
   {
      auto it = std::upper_bound(fullCdf.begin(), fullCdf.end(), u);
      return static_cast<uint32_t>(std::distance(fullCdf.begin(), it));
   }

   // Run TRACKING, YOLO, and EYTZINGER samplers against the same reference
   // distribution. Each mode is instantiated via the dual-compile sampler and
   // exercised entirely on the CPU.
   bool testSamplers(const char* name, const std::vector<float>& weights) const
   {
      const uint32_t N = static_cast<uint32_t>(weights.size());
      if (N < 2)
         return true;

      float totalWeight = 0.0f;
      for (uint32_t i = 0; i < N; i++)
         totalWeight += weights[i];
      const float rcpTotal = 1.0f / totalWeight;

      std::vector<float> pdfRef(N);
      std::vector<float> fullCdf(N);
      float              acc = 0.0f;
      for (uint32_t i = 0; i < N; i++)
      {
         pdfRef[i] = weights[i] * rcpTotal;
         acc += pdfRef[i];
         fullCdf[i] = acc;
      }
      fullCdf[N - 1] = 1.0f; // pin the last entry; reference must treat it as exact

      // Storage for TRACKING / YOLO (N-1 entries, last bucket implicit at 1.0).
      std::vector<float> cdfStorage(N - 1);
      nbl::hlsl::sampling::computeNormalizedCumulativeHistogram<float>({weights}, cdfStorage.data());

      // Storage for EYTZINGER (2*P entries, level-order implicit binary tree).
      const uint32_t     P = nbl::hlsl::sampling::eytzingerLeafCount(N);
      std::vector<float> treeStorage(2u * P, 0.0f);
      nbl::hlsl::sampling::buildEytzinger<float>({weights}, treeStorage.data());

      bool pass = true;
      pass &= testSamplerMode<nbl::hlsl::sampling::CumulativeProbabilityMode::TRACKING>("TRACKING", name, N, pdfRef, fullCdf, cdfStorage.data());
      pass &= testSamplerMode<nbl::hlsl::sampling::CumulativeProbabilityMode::YOLO>("YOLO", name, N, pdfRef, fullCdf, cdfStorage.data());
      pass &= testSamplerMode<nbl::hlsl::sampling::CumulativeProbabilityMode::EYTZINGER>("EYTZINGER", name, N, pdfRef, fullCdf, treeStorage.data());
      return pass;
   }

   template<nbl::hlsl::sampling::CumulativeProbabilityMode Mode>
   bool testSamplerMode(const char* modeName, const char* caseName, uint32_t N,
      const std::vector<float>& pdfRef, const std::vector<float>& fullCdf, const float* accessorData) const
   {
      using Sampler = nbl::hlsl::sampling::CumulativeProbabilitySampler<
         float, float, uint32_t, ReadOnlyAccessor<float>, Mode>;

      ReadOnlyAccessor<float> accessor {accessorData};
      Sampler                 sampler = Sampler::create(accessor, N);

      bool pass = true;

      // backwardPdf(v) == pdfRef[v], and the implied PDF sums to 1.
      float backwardSum = 0.0f;
      for (uint32_t v = 0; v < N; v++)
      {
         const float got      = sampler.backwardPdf(v);
         const float expected = pdfRef[v];
         const float err      = std::abs(got - expected);
         const float tol      = 1e-5f;
         if (err > tol)
         {
            m_logger->log("Sampler[%s][%s] backwardPdf[%u]: expected %e, got %e (err=%e)",
               system::ILogger::ELL_ERROR, modeName, caseName, v, expected, got, err);
            pass = false;
         }
         backwardSum += got;
      }
      if (std::abs(backwardSum - 1.0f) > 1e-5f)
      {
         m_logger->log("Sampler[%s][%s] backwardPdf sum: expected 1.0, got %f",
            system::ILogger::ELL_ERROR, modeName, caseName, backwardSum);
         pass = false;
      }

      // generate(u) lands in the correct bucket for a grid of u values, and
      // generate(u, cache) produces forwardPdf matching backwardPdf(result).
      std::mt19937                          rng(1234u + N);
      std::uniform_real_distribution<float> udist(0.0f, std::nextafter(1.0f, 0.0f));
      constexpr uint32_t                    kTrials = 2048;

      for (uint32_t k = 0; k < kTrials; k++)
      {
         const float    u   = udist(rng);
         const uint32_t ref = referenceUpperBound(fullCdf, u);

         const uint32_t idx = sampler.generate(u);
         if (idx != ref)
         {
            m_logger->log("Sampler[%s][%s] generate(%.7f): expected bucket %u, got %u",
               system::ILogger::ELL_ERROR, modeName, caseName, u, ref, idx);
            pass = false;
            continue;
         }

         typename Sampler::cache_type cache;
         const uint32_t               idxCache = sampler.generate(u, cache);
         if (idxCache != ref)
         {
            m_logger->log("Sampler[%s][%s] generate(u,cache)(%.7f): expected %u, got %u",
               system::ILogger::ELL_ERROR, modeName, caseName, u, ref, idxCache);
            pass = false;
            continue;
         }

         const float forwardP  = sampler.forwardPdf(u, cache);
         const float backwardP = sampler.backwardPdf(idxCache);
         if (std::abs(forwardP - backwardP) > 1e-6f)
         {
            m_logger->log("Sampler[%s][%s] fwd/bwd pdf mismatch at u=%.7f bucket=%u: fwd=%e bwd=%e",
               system::ILogger::ELL_ERROR, modeName, caseName, u, idxCache, forwardP, backwardP);
            pass = false;
         }
      }

      if (pass)
         m_logger->log("  [%-9s %s] PASSED", system::ILogger::ELL_PERFORMANCE, modeName, caseName);
      return pass;
   }

   system::ILogger* m_logger;
};

#endif
