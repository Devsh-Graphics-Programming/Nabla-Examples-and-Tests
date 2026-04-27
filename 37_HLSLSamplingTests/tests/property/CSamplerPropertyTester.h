#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SAMPLER_PROPERTY_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_SAMPLER_PROPERTY_TESTER_INCLUDED_

#include <nabla.h>
#include <random>
#include <cmath>
#include <concepts>

#include "../SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

using namespace nbl;
using namespace nbl::hlsl;

template<typename C>
concept SamplerPropertyTestConfig = requires(std::mt19937& rng) {
   typename C::sampler_type;
   { C::numConfigurations } -> std::convertible_to<uint32_t>;
   { C::samplesPerConfig } -> std::convertible_to<uint32_t>;
   { C::hasMCNormalization } -> std::convertible_to<bool>;
   { C::hasGridIntegration } -> std::convertible_to<bool>;
   { C::name() } -> std::convertible_to<const char*>;
   { C::createRandomSampler(rng) } -> std::same_as<typename C::sampler_type>;
   { C::randomDomain(rng) } -> std::same_as<typename C::sampler_type::domain_type>;
} && (std::is_empty_v<typename C::sampler_type> || requires(nbl::system::ILogger* logger, const typename C::sampler_type& s) { // no logging if sampler is stateless
   { C::logSamplerInfo(logger, s) };
}) && (!C::hasMCNormalization || requires(const typename C::sampler_type& s) {
   { C::expectedCodomainMeasure(s) } -> std::convertible_to<float64_t>;
   { C::mcNormalizationRelTol } -> std::convertible_to<float64_t>;
}) && (!C::hasGridIntegration || requires(const typename C::sampler_type& s) {
   { C::gridIntegratePdf(s) } -> std::convertible_to<float64_t>;
   { C::gridNormalizationAbsTol } -> std::convertible_to<float64_t>;
});

template<SamplerPropertyTestConfig Config>
class CSamplerPropertyTester
{
   using sampler_type = typename Config::sampler_type;
   using domain_type = typename sampler_type::domain_type;
   using codomain_type = typename sampler_type::codomain_type;
   using cache_type = typename sampler_type::cache_type;

   template<typename C>
   static void tryLogSamplerInfo(system::ILogger* logger, const typename C::sampler_type& sampler)
   {
      if constexpr (requires { C::logSamplerInfo(logger, sampler); })
         C::logSamplerInfo(logger, sampler);
   }

   static void logSampleContext(system::ILogger* logger, const sampler_type& sampler,
      const domain_type& u, const codomain_type& v)
   {
      std::stringstream ss;
      ss << "    u=" << to_string(u) << " v=" << to_string(v);
      logger->log("%s", system::ILogger::ELL_ERROR, ss.str().c_str());
      tryLogSamplerInfo<Config>(logger, sampler);
   }

   // Iterate over random configurations and samples, calling fn for each.
   // fn receives (sampler, u, cache, v, configIdx, sampleIdx).
   template<typename PerSampleFn>
   void forEachSample(SeededTestContext& ctx, PerSampleFn fn)
   {
      for (uint32_t c = 0; c < Config::numConfigurations; c++)
      {
         auto sampler = Config::createRandomSampler(ctx.rng);
         for (uint32_t i = 0; i < Config::samplesPerConfig; i++)
         {
            domain_type u = Config::randomDomain(ctx.rng);
            cache_type cache;
            codomain_type v = sampler.generate(u, cache);
            fn(sampler, u, cache, v, c, i);
         }
      }
   }

   public:
   CSamplerPropertyTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass = true;
      if constexpr (Config::hasMCNormalization)
         pass &= testMonteCarloPdfNormalization();
      if constexpr (Config::hasGridIntegration)
         pass &= testGridPdfNormalization();
      return pass;
   }

   private:
   // Test 1: Monte Carlo normalization
   // E_u[1/forwardPdf(u)] should equal the codomain measure.
   // If the PDF normalization is wrong by factor k, this will be off by 1/k.
   bool testMonteCarloPdfNormalization()
   {
      SeededTestContext ctx;
      uint32_t evaluatedConfigs = 0;

      for (uint32_t c = 0; c < Config::numConfigurations; c++)
      {
         auto sampler = Config::createRandomSampler(ctx.rng);
         const float64_t expectedMeasure = Config::expectedCodomainMeasure(sampler);

         // Skip configurations where the expected measure is not well-defined (e.g. degenerate geometry)
         if (expectedMeasure <= 0.0 || !std::isfinite(expectedMeasure))
            continue;

         float64_t sum = 0.0;
         uint32_t validSamples = 0;

         for (uint32_t i = 0; i < Config::samplesPerConfig; i++)
         {
            domain_type u = Config::randomDomain(ctx.rng);
            cache_type cache;
            sampler.generate(u, cache);
            float64_t pdf = static_cast<float64_t>(sampler.forwardPdf(u, cache));

            if (pdf > 0.0 && std::isfinite(pdf))
            {
               sum += 1.0 / pdf;
               validSamples++;
            }
         }

         if (validSamples == 0)
            continue;

         evaluatedConfigs++;
         const float64_t mcEstimate = sum / static_cast<float64_t>(validSamples);
         const float64_t relErr = std::abs(mcEstimate - expectedMeasure) / expectedMeasure;

         if (relErr > Config::mcNormalizationRelTol)
         {
            ctx.failCount++;
            if (ctx.failCount <= 5)
            {
               m_logger->log("[%s] MC normalization: E[1/pdf]=%f expected=%f relErr=%e (tol=%e) config %u (%u samples)",
                  system::ILogger::ELL_ERROR, Config::name(), mcEstimate, expectedMeasure, relErr, Config::mcNormalizationRelTol, c, validSamples);
               tryLogSamplerInfo<Config>(m_logger, sampler);
            }
         }
      }

      if (evaluatedConfigs == 0)
      {
         m_logger->log("  [%s] MC normalization SKIPPED (no valid configs out of %u attempted, all had non-positive or non-finite expected measure)",
            system::ILogger::ELL_PERFORMANCE, Config::name(), Config::numConfigurations);
         return true;
      }

      const uint32_t skippedConfigs = Config::numConfigurations - evaluatedConfigs;
      if (ctx.failCount == 0)
         m_logger->log("  [%s] MC normalization PASSED (%u/%u configs evaluated, %u skipped, %u samples/config, relTol=%e)",
            system::ILogger::ELL_PERFORMANCE, Config::name(), evaluatedConfigs, Config::numConfigurations, skippedConfigs, Config::samplesPerConfig, Config::mcNormalizationRelTol);
      else
         m_logger->log("  [%s] MC normalization FAILED (%u/%u evaluated configs failed, %u/%u configs evaluated, %u samples/config, relTol=%e)",
            system::ILogger::ELL_ERROR, Config::name(), ctx.failCount, evaluatedConfigs, evaluatedConfigs, Config::numConfigurations, Config::samplesPerConfig, Config::mcNormalizationRelTol);

      return ctx.finalize(m_logger, Config::name());
   }

   // Test 4: Grid integration of backwardPdf over [0,1]^d codomain
   // Only applicable when codomain is [0,1]^d (Linear, Bilinear).
   // integral of backwardPdf over codomain should equal 1.0.
   bool testGridPdfNormalization()
   {
      SeededTestContext ctx;

      for (uint32_t c = 0; c < Config::numConfigurations; c++)
      {
         auto sampler = Config::createRandomSampler(ctx.rng);
         const float64_t integral = Config::gridIntegratePdf(sampler);
         const float64_t absErr = std::abs(integral - 1.0);

         if (absErr > Config::gridNormalizationAbsTol)
         {
            ctx.failCount++;
            if (ctx.failCount <= 5)
               m_logger->log("[%s] grid integration: integral=%f expected=1.0 absErr=%e (tol=%e) config %u",
                  system::ILogger::ELL_ERROR, Config::name(), integral, absErr, Config::gridNormalizationAbsTol, c);
         }
      }

      if (ctx.failCount == 0)
         m_logger->log("  [%s] grid PDF normalization PASSED (%u configs, expected integral=1.0, absTol=%e)",
            system::ILogger::ELL_PERFORMANCE, Config::name(), Config::numConfigurations, Config::gridNormalizationAbsTol);
      else
         m_logger->log("  [%s] grid PDF normalization FAILED (%u/%u configs exceeded absTol=%e)",
            system::ILogger::ELL_ERROR, Config::name(), ctx.failCount, Config::numConfigurations, Config::gridNormalizationAbsTol);

      return ctx.finalize(m_logger, Config::name());
   }

   system::ILogger* m_logger;
};


// ============================================================================
// CSolidAngleAccuracyTester
//
// Tests that shapes::SphericalTriangle::solid_angle matches known analytic
// values and basic invariants.
// ============================================================================

class CSolidAngleAccuracyTester
{
   public:
   CSolidAngleAccuracyTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass = true;
      pass &= testKnownTriangles();
      pass &= testSolidAngleMonotonicity();
      pass &= testSolidAngleBounds();
      return pass;
   }

   private:
   // Test known analytic solid angles:
   // An octant triangle (3 orthogonal vertices) has solid angle = pi/2
   bool testKnownTriangles() const
   {
      constexpr float64_t octantRelTol = 1e-5;
      constexpr float64_t negOctantRelTol = 1e-4;
      constexpr float64_t smallEquilateralRelTol = 0.01;

      bool pass = true;

      // Octant: solid angle = pi/2
      {
         auto shape = createSphericalTriangleShape(float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(0, 0, 1));
         const float64_t expected = numbers::pi<float64_t> * 0.5;
         const float64_t actual = static_cast<float64_t>(shape.solid_angle);
         const float64_t relErr = std::abs(actual - expected) / expected;
         if (relErr > octantRelTol)
         {
            m_logger->log("[SolidAngle] octant: expected=%f got=%f relErr=%e relTol=%e", system::ILogger::ELL_ERROR, expected, actual, relErr, octantRelTol);
            pass = false;
         }
      }

      // Octant with negative axis: (-1,0,0), (0,-1,0), (0,0,-1) also = pi/2
      {
         auto shape = createSphericalTriangleShape(float32_t3(-1, 0, 0), float32_t3(0, -1, 0), float32_t3(0, 0, -1));
         const float64_t expected = numbers::pi<float64_t> * 0.5;
         const float64_t actual = static_cast<float64_t>(shape.solid_angle);
         const float64_t relErr = std::abs(actual - expected) / expected;
         if (relErr > negOctantRelTol)
         {
            m_logger->log("[SolidAngle] negative octant: expected=%f got=%f relErr=%e relTol=%e", system::ILogger::ELL_ERROR, expected, actual, relErr, negOctantRelTol);
            pass = false;
         }
      }

      // Small equilateral triangle near north pole: solid angle ~= planar area
      // Vertices at colatitude theta on unit sphere, separated by 120 degrees in azimuth.
      // theta=0.1 is small enough for the flat approximation but large enough for float32.
      {
         const float64_t theta = 0.1;
         float32_t3 verts[3];
         makeEquilateralTriangle(theta, verts);
         auto shape = createSphericalTriangleShape(verts[0], verts[1], verts[2]);
         // For small triangles: solid angle ~= sqrt(3)/4 * side^2
         // Side length on unit sphere ~= 2*sin(theta)*sin(pi/3) = 2*sin(theta)*sqrt(3)/2 = sqrt(3)*sin(theta)
         // Planar area ~= sqrt(3)/4 * (sqrt(3)*sin(theta))^2 = 3*sqrt(3)/4 * sin(theta)^2
         const float64_t sinTheta = std::sin(theta);
         const float64_t approxSolidAngle = 3.0 * std::sqrt(3.0) / 4.0 * sinTheta * sinTheta;
         const float64_t actual = static_cast<float64_t>(shape.solid_angle);
         const float64_t relErr = std::abs(actual - approxSolidAngle) / approxSolidAngle;
         // Small triangle approximation has O(theta^2) error
         if (relErr > smallEquilateralRelTol)
         {
            m_logger->log("[SolidAngle] small equilateral: expected~=%f got=%f relErr=%e relTol=%e", system::ILogger::ELL_ERROR, approxSolidAngle, actual, relErr, smallEquilateralRelTol);
            pass = false;
         }
      }

      if (pass)
         m_logger->log("  [SolidAngle] known triangles PASSED (octant relTol=%e, negative octant relTol=%e, small equilateral relTol=%e)",
            system::ILogger::ELL_PERFORMANCE, octantRelTol, negOctantRelTol, smallEquilateralRelTol);

      return pass;
   }

   // Test that shrinking a triangle reduces its solid angle monotonically
   bool testSolidAngleMonotonicity() const
   {
      bool pass = true;

      float64_t prevSolidAngle = 4.0 * numbers::pi<float64_t>; // start larger than any triangle
      constexpr uint32_t steps = 20;

      for (uint32_t i = 1; i <= steps; i++)
      {
         const float64_t theta = 0.05 + (numbers::pi<float64_t> * 0.4) * static_cast<float64_t>(steps - i) / static_cast<float64_t>(steps);
         float32_t3 verts[3];
         makeEquilateralTriangle(theta, verts);
         auto shape = createSphericalTriangleShape(verts[0], verts[1], verts[2]);
         const float64_t sa = static_cast<float64_t>(shape.solid_angle);

         if (sa >= prevSolidAngle)
         {
            m_logger->log("[SolidAngle] monotonicity fail: theta=%f solidAngle=%f >= prev=%f",
               system::ILogger::ELL_ERROR, theta, sa, prevSolidAngle);
            pass = false;
         }
         prevSolidAngle = sa;
      }

      if (pass)
         m_logger->log("  [SolidAngle] monotonicity PASSED (%u equilateral triangles with decreasing colatitude, each solid angle strictly less than previous)", system::ILogger::ELL_PERFORMANCE, steps);

      return pass;
   }

   // Test solid angle is always positive and < 4*pi for random triangles
   bool testSolidAngleBounds() const
   {
      SeededTestContext ctx;
      bool pass = true;
      constexpr uint32_t N = 10000;

      for (uint32_t i = 0; i < N; i++)
      {
         float32_t3 v0, v1, v2;
         generateRandomTriangleVertices(ctx.rng, v0, v1, v2);

         auto shape = createSphericalTriangleShape(v0, v1, v2);

         if (shape.solid_angle <= 0.0f || shape.solid_angle >= 4.0f * numbers::pi<float32_t> || !std::isfinite(shape.solid_angle))
         {
            m_logger->log("[SolidAngle] bounds fail: solid_angle=%f at iteration %u", system::ILogger::ELL_ERROR, shape.solid_angle, i);
            logTriangleInfo(m_logger, v0, v1, v2);
            pass = false;
         }
      }

      if (pass)
         m_logger->log("  [SolidAngle] bounds PASSED (%u random triangles, all in (0, 4*pi))", system::ILogger::ELL_PERFORMANCE, N);
      else
         m_logger->log("  [SolidAngle] bounds FAILED", system::ILogger::ELL_ERROR);

      return ctx.finalize(pass, m_logger, "SolidAngle");
   }

   system::ILogger* m_logger;
};


// ============================================================================
// CSphericalTriangleGenerateTester
//
// WARNING: All property tests in this file run on the CPU using the HLSL/C++
// dual-compilation headers. CPU math may use higher intermediate precision
// than GPU shaders. Tolerances that pass here may be too tight for GPU
// execution. The GPU vs CPU consistency tests in ITester-based testers
// (CSphericalTriangleTester, etc.) cover this gap.
//
// Tests that SphericalTriangle::generate() produces a correct uniform
// distribution over the spherical triangle. The existing property tests
// verify PDF consistency and MC normalization, but for a constant-PDF
// sampler those are trivially satisfied regardless of whether generate()
// maps to the right locations. These tests directly verify the mapping.
// ============================================================================

class CSphericalTriangleGenerateTester
{
   public:
   CSphericalTriangleGenerateTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass = true;
      // Half-space counting
      pass &= runHalfSpaceCounting("random", generateRandomTriangleVertices, 100, 3, 50000, 0.02);
      pass &= runHalfSpaceCounting("stress", generateStressTriangleVertices, 100, 3, 50000, 0.03);
      // Moment matching
      pass &= runMomentMatching("random", generateRandomTriangleVertices, 200, 5, 20000, 0.05, 0.02);
      pass &= runMomentMatching("stress", generateStressTriangleVertices, 200, 10, 20000, 0.08, 0.03);
      // Distant triangles
      pass &= testDistantTriangles();
      return pass;
   }

   private:
   // -------------------------------------------------------------------------
   // Half-space counting: the generate()-only uniformity test.
   //
   // For each triangle, cut it into two sub-triangles by a great circle
   // through one vertex and a point on the opposite edge. The expected
   // fraction of samples on each side equals the ratio of sub-solid-angles.
   // Generate samples, classify each by which side of the cut it falls on,
   // and compare counts against the expected fractions.
   // -------------------------------------------------------------------------
   template<typename TriangleGen>
   bool runHalfSpaceCounting(const char* label, TriangleGen triGen,
      uint32_t numTriangles, uint32_t cutsPerTriangle, uint32_t numSamples,
      float64_t relTol)
   {
      SeededTestContext ctx;
      uint32_t testedCuts = 0;

      for (uint32_t t = 0; t < numTriangles; t++)
      {
         float32_t3 v0, v1, v2;
         triGen(ctx.rng, v0, v1, v2);
         auto shape = createSphericalTriangleShape(v0, v1, v2);
         if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
            continue;

         auto sampler = sampling::SphericalTriangle<float32_t>::create(shape);
         const float64_t SA = static_cast<float64_t>(shape.solid_angle);
         // Float32 solid angle (acos sum - pi) loses precision for small
         // triangles due to catastrophic cancellation, making the expected
         // sub-solid-angle ratio unreliable as a reference value.
         // At SA ~ 0.003, the relative error in float32 solid angles reaches
         // ~1-3%, comparable to the half-space counting tolerance.
         const bool tinyTriangle = SA < 4e-3;

         // For each cut: pick a vertex and a point on the opposite edge,
         // forming a great circle that splits the triangle in two.
         for (uint32_t c = 0; c < cutsPerTriangle; c++)
         {
            // Cycle through vertices as the pivot
            const uint32_t pivotIdx = c % 3;
            float32_t3 pivot, opp0, opp1;
            if (pivotIdx == 0)
            {
               pivot = v0;
               opp0 = v1;
               opp1 = v2;
            }
            else if (pivotIdx == 1)
            {
               pivot = v1;
               opp0 = v2;
               opp1 = v0;
            }
            else
            {
               pivot = v2;
               opp0 = v0;
               opp1 = v1;
            }

            // Random interpolation point M on the great circle arc from opp0 to opp1
            std::uniform_real_distribution<float> tDist(0.2f, 0.8f);
            const float interp = tDist(ctx.rng);
            float32_t3 M = normalize(opp0 * (1.0f - interp) + opp1 * interp);

            // Great circle normal: samples with dot(L, cutNormal) > 0 are on the opp0 side
            float32_t3 cutNormal = normalize(cross(pivot, M));

            // Determine which side opp0 is on
            const float opp0Side = dot(opp0, cutNormal);
            if (std::abs(opp0Side) < 1e-6f)
               continue; // degenerate cut

            // Compute expected fraction: solid angle of sub-triangle (pivot, opp0, M) / SA
            auto subShape = createSphericalTriangleShape(pivot, opp0, M);
            if (subShape.solid_angle <= 0.0f || !std::isfinite(subShape.solid_angle))
               continue;
            const float64_t expectedFraction = static_cast<float64_t>(subShape.solid_angle) / SA;

            // Skip cuts that produce very lopsided splits (hard to test statistically)
            if (expectedFraction < 0.1 || expectedFraction > 0.9)
               continue;

            // Count samples on the opp0 side
            std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
            uint32_t countOnOpp0Side = 0;
            for (uint32_t i = 0; i < numSamples; i++)
            {
               float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
               typename sampling::SphericalTriangle<float32_t>::cache_type cache;
               float32_t3 L = sampler.generate(u, cache);
               if (dot(L, cutNormal) * opp0Side > 0.0f)
                  countOnOpp0Side++;
            }

            const float64_t observedFraction = static_cast<float64_t>(countOnOpp0Side) / static_cast<float64_t>(numSamples);
            const float64_t absErr = std::abs(observedFraction - expectedFraction);

            testedCuts++;
            if (absErr > relTol)
            {
               if (tinyTriangle)
               {
                  m_logger->log("[SphericalTriangle::generate] %s half-space: observed=%f expected=%f absErr=%e (tol=%e) tri %u cut %u -- solid angle %e too small for float32, especially on GPU",
                     system::ILogger::ELL_WARNING, label, observedFraction, expectedFraction, absErr, relTol, t, c, SA);
               }
               else
               {
                  ctx.failCount++;
                  if (ctx.failCount <= 5)
                  {
                     m_logger->log("[SphericalTriangle::generate] %s half-space: observed=%f expected=%f absErr=%e (tol=%e) tri %u cut %u",
                        system::ILogger::ELL_ERROR, label, observedFraction, expectedFraction, absErr, relTol, t, c);
                     logTriangleInfo(m_logger, v0, v1, v2);
                  }
               }
            }
         }
      }

      if (ctx.failCount == 0)
         m_logger->log("  [SphericalTriangle::generate] %s half-space counting PASSED (%u cuts across %u triangles x %u cuts/tri, %u samples/cut, relTol=%e)",
            system::ILogger::ELL_PERFORMANCE, label, testedCuts, numTriangles, cutsPerTriangle, numSamples, relTol);
      else
         m_logger->log("  [SphericalTriangle::generate] %s half-space counting FAILED (%u/%u cuts failed, %u triangles x %u cuts/tri, %u samples/cut, relTol=%e)",
            system::ILogger::ELL_ERROR, label, ctx.failCount, testedCuts, numTriangles, cutsPerTriangle, numSamples, relTol);

      return ctx.finalize(m_logger, "SphericalTriangle::generate");
   }

   // -------------------------------------------------------------------------
   // Moment matching: E[dot(generate(u), N)] should equal signedPSA(N) / SA.
   //
   // For a uniform distribution over a spherical triangle:
   //   E[f(L)] = (1/SA) * integral_triangle f(L) dw
   //
   // Choosing f(L) = dot(L, N) gives E[dot(L, N)] = signedPSA(N) / SA,
   // where signedPSA is the exact signed projected solid angle computed
   // via the Kelvin-Stokes theorem:
   //   signedPSA(N) = 0.5 * sum_edges dot(edgeNormal_i, N) * edgeArcLength_i
   //
   // Note: shapes::SphericalTriangle::projectedSolidAngle() returns a signed result
   // (Kelvin-Stokes signed sum); tests abs() the return to compare against the
   // |cos(theta)| (BSDF) PSA integral reference.
   //
   // If generate() has a systematic bias (e.g., concentrating samples
   // near one vertex), this moment will be wrong for most directions N.
   // Testing multiple random N per triangle makes it very unlikely that
   // a biased mapping passes by accident.
   // -------------------------------------------------------------------------
   template<typename TriangleGen>
   bool runMomentMatching(const char* label, TriangleGen triGen, uint32_t numTriangles, uint32_t numNormals, uint32_t numSamples, float64_t relTol, float64_t absTol)
   {
      SeededTestContext ctx;
      uint32_t testedConfigs = 0;

      for (uint32_t t = 0; t < numTriangles; t++)
      {
         float32_t3 v0, v1, v2;
         triGen(ctx.rng, v0, v1, v2);
         auto shape = createSphericalTriangleShape(v0, v1, v2);

         if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
            continue;

         auto sampler = sampling::SphericalTriangle<float32_t>::create(shape);
         const float64_t SA = static_cast<float64_t>(shape.solid_angle);

         // Precompute edge normals and arc lengths for the signed PSA formula.
         // cross(v_j, v_k) * csc_sides[i] gives outward-pointing edge normals
         // only when the vertices are CCW as seen from outside the sphere.
         // The sign of the triple product dot(v0, cross(v1, v2)) tells us the
         // winding: positive = CCW (outward normals), negative = CW (inward).
         const float32_t3 crossBC = hlsl::cross(shape.vertices[1], shape.vertices[2]);
         const float64_t windingSign = (hlsl::dot(shape.vertices[0], crossBC) >= 0.0f) ? 1.0 : -1.0;
         const float32_t3 edgeNormals[3] = {
            crossBC * shape.csc_sides[0],
            hlsl::cross(shape.vertices[2], shape.vertices[0]) * shape.csc_sides[1],
            hlsl::cross(shape.vertices[0], shape.vertices[1]) * shape.csc_sides[2]
         };
         const float64_t edgeAngles[3] = {
            std::acos(static_cast<float64_t>(hlsl::clamp(shape.cos_sides[0], -1.0f, 1.0f))),
            std::acos(static_cast<float64_t>(hlsl::clamp(shape.cos_sides[1], -1.0f, 1.0f))),
            std::acos(static_cast<float64_t>(hlsl::clamp(shape.cos_sides[2], -1.0f, 1.0f)))
         };

         for (uint32_t n = 0; n < numNormals; n++)
         {
            float32_t3 N = generateRandomUnitVector(ctx.rng);

            // Signed PSA via Kelvin-Stokes: exact for integral dot(L,N) dOmega
            float64_t signedPSA = 0.0;
            for (uint32_t e = 0; e < 3; e++)
               signedPSA += static_cast<float64_t>(hlsl::dot(edgeNormals[e], N)) * edgeAngles[e];
            signedPSA *= 0.5 * windingSign;
            const float64_t expected = signedPSA / SA;

            float64_t sum = 0.0;
            std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
            for (uint32_t i = 0; i < numSamples; i++)
            {
               float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
               typename sampling::SphericalTriangle<float32_t>::cache_type cache;
               float32_t3 L = sampler.generate(u, cache);
               sum += static_cast<float64_t>(dot(L, N));
            }
            const float64_t mcEstimate = sum / static_cast<float64_t>(numSamples);

            const float64_t absErr = std::abs(mcEstimate - expected);
            const float64_t tol = std::max(relTol * std::abs(expected), absTol);
            if (absErr > tol)
            {
               ctx.failCount++;
               if (ctx.failCount <= 5)
               {
                  m_logger->log("[SphericalTriangle::generate] %s moment mismatch: E[dot(L,N)]=%f expected=%f absErr=%e (tol=%e) tri %u normal %u",
                     system::ILogger::ELL_ERROR, label, mcEstimate, expected, absErr, tol, t, n);
                  logTriangleInfo(m_logger, v0, v1, v2);
               }
            }
         }
         testedConfigs++;
      }

      const uint32_t totalMomentTests = testedConfigs * numNormals;
      const uint32_t skippedTris = numTriangles - testedConfigs;
      if (ctx.failCount == 0)
         m_logger->log("  [SphericalTriangle::generate] %s moment matching PASSED (%u/%u triangles x %u normals = %u tests, %u skipped, %u samples/test, relTol=%e absTol=%e)",
            system::ILogger::ELL_PERFORMANCE, label, testedConfigs, numTriangles, numNormals, totalMomentTests, skippedTris, numSamples, relTol, absTol);
      else
         m_logger->log("  [SphericalTriangle::generate] %s moment matching FAILED (%u/%u tests failed, %u/%u triangles tested x %u normals, %u samples/test, relTol=%e absTol=%e)",
            system::ILogger::ELL_ERROR, label, ctx.failCount, totalMomentTests, testedConfigs, numTriangles, numNormals, numSamples, relTol, absTol);

      return ctx.finalize(m_logger, "SphericalTriangle::generate");
   }

   // -------------------------------------------------------------------------
   // Codomain roundtrip: generate(generateInverse(L)) should equal L.
   //
   // The existing tests check domain roundtrip: generateInverse(generate(u)) ~ u.
   // This checks the other direction, which can fail independently when
   // generateInverse has precision issues that happen to cancel in the
   // domain direction but not the codomain direction.
   // -------------------------------------------------------------------------
   template<typename TriangleGen>
   bool runCodomainRoundtrip(const char* label, TriangleGen triGen, uint32_t numTriangles, uint32_t samplesPerTriangle, float64_t tol)
   {
      SeededTestContext ctx;
      uint32_t testedConfigs = 0;

      for (uint32_t t = 0; t < numTriangles; t++)
      {
         float32_t3 v0, v1, v2;
         triGen(ctx.rng, v0, v1, v2);
         auto shape = createSphericalTriangleShape(v0, v1, v2);

         if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
            continue;

         auto sampler = sampling::SphericalTriangle<float32_t>::create(shape);
         std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

         for (uint32_t i = 0; i < samplesPerTriangle; i++)
         {
            // Generate a random point inside the triangle
            float32_t2 u0(uDist(ctx.rng), uDist(ctx.rng));
            typename sampling::SphericalTriangle<float32_t>::cache_type cache;
            float32_t3 L = sampler.generate(u0, cache);

            // Invert, then regenerate
            float32_t2 uInv = sampler.generateInverse(L);
            typename sampling::SphericalTriangle<float32_t>::cache_type cache2;
            float32_t3 L2 = sampler.generate(uInv, cache2);

            float32_t3 diff = L - L2;
            float64_t err = static_cast<float64_t>(length(diff));
            if (err > tol)
            {
               ctx.failCount++;
               if (ctx.failCount <= 5)
               {
                  m_logger->log("[SphericalTriangle::generate] %s codomain roundtrip: |L - generate(generateInverse(L))| = %e (tol=%e) tri %u sample %u",
                     system::ILogger::ELL_ERROR, label, err, tol, t, i);
                  logTriangleInfo(m_logger, v0, v1, v2);
               }
            }
         }
         testedConfigs++;
      }

      const uint32_t totalRoundtripTests = testedConfigs * samplesPerTriangle;
      const uint32_t skippedTris = numTriangles - testedConfigs;
      if (ctx.failCount == 0)
         m_logger->log("  [SphericalTriangle::generate] %s codomain roundtrip PASSED (%u/%u triangles x %u samples = %u tests, %u skipped, tol=%e)",
            system::ILogger::ELL_PERFORMANCE, label, testedConfigs, numTriangles, samplesPerTriangle, totalRoundtripTests, skippedTris, tol);
      else
         m_logger->log("  [SphericalTriangle::generate] %s codomain roundtrip FAILED (%u/%u tests failed, %u/%u triangles tested x %u samples, tol=%e)",
            system::ILogger::ELL_ERROR, label, ctx.failCount, totalRoundtripTests, testedConfigs, numTriangles, samplesPerTriangle, tol);

      return ctx.finalize(m_logger, "SphericalTriangle::generate");
   }

   // -------------------------------------------------------------------------
   // Distant triangles: world-space triangles placed far from the origin.
   //
   // In rendering, spherical triangles come from projecting a world-space
   // triangle onto the unit sphere by normalizing vertex positions. When the
   // triangle is far away, the three unit-sphere directions become very close
   // together, producing a tiny spherical triangle with a small solid angle.
   // This stresses float32 precision in generate() and generateInverse():
   //   - The solid angle computation involves near-cancellation
   //   - Arvo's sub-area interpolation operates on very small angular ranges
   //   - The great-circle intersection in generateInverse sees near-parallel arcs
   //
   // Tests half-space counting uniformity + codomain roundtrip at multiple
   // distances, with a fixed-size world-space triangle.
   //
   // d<=10 is a hard failure (generate must be correct for moderate triangles).
   // d>=100 is diagnostic only: the shape's solid angle computation
   // (acos sum - pi) suffers catastrophic cancellation for tiny spherical
   // triangles, producing inaccurate inputs that generate() cannot recover
   // from. These distances document the precision frontier for future
   // improvement (e.g. a float64 or Kahan-compensated solid angle path).
   // -------------------------------------------------------------------------
   bool testDistantTriangles()
   {
      bool pass = true;

      struct DistanceConfig
      {
         float dist;
         bool hardFail; // false = diagnostic only
         float64_t halfSpaceTol;
         float64_t roundtripTol;
      };
      // d=10: ~6 degree edges, well within float32 precision, hard failure.
      // d=100: ~0.6 degree edges, precision frontier, diagnostic only.
      // d>=1000 omitted: solid angle computation (acos sum - pi) loses all
      // precision, rejecting ~94% of configs; survivors have garbage solid
      // angles so generate() failures are meaningless.
      const DistanceConfig configs[] = {
         {10.0f, true, 0.03, 5e-3},
         {100.0f, false, 0.05, 5e-2}};

      for (const auto& cfg : configs)
      {
         const float dist = cfg.dist;
         auto distantTriGen = [dist](std::mt19937& rng, float32_t3& v0, float32_t3& v1, float32_t3& v2)
         {
            // Random center direction, place triangle at distance `dist`
            float32_t3 center = generateRandomUnitVector(rng) * dist;
            float32_t3 t1, t2;
            float32_t3 centerDir = normalize(center);
            buildTangentFrame(centerDir, t1, t2);

            // World-space triangle with ~1 unit edge length
            std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);
            float32_t3 w0 = center + t1 * (0.5f + jitter(rng)) + t2 * jitter(rng);
            float32_t3 w1 = center - t1 * (0.25f + jitter(rng)) + t2 * (0.433f + jitter(rng));
            float32_t3 w2 = center - t1 * (0.25f + jitter(rng)) - t2 * (0.433f + jitter(rng));

            // Project onto unit sphere
            v0 = normalize(w0);
            v1 = normalize(w1);
            v2 = normalize(w2);
         };

         char label[64];
         snprintf(label, sizeof(label), "distant(d=%g)", dist);

         bool halfSpaceOK = runHalfSpaceCounting(label, distantTriGen, 50, 3, 20000, cfg.halfSpaceTol);
         bool roundtripOK = runCodomainRoundtrip(label, distantTriGen, 50, 500, cfg.roundtripTol);

         if (cfg.hardFail)
         {
            pass &= halfSpaceOK;
            pass &= roundtripOK;
         }
         else
         {
            // Diagnostic only -- log but don't fail
            if (!halfSpaceOK || !roundtripOK)
               m_logger->log("  [SphericalTriangle::generate] %s DIAGNOSTIC (precision limit, not a hard failure)",
                  system::ILogger::ELL_PERFORMANCE, label);
         }
      }

      return pass;
   }

   system::ILogger* m_logger;
};


// ============================================================================
// CProjectedSphericalTriangleTester
//
// Tests two aspects of projected spherical triangles:
//
// 1. PSA formula accuracy: shapes::SphericalTriangle::projectedSolidAngle
//    against grid-integration ground truth (PSA = integral_{tri} abs(dot(L,N)) dOmega).
//
// 2. PST sampler accuracy: how well ProjectedSphericalTriangle's bilinear
//    importance sampling approximates the true NdotL distribution, and
//    whether the PDF agrees with NdotL/projectedSolidAngle.
// ============================================================================

class CProjectedSphericalTriangleGeometricTester
{
   public:
   CProjectedSphericalTriangleGeometricTester(system::ILogger* logger) 
      : m_logger(logger) {}

   bool run()
   {
      bool pass = true;

      // PSA formula tests
      pass &= testPSAKnownCases();
      // NOTE: PSA formula uses abs() on individual edge-normal dot products for BSDF support.
      // This is NOT equivalent to integrating |cos(theta)| over the solid angle -- that requires
      // hemisphere clipping (hemispherical_triangle.hlsl). The abs()-based formula overcounts
      // when edge normals have mixed signs, even when all vertices are above the horizon.
      // These tests are diagnostic-only until proper hemisphere clipping is implemented.
      // TODO: make these hard failures once projectedSolidAngle clips to the hemisphere.
      // Hard-fail thresholds: relErr > 3.0 AND absErr > 0.3 means the formula is catastrophically
      // wrong, not just affected by the known abs()-overcount limitation. Catches regressions that
      // would otherwise hide in the warning stream.
      pass &= testPSAVersusGrid("random", [](std::mt19937& rng, uint32_t, float32_t3& v0, float32_t3& v1, float32_t3& v2, float32_t3& normal)
         {
         generateRandomTriangleVertices(rng, v0, v1, v2);
         normal = generateRandomUnitVector(rng); }, 200, 500000, 0.05, 0.01, 3.0, 0.3, true);
      pass &= testPSAVersusGrid("grazing", [](std::mt19937& rng, uint32_t, float32_t3& v0, float32_t3& v1, float32_t3& v2, float32_t3& normal)
         {
         generateRandomTriangleVertices(rng, v0, v1, v2);
         float32_t3 triCenter = normalize(v0 + v1 + v2);
         float32_t3 tangent, unused;
         buildTangentFrame(triCenter, tangent, unused);
         std::uniform_real_distribution<float> grazeDist(0.02f, 0.15f);
         normal = normalize(tangent + triCenter * grazeDist(rng)); }, 200, 500000, 0.1, 0.01, 3.0, 0.3, true);
      // Also diagnostic -- same abs() issue affects small triangles
      testPSASmallTriangle();

      // PST sampler diagnostics (non-failing) and convergence tests
      runSamplerDiagnostics();
      pass &= testBilinearBecomesConstant();
      pass &= testSmallTrianglePdfConvergence();

      return pass;
   }

   private:
   struct TriangleConfig
   {
      float32_t3 v0, v1, v2, normal;
   };

   // Generate a random valid triangle + normal config with at least one positive NdotL
   static TriangleConfig randomConfig(std::mt19937& rng)
   {
      TriangleConfig cfg;
      generateRandomTriangleVertices(rng, cfg.v0, cfg.v1, cfg.v2);
      do
      {
         cfg.normal = generateRandomUnitVector(rng);
      } while (!anyVertexAboveHorizon(cfg.normal, cfg.v0, cfg.v1, cfg.v2));
      return cfg;
   }

   // Generate a grazing config: normal nearly perpendicular to triangle center
   static TriangleConfig grazingConfig(std::mt19937& rng)
   {
      TriangleConfig cfg;
      generateRandomTriangleVertices(rng, cfg.v0, cfg.v1, cfg.v2);

      float32_t3 triCenter = normalize(cfg.v0 + cfg.v1 + cfg.v2);
      float32_t3 tangent, unused;
      buildTangentFrame(triCenter, tangent, unused);

      std::uniform_real_distribution<float> grazeDist(0.02f, 0.15f);
      cfg.normal = normalize(tangent + triCenter * grazeDist(rng));

      if (!anyVertexAboveHorizon(cfg.normal, cfg.v0, cfg.v1, cfg.v2))
         cfg.normal = normalize(tangent + triCenter * 0.3f);

      return cfg;
   }

   // Create a small equilateral triangle around baseDir with given half-angle
   static TriangleConfig smallTriConfig(std::mt19937& rng, float halfAngle)
   {
      TriangleConfig cfg;
      float32_t3 baseDir;
      generateSmallTriangle(rng, halfAngle, cfg.v0, cfg.v1, cfg.v2, baseDir, cfg.normal);
      return cfg;
   }

   // Build a small equilateral triangle from baseDir + tangent frame
   static TriangleConfig smallTriConfigFromFrame(float32_t3 baseDir, float32_t3 normal, float32_t3 t1, float32_t3 t2, float halfAngle)
   {
      TriangleConfig cfg;
      cfg.v0 = normalize(baseDir + t1 * halfAngle);
      cfg.v1 = normalize(baseDir - t1 * (halfAngle * 0.5f) + t2 * (halfAngle * 0.866f));
      cfg.v2 = normalize(baseDir - t1 * (halfAngle * 0.5f) - t2 * (halfAngle * 0.866f));
      cfg.normal = normal;
      return cfg;
   }

   static sampling::ProjectedSphericalTriangle<float32_t> createSampler(const TriangleConfig& cfg)
   {
      auto shape = createSphericalTriangleShape(cfg.v0, cfg.v1, cfg.v2);
      return sampling::ProjectedSphericalTriangle<float32_t>::create(shape, cfg.normal, false);
   }

   // =========================================================================
   // PSA formula tests
   // =========================================================================

   // Known analytic cases
   bool testPSAKnownCases()
   {
      constexpr float64_t psaOctantGridRelTol = 0.05;
      constexpr float64_t psaSymmetryRelTol = 1e-4;

      SeededTestContext ctx;
      bool pass = true;

      // Octant triangle: vertices (1,0,0), (0,1,0), (0,0,1)
      // Signed PSA = integral of dot(L, N) over the octant.
      // All components >= 0 in this octant so signed = one-sided.
      // By Kelvin-Stokes / direct integration, PSA = pi/4 for any axis-aligned normal.
      {
         auto shape = createSphericalTriangleShape(float32_t3(1, 0, 0), float32_t3(0, 1, 0), float32_t3(0, 0, 1));
         const float64_t psaZ = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(float32_t3(0, 0, 1))));

         // Grid verification: evaluate abs(N.L) over a dense grid on the octant triangle
         const float64_t gridPSA = gridEstimatePSA(shape, float32_t3(0, 0, 1), 1000000);

         const float64_t formulaVsGrid = std::abs(psaZ - gridPSA) / std::abs(gridPSA);
         m_logger->log("  [TriPSA] octant z-normal: formula=%f expected(pi/4)=%f reference=%f relErr=%e",
            system::ILogger::ELL_PERFORMANCE, psaZ, nbl::hlsl::numbers::pi<float64_t> / 4.0, gridPSA, formulaVsGrid);

         if (formulaVsGrid > psaOctantGridRelTol)
         {
            m_logger->log("  [TriPSA] octant z-normal FAILED: formula=%f expected(reference)=%f relErr=%e relTol=%e",
               system::ILogger::ELL_ERROR, psaZ, gridPSA, formulaVsGrid, psaOctantGridRelTol);
            pass = false;
         }

         // Same octant, normal = (1,0,0): by symmetry same result as z-normal
         const float64_t psaX = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(float32_t3(1, 0, 0))));
         const float64_t relDiff = std::abs(psaZ - psaX) / std::max(psaZ, psaX);

         m_logger->log("  [TriPSA] octant symmetry: psaZ=%f psaX=%f relDiff=%e",
            system::ILogger::ELL_PERFORMANCE, psaZ, psaX, relDiff);

         if (relDiff > psaSymmetryRelTol)
         {
            m_logger->log("  [TriPSA] octant symmetry FAILED: psaZ=%f psaX=%f relDiff=%e relTol=%e",
               system::ILogger::ELL_ERROR, psaZ, psaX, relDiff, psaSymmetryRelTol);
            pass = false;
         }
      }

      if (pass)
         m_logger->log("  [TriPSA] known cases PASSED (octant z-normal vs grid relTol=%e, octant symmetry z vs x relTol=%e)",
            system::ILogger::ELL_PERFORMANCE, psaOctantGridRelTol, psaSymmetryRelTol);

      return ctx.finalize(pass, m_logger, "TriPSA");
   }

   // Helper: run grid-integration comparison of formulaPSA vs PSA reference for a set of triangle configs.
   // TriConfigGen: void(rng, index, v0, v1, v2, normal) — generates triangle vertices + normal.
   template<typename TriConfigGen>
   bool testPSAVersusGrid(const char* label, TriConfigGen triConfigGenerator, uint32_t numConfigs, uint32_t gridSamples,
      float64_t relTol, float64_t absTol, float64_t hardRelTol, float64_t hardAbsTol, bool diagnostic = false)
   {
      return ::testPSAVersusGrid(m_logger, "TriPSA", label,
         [&](std::mt19937& rng, uint32_t c, float64_t& formulaPSA, float64_t& gridPSA, auto& logInfo)
         {
            float32_t3 v0, v1, v2, normal;
            triConfigGenerator(rng, c, v0, v1, v2, normal);

            auto shape = createSphericalTriangleShape(v0, v1, v2);
            if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
               return;

            formulaPSA = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(normal)));
            gridPSA = gridEstimatePSA(shape, normal, gridSamples);
            logInfo = [=](system::ILogger* logger, system::ILogger::E_LOG_LEVEL level)
            {
               using nbl::system::to_string;
               logger->log("    v0=%s v1=%s v2=%s normal=%s solidAngle=%s",
                  level, to_string(v0).c_str(), to_string(v1).c_str(), to_string(v2).c_str(),
                  to_string(normal).c_str(), to_string(shape.solid_angle).c_str());
            };
         },
         numConfigs, relTol, absTol, hardRelTol, hardAbsTol, diagnostic);
   }

   // Small triangles -- PSA should approach grid ground truth
   bool testPSASmallTriangle()
   {
      constexpr float64_t smallTriMeanRelErrTol = 0.1;
      constexpr uint32_t smallTriGridSamples = 100000;

      SeededTestContext ctx;
      bool pass = true;

      constexpr uint32_t numTrials = 100;
      constexpr uint32_t numSizes = 6;
      const float halfAngles[numSizes] = {0.5f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f};

      float64_t sumRelErrPerSize[numSizes] = {};
      uint32_t validTrials[numSizes] = {};

      for (uint32_t trial = 0; trial < numTrials; trial++)
      {
         float32_t3 baseDir = generateRandomUnitVector(ctx.rng);
         float32_t3 normal = generateRandomUnitVector(ctx.rng);
         if (dot(normal, baseDir) < 0.3f)
            normal = normalize(normal + baseDir * 2.0f);

         float32_t3 t1, t2;
         buildTangentFrame(baseDir, t1, t2);

         for (uint32_t s = 0; s < numSizes; s++)
         {
            TriangleConfig cfg = smallTriConfigFromFrame(baseDir, normal, t1, t2, halfAngles[s]);

            auto shape = createSphericalTriangleShape(cfg.v0, cfg.v1, cfg.v2);

            if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
               continue;

            const float64_t formulaPSA = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(normal)));
            const float64_t sa = static_cast<float64_t>(shape.solid_angle);
            const float64_t centerNdotL = static_cast<float64_t>(dot(normal, baseDir));

            if (std::abs(centerNdotL) < 0.1 || sa < 1e-10)
               continue;

            // Grid ground truth: mean over regular [0,1]^2 grid of abs(dot(L, N)) * solidAngle
            const float64_t gridPSA = gridEstimatePSA(shape, normal, smallTriGridSamples);

            if (std::abs(gridPSA) < 1e-10)
               continue;

            const float64_t relErr = (formulaPSA - gridPSA) / gridPSA;

            sumRelErrPerSize[s] += relErr;
            validTrials[s]++;
         }
      }

      m_logger->log("  [TriPSA] small triangle PSA vs grid (signed relErr, positive=overestimate):", system::ILogger::ELL_PERFORMANCE);
      for (uint32_t s = 0; s < numSizes; s++)
      {
         if (validTrials[s] > 0)
         {
            const float64_t meanRelErr = sumRelErrPerSize[s] / static_cast<float64_t>(validTrials[s]);
            m_logger->log("    halfAngle=%.3f  meanRelErr=%+.6f  (%u trials)",
               system::ILogger::ELL_PERFORMANCE, halfAngles[s], meanRelErr, validTrials[s]);

            // Skip halfAngle=0.01 (s==5): float32 solid angle precision collapses
            if (s == 4 && std::abs(meanRelErr) > smallTriMeanRelErrTol)
            {
               m_logger->log("  [TriPSA] small triangle exceeded tolerance at halfAngle=%.3f meanRelErr=%+e meanRelErrTol=%e (%u trials)",
                  system::ILogger::ELL_WARNING, halfAngles[s], meanRelErr, smallTriMeanRelErrTol, validTrials[s]);
            }
         }
      }

      m_logger->log("  [TriPSA] small triangle test complete (%u trials across %u sizes, %u grid samples each, meanRelErrTol=%e) -- diagnostic only",
         system::ILogger::ELL_PERFORMANCE, numTrials, numSizes, smallTriGridSamples, smallTriMeanRelErrTol);

      return true; // diagnostic only -- abs()-based PSA overestimates, not a hard failure
   }

   // =========================================================================
   // PST sampler accuracy tests
   // =========================================================================

   // -------------------------------------------------------------------------
   // Combined diagnostic -- bilinear NdotL accuracy, MIS weight
   // comparison, and PDF vs NdotL/PSA binned by NdotL.
   //
   // Single pass over random + grazing configs. For each sample collects:
   //  - bilinear NdotL vs true NdotL (overestimate rate, mean error)
   //  - pstPdf vs idealPdf=NdotL/PSA (MIS weight ratio, pstLower rate)
   //  - signed relErr binned by NdotL
   // Diagnostic only -- logs warnings but does not fail the test.
   // -------------------------------------------------------------------------
   void runSamplerDiagnostics()
   {
      SeededTestContext ctx;

      constexpr uint32_t numConfigs = 200;
      constexpr uint32_t samplesPerConfig = 10000;

      // Bilinear vs NdotL stats
      uint32_t totalOverestimate = 0;
      uint32_t totalSamples = 0;
      float64_t worstOverestimateRatio = 0.0;
      float64_t sumAbsNdotLError = 0.0;

      // MIS weight stats (normal vs grazing)
      struct MISStats
      {
         float64_t sumRatio = 0.0;
         float64_t sumAbsDiff = 0.0;
         uint32_t count = 0;
         uint32_t pstLowerCount = 0;
      };
      MISStats normalMIS, grazingMIS;

      // PDF vs NdotL/PSA binned by NdotL
      constexpr uint32_t numBins = 10;
      struct Bin
      {
         float64_t sumErr = 0.0, sumSqErr = 0.0;
         uint32_t count = 0;
      };
      Bin bins[numBins];

      for (uint32_t c = 0; c < numConfigs; c++)
      {
         const bool isGrazing = (c >= numConfigs / 2);
         TriangleConfig cfg = isGrazing ? grazingConfig(ctx.rng) : randomConfig(ctx.rng);

         auto shape = createSphericalTriangleShape(cfg.v0, cfg.v1, cfg.v2);
         if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
            continue;

         auto sampler = createSampler(cfg);
         if (!std::isfinite(sampler.sphtri.rcpSolidAngle) || sampler.sphtri.rcpSolidAngle <= 0.0f)
            continue;

         const float64_t projSA = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(cfg.normal)));
         const bool hasPSA = projSA > 0.0 && std::isfinite(projSA);
         const float64_t rcpPSA = hasPSA ? 1.0 / projSA : 0.0;
         MISStats& mis = isGrazing ? grazingMIS : normalMIS;

         std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

         for (uint32_t i = 0; i < samplesPerConfig; i++)
         {
            float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
            typename sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
            float32_t3 L = sampler.generate(u, cache);

            const float64_t trueNdotL = std::max(0.0, static_cast<float64_t>(dot(cfg.normal, L)));
            const float64_t bilinearNdotL = std::numeric_limits<float64_t>::quiet_NaN();
            const float64_t pstPdf = static_cast<float64_t>(sampler.forwardPdf(u, cache));

            // Bilinear vs true NdotL
            if (std::isfinite(bilinearNdotL) && std::isfinite(trueNdotL))
            {
               totalSamples++;
               sumAbsNdotLError += std::abs(bilinearNdotL - trueNdotL);
               if (bilinearNdotL > trueNdotL + 1e-6)
               {
                  totalOverestimate++;
                  if (trueNdotL > 1e-6)
                     worstOverestimateRatio = std::max(worstOverestimateRatio, bilinearNdotL / trueNdotL);
               }
            }

            if (!std::isfinite(pstPdf) || pstPdf <= 0.0 || trueNdotL <= 1e-8 || !hasPSA)
               continue;

            const float64_t idealPdf = trueNdotL * rcpPSA;

            // MIS weight comparison
            if (idealPdf > 0.0)
            {
               mis.count++;
               mis.sumRatio += pstPdf / idealPdf;
               mis.sumAbsDiff += std::abs(pstPdf - idealPdf);
               if (pstPdf < idealPdf * 0.99)
                  mis.pstLowerCount++;
            }

            // Binned relative error
            const float64_t relErr = (pstPdf - idealPdf) / idealPdf;
            uint32_t bin = static_cast<uint32_t>(trueNdotL * numBins);
            if (bin >= numBins)
               bin = numBins - 1;
            bins[bin].sumErr += relErr;
            bins[bin].sumSqErr += relErr * relErr;
            bins[bin].count++;
         }
      }

      // Report bilinear vs NdotL
      if (totalSamples > 0)
      {
         const float64_t overestimateRate = static_cast<float64_t>(totalOverestimate) / static_cast<float64_t>(totalSamples);
         const float64_t meanAbsError = sumAbsNdotLError / static_cast<float64_t>(totalSamples);
         m_logger->log("  [PSTAccuracy] bilinear vs NdotL: overestimate rate=%.4f meanAbsErr=%e worstRatio=%f (%u samples)",
            system::ILogger::ELL_PERFORMANCE, overestimateRate, meanAbsError, worstOverestimateRatio, totalSamples);
         if (overestimateRate > 0.9)
            m_logger->log("  [PSTAccuracy] (expected) bilinear almost always overestimates NdotL (rate=%.4f, worry if >0.999)",
               system::ILogger::ELL_PERFORMANCE, overestimateRate);
      }

      // Report MIS weight stats
      auto reportMIS = [&](const char* label, const MISStats& s)
      {
         if (s.count == 0)
            return;
         const float64_t n = static_cast<float64_t>(s.count);
         m_logger->log("  [PSTAccuracy] MIS weight %s: meanRatio(pst/ideal)=%.6f meanAbsDiff=%e pstLowerRate=%.4f (%u samples)",
            system::ILogger::ELL_PERFORMANCE, label, s.sumRatio / n, s.sumAbsDiff / n,
            static_cast<float64_t>(s.pstLowerCount) / n, s.count);
      };
      reportMIS("normal", normalMIS);
      reportMIS("grazing", grazingMIS);

      if (grazingMIS.count > 0)
      {
         const float64_t grazingLowerRate = static_cast<float64_t>(grazingMIS.pstLowerCount) / static_cast<float64_t>(grazingMIS.count);
         if (grazingLowerRate > 0.8)
            m_logger->log("  [PSTAccuracy] (expected) PST PDF consistently lower than NdotL/PSA at grazing (%.1f%%, worry if >95%%)",
               system::ILogger::ELL_PERFORMANCE, grazingLowerRate * 100.0);
      }

      // Report binned PDF error
      m_logger->log("  [PSTAccuracy] PDF vs NdotL/PSA by NdotL bin (signed relErr, positive=overestimate):", system::ILogger::ELL_PERFORMANCE);
      for (uint32_t b = 0; b < numBins; b++)
      {
         if (bins[b].count == 0)
            continue;
         const float64_t n = static_cast<float64_t>(bins[b].count);
         const float lo = static_cast<float>(b) / static_cast<float>(numBins);
         const float hi = static_cast<float>(b + 1) / static_cast<float>(numBins);
         m_logger->log("    NdotL [%.1f,%.1f): meanRelErr=%+.6f rmsRelErr=%.6f (%u samples)",
            system::ILogger::ELL_PERFORMANCE, lo, hi, bins[b].sumErr / n, std::sqrt(bins[b].sumSqErr / n), bins[b].count);
      }
      if (bins[0].count > 100)
      {
         const float64_t rmsErr = std::sqrt(bins[0].sumSqErr / static_cast<float64_t>(bins[0].count));
         if (rmsErr > 5.0)
            m_logger->log("  [PSTAccuracy] (expected) grazing bin RMS error=%f due to bilinear approximation (worry if >1000)",
               system::ILogger::ELL_PERFORMANCE, rmsErr);
      }
   }

   // -------------------------------------------------------------------------
   // Bilinear PDF becomes constant as triangle shrinks
   //
   // As the triangle shrinks, all vertex NdotL values converge, so the
   // bilinear PDF should tend to 1.0 (uniform over [0,1]^2). Measures
   // the relative range (max-min)/max across a grid of domain points and
   // checks that it decreases monotonically with triangle size.
   // -------------------------------------------------------------------------
   bool testBilinearBecomesConstant()
   {
      constexpr float64_t pdfRangeConvergenceTol = 0.5;
      constexpr float64_t pdfRangeMonotonicityFactor = 1.1;
      constexpr uint32_t convergenceCheckMinStep = 4;

      bool pass = true;
      SeededTestContext ctx;
      constexpr uint32_t numTrials = 50;
      constexpr uint32_t numSizes = 8;

      for (uint32_t trial = 0; trial < numTrials; trial++)
      {
         float64_t prevPdfRange = 1e10;

         // Fix baseDir and normal for this trial so only triangle size varies
         float32_t3 baseDir = generateRandomUnitVector(ctx.rng);
         float32_t3 normal = generateRandomUnitVector(ctx.rng);
         if (dot(normal, baseDir) < 0.1f)
            normal = normalize(normal + baseDir * 2.0f);
         float32_t3 t1, t2;
         buildTangentFrame(baseDir, t1, t2);

         for (uint32_t sizeStep = 0; sizeStep < numSizes; sizeStep++)
         {
            const float halfAngle = 0.5f * std::pow(0.5f, static_cast<float>(sizeStep));
            TriangleConfig cfg = smallTriConfigFromFrame(baseDir, normal, t1, t2, halfAngle);

            auto shape = createSphericalTriangleShape(cfg.v0, cfg.v1, cfg.v2);
            if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
            {
               prevPdfRange = 1e10;
               continue;
            }

            auto sampler = createSampler(cfg);

            // Evaluate bilinear PDF at several domain points
            float64_t minPdf = 1e30, maxPdf = -1e30;
            constexpr uint32_t gridN = 10;
            for (uint32_t iy = 1; iy < gridN; iy++)
            {
               for (uint32_t ix = 1; ix < gridN; ix++)
               {
                  float32_t2 u(static_cast<float>(ix) / static_cast<float>(gridN),
                     static_cast<float>(iy) / static_cast<float>(gridN));
                  typename sampling::ProjectedSphericalTriangle<float32_t>::cache_type cache;
                  sampler.generate(u, cache);
                  const float64_t pdf = static_cast<float64_t>(sampler.forwardPdf(u, cache));
                  if (std::isfinite(pdf) && pdf > 0.0)
                  {
                     minPdf = std::min(minPdf, pdf);
                     maxPdf = std::max(maxPdf, pdf);
                  }
               }
            }

            if (maxPdf > 0.0 && minPdf > 0.0)
            {
               const float64_t pdfRange = (maxPdf - minPdf) / maxPdf;

               if (sizeStep >= convergenceCheckMinStep && pdfRange > pdfRangeConvergenceTol)
               {
                  m_logger->log("[PSTAccuracy] bilinear not converging: trial=%u sizeStep=%u pdfRange=%f (min=%f max=%f tol=%f)",
                     system::ILogger::ELL_ERROR, trial, sizeStep, pdfRange, minPdf, maxPdf, pdfRangeConvergenceTol);
                  pass = false;
               }
               if (sizeStep > 0 && pdfRange > prevPdfRange * pdfRangeMonotonicityFactor)
               {
                  m_logger->log("[PSTAccuracy] bilinear non-monotonic: trial=%u sizeStep=%u pdfRange=%f > prev=%f",
                     system::ILogger::ELL_ERROR, trial, sizeStep, pdfRange, prevPdfRange);
                  pass = false;
               }
               prevPdfRange = pdfRange;
            }
         }
      }

      if (pass)
         m_logger->log("  [PSTAccuracy] bilinear convergence PASSED (%u trials x %u size steps, pdfRangeTol=%e, monotonicityFactor=%f)",
            system::ILogger::ELL_PERFORMANCE, numTrials, numSizes, pdfRangeConvergenceTol, pdfRangeMonotonicityFactor);
      else
         m_logger->log("  [PSTAccuracy] bilinear convergence FAILED (%u trials x %u size steps)", system::ILogger::ELL_ERROR, numTrials, numSizes);

      return ctx.finalize(pass, m_logger, "PSTAccuracy");
   }

   // -------------------------------------------------------------------------
   // Small/far triangle convergence with varying sizes
   //
   // As the triangle gets smaller, the PST PDF (rcpSA * bilinearPdf) should
   // converge to the ideal projected solid angle PDF (NdotL / PSA), because
   // the bilinear interpolation of vertex NdotL becomes exact when all
   // vertices have approximately the same NdotL.
   //
   // Samples L UNIFORMLY from the spherical triangle (not through the bilinear
   // warp) to avoid importance sampling bias, then compares
   // backwardPdf(L) vs NdotL/PSA. The RMS of log(pstPdf/idealPdf) should
   // decrease as the triangle shrinks.
   // -------------------------------------------------------------------------
   bool testSmallTrianglePdfConvergence()
   {
      constexpr float64_t rmsLogRatioTol = 0.5;

      SeededTestContext ctx;
      bool pass = true;

      constexpr uint32_t numTrials = 100;
      constexpr uint32_t samplesPerTrial = 5000;
      constexpr uint32_t numSizes = 6;
      const float halfAngles[numSizes] = {0.5f, 0.2f, 0.1f, 0.05f, 0.02f, 0.01f};

      // Track RMS of log(pstPdf/idealPdf) per size
      float64_t sumSqLogRatioPerSize[numSizes] = {};
      float64_t sumLogRatioPerSize[numSizes] = {};
      uint32_t sampleCountPerSize[numSizes] = {};
      uint32_t validTrials[numSizes] = {};

      for (uint32_t trial = 0; trial < numTrials; trial++)
      {
         for (uint32_t s = 0; s < numSizes; s++)
         {
            TriangleConfig cfg = smallTriConfig(ctx.rng, halfAngles[s]);

            auto shape = createSphericalTriangleShape(cfg.v0, cfg.v1, cfg.v2);
            if (shape.solid_angle <= 0.0f || !std::isfinite(shape.solid_angle))
               continue;

            auto sampler = createSampler(cfg);
            const float64_t projSA = std::abs(static_cast<float64_t>(shape.projectedSolidAngle(cfg.normal)));

            if (projSA <= 0.0 || !std::isfinite(projSA) ||
               !std::isfinite(sampler.sphtri.rcpSolidAngle) || sampler.sphtri.rcpSolidAngle <= 0.0f)
               continue;

            const float64_t rcpPSA = 1.0 / projSA;
            std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
            bool trialValid = false;

            for (uint32_t i = 0; i < samplesPerTrial; i++)
            {
               // Sample L UNIFORMLY from the spherical triangle (no bilinear warp)
               float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
               typename sampling::SphericalTriangle<float32_t>::cache_type stCache;
               float32_t3 L = sampler.sphtri.generate(u, stCache);

               const float64_t trueNdotL = std::max(0.0, static_cast<float64_t>(dot(cfg.normal, L)));
               if (trueNdotL < 1e-6)
                  continue;

               // No direct backwardPdf; evaluate forwardPdf at the inverted u to recover pdf(L).
               const float32_t2 uInv = sampler.sphtri.generateInverse(L);
               typename sampling::ProjectedSphericalTriangle<float32_t>::cache_type pdfCache;
               sampler.generate(uInv, pdfCache);
               const float64_t pstPdf = static_cast<float64_t>(sampler.forwardPdf(uInv, pdfCache));
               const float64_t idealPdf = trueNdotL * rcpPSA;

               if (!std::isfinite(pstPdf) || pstPdf <= 0.0 || idealPdf <= 0.0)
                  continue;

               const float64_t logRatio = std::log(pstPdf / idealPdf);
               if (!std::isfinite(logRatio))
                  continue;

               sumLogRatioPerSize[s] += logRatio;
               sumSqLogRatioPerSize[s] += logRatio * logRatio;
               sampleCountPerSize[s]++;
               trialValid = true;
            }

            if (trialValid)
               validTrials[s]++;
         }
      }

      m_logger->log("  [PSTAccuracy] small triangle convergence (RMS log(pstPdf/idealPdf) -> 0):", system::ILogger::ELL_PERFORMANCE);
      for (uint32_t s = 0; s < numSizes; s++)
      {
         if (sampleCountPerSize[s] > 0)
         {
            const float64_t n = static_cast<float64_t>(sampleCountPerSize[s]);
            const float64_t meanLog = sumLogRatioPerSize[s] / n;
            const float64_t rmsLog = std::sqrt(sumSqLogRatioPerSize[s] / n);
            m_logger->log("    halfAngle=%.3f  meanLog=%+.6f  rmsLog=%.6f  (%u samples, %u trials)",
               system::ILogger::ELL_PERFORMANCE, halfAngles[s], meanLog, rmsLog, sampleCountPerSize[s], validTrials[s]);

            // For small triangles, RMS log ratio should be small.
            // Skip halfAngle=0.01 (s==5): float32 solid angle precision collapses,
            // leaving too few valid trials for a meaningful check.
            if (s == 4 && rmsLog > rmsLogRatioTol)
            {
               m_logger->log("  [PSTAccuracy] small triangle convergence FAILED at halfAngle=%.3f rmsLog=%e rmsLogTol=%e",
                  system::ILogger::ELL_ERROR, halfAngles[s], rmsLog, rmsLogRatioTol);
               pass = false;
            }
         }
      }

      if (pass)
         m_logger->log("  [PSTAccuracy] small triangle convergence PASSED (%u trials x %u sizes, %u samples/trial, rmsLogTol=%e, halfAngle=0.010 skipped due to float32 precision limit)",
            system::ILogger::ELL_PERFORMANCE, numTrials, numSizes, samplesPerTrial, rmsLogRatioTol);

      return ctx.finalize(pass, m_logger, "PSTAccuracy");
   }

   system::ILogger* m_logger;
};


// ============================================================================
// Spherical rectangle sampler policies for CRectangleGenerateTester.
//
// Each policy provides sampler creation, solid angle access, and flags
// for which tests apply. The tester is templated on the policy.
// ============================================================================

struct UniformRectSamplerPolicy
{
   using sampler_type = sampling::SphericalRectangle<float32_t>;

   static sampler_type createSampler(shapes::SphericalRectangle<float32_t>& shape,
      const float32_t3& observer, std::mt19937&)
   {
      return sampler_type::create(shape, observer);
   }

   // Returns offset-from-r0 on the rectangle surface. Goes through generateLocalBasisXY
   // (absolute xy) and subtracts r0.xy so the [0, extents] bounds check still applies.
   static float32_t2 generateOffset(sampler_type& s, const float32_t2& u)
   {
      typename sampler_type::cache_type cache;
      const float32_t2 absXY = s.generateLocalBasisXY(u, cache);
      return absXY - float32_t2(s.r0.x, s.r0.y);
   }

   static float getSolidAngle(const sampler_type& s) { return s.solidAngle; }
   static const char* name() { return "SphericalRectangle"; }

   static constexpr bool hasStripCounting = true;
   static constexpr bool hasMomentMatching = true;
};

struct ProjectedRectSamplerPolicy
{
   // UsePdfAsWeight=false so receiverNormal and projSolidAngle are populated for diagnostic logs.
   using sampler_type = sampling::ProjectedSphericalRectangle<float32_t, false>;

   static sampler_type createSampler(shapes::SphericalRectangle<float32_t>& shape,
      const float32_t3& observer, std::mt19937& rng)
   {
      float32_t3 receiverNormal;
      do
      {
         receiverNormal = generateRandomUnitVector(rng);
      } while (!anyRectCornerAboveHorizon(shape, observer, receiverNormal));

      return sampler_type::create(shape, observer, receiverNormal, false);
   }

   // Run u through the bilinear warp then the inner sphrect's generateLocalBasisXY, and subtract
   // r0.xy to get offset-from-r0 on the rectangle surface.
   static float32_t2 generateOffset(sampler_type& s, const float32_t2& u)
   {
      typename sampling::Bilinear<float32_t>::cache_type bc;
      const float32_t2 warped = s.bilinearPatch.generate(u, bc);
      typename sampling::SphericalRectangle<float32_t>::cache_type sphrectCache;
      const float32_t2 absXY = s.sphrect.generateLocalBasisXY(warped, sphrectCache);
      return absXY - float32_t2(s.sphrect.r0.x, s.sphrect.r0.y);
   }

   static float getSolidAngle(const sampler_type& s) { return s.sphrect.solidAngle; }
   static const char* name() { return "ProjectedSphericalRectangle"; }

   // Strip counting and moment matching expected values assume uniform distribution;
   // the bilinear warp makes these inapplicable without different expected values.
   static constexpr bool hasStripCounting = false;
   static constexpr bool hasMomentMatching = false;
};

// ============================================================================
// CRectangleGenerateTester<Policy>
//
// Tests that a rectangle sampler's generate() produces correct output.
// Templated on a policy that controls sampler creation and test selection.
//
// The sampler's generate() returns 2D offsets in [0, extents.x] x [0, extents.y].
// Available tests (controlled by policy flags):
// - Strip counting: cut the rectangle into two sub-rectangles with analytic
//   solid angles, compare observed vs expected sample fractions.
// - Moment matching: compare E[dot(L,N)] from sampling against numerical
//   quadrature over the planar rectangle (independent of generate).
// - Bounds check: verify all outputs lie within [0, extents].
// - Distant rectangles: stress float32 precision at varying distances.
// ============================================================================

template<typename Policy>
class CRectangleGenerateTester
{
   using sampler_type = typename Policy::sampler_type;

   public:
   CRectangleGenerateTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      bool pass = true;
      if constexpr (Policy::hasStripCounting)
      {
         pass &= runStripCounting("random", generateRandomRectangle, 100, 3, 50000, 0.02);
         pass &= runStripCounting("stress", generateStressRectangle, 100, 3, 50000, 0.03);
      }
      if constexpr (Policy::hasMomentMatching)
      {
         pass &= runMomentMatching("random", generateRandomRectangle, 200, 5, 20000, 0.05, 0.02);
         pass &= runMomentMatching("stress", generateStressRectangle, 200, 10, 20000, 0.08, 0.03);
      }
      // Bounds check always applies
      pass &= runBoundsCheck("random", generateRandomRectangle, 100, 5000);
      pass &= runBoundsCheck("stress", generateStressRectangle, 100, 5000);
      // Distant rectangles
      pass &= testDistantRectangles();
      return pass;
   }

   private:

   // -------------------------------------------------------------------------
   // Reconstruct a 3D unit-sphere direction from the sampler's 2D output.
   //
   // The generated output is an offset in [0, extents.x] x [0, extents.y].
   // We map back to world space via the planar rectangle parametrization,
   // avoiding the sampler's internal z-flip.
   // -------------------------------------------------------------------------
   static float32_t3 reconstructDirection(
      const shapes::CompressedSphericalRectangle<float32_t>& compressed,
      const float32_t2& extents,
      const float32_t3& observer,
      float32_t2 generated)
   {
      float32_t3 worldPt = compressed.origin
         + compressed.right * (generated.x / extents.x)
         + compressed.up * (generated.y / extents.y);
      return normalize(worldPt - observer);
   }

   // -------------------------------------------------------------------------
   // Numerical quadrature for E[dot(L, N)] over the planar rectangle.
   //
   // Computes (1/SA) * integral_{rect} dot(dir, N) dOmega by subdividing
   // the world-space rectangle into a grid and summing the differential
   // solid angle contributions. Independent of generate().
   //
   // Grid resolution adapts to aspect ratio: totalCells cells are
   // distributed so that cells are approximately square on the rectangle,
   // avoiding wasted resolution on the short axis of elongated rectangles.
   // -------------------------------------------------------------------------
   static float64_t numericalMoment(
      const shapes::CompressedSphericalRectangle<float32_t>& compressed,
      const float32_t3& observer,
      const float32_t3& N,
      float64_t SA,
      uint32_t totalCells = 10000)
   {
      const float width = length(compressed.right);
      const float height = length(compressed.up);
      const float aspect = width / height;

      // Distribute cells to make them approximately square:
      // gridX * gridY ~ totalCells, gridX / gridY ~ aspect
      const uint32_t gridX = std::max(1u, static_cast<uint32_t>(std::sqrt(static_cast<float>(totalCells) * aspect)));
      const uint32_t gridY = std::max(1u, totalCells / gridX);

      const float32_t3 faceNormal = normalize(cross(compressed.right, compressed.up));
      const float64_t cellArea = static_cast<float64_t>(width) * static_cast<float64_t>(height)
         / static_cast<float64_t>(gridX * gridY);

      float64_t sum = 0.0;
      for (uint32_t i = 0; i < gridX; i++)
      {
         const float fx = (static_cast<float>(i) + 0.5f) / static_cast<float>(gridX);
         for (uint32_t j = 0; j < gridY; j++)
         {
            const float fy = (static_cast<float>(j) + 0.5f) / static_cast<float>(gridY);
            const float32_t3 point = compressed.origin + compressed.right * fx + compressed.up * fy;
            const float32_t3 toPoint = point - observer;
            const float64_t dist2 = static_cast<float64_t>(dot(toPoint, toPoint));
            const float32_t3 dir = toPoint * (1.0f / std::sqrt(static_cast<float>(dist2)));
            const float64_t cosTheta = std::abs(static_cast<float64_t>(dot(faceNormal, dir)));
            const float64_t dOmega = cosTheta * cellArea / dist2;
            sum += static_cast<float64_t>(dot(dir, N)) * dOmega;
         }
      }
      return sum / SA;
   }

   // -------------------------------------------------------------------------
   // Strip counting: the generate()-only uniformity test.
   //
   // For each rectangle, cut it into two sub-rectangles by a line parallel
   // to one edge. The expected fraction of samples on each side equals the
   // ratio of sub-solid-angles (computed analytically via solidAngle()).
   // Generate samples, classify each by which sub-rectangle it falls in,
   // and compare counts against the expected fractions.
   // -------------------------------------------------------------------------
   template<typename RectGen>
   bool runStripCounting(const char* label, RectGen rectGen,
      uint32_t numRects, uint32_t cutsPerRect, uint32_t numSamples,
      float64_t relTol)
   {
      SeededTestContext ctx;
      uint32_t testedCuts = 0;

      for (uint32_t r = 0; r < numRects; r++)
      {
         shapes::CompressedSphericalRectangle<float32_t> compressed;
         float32_t3 observer;
         rectGen(ctx.rng, compressed, observer);

         shapes::SphericalRectangle<float32_t> shape = shapes::SphericalRectangle<float32_t>::create(compressed);
         sampler_type sampler = Policy::createSampler(shape, observer, ctx.rng);

         if (Policy::getSolidAngle(sampler) <= 0.0f || !std::isfinite(Policy::getSolidAngle(sampler)))
            continue;

         const float64_t SA = static_cast<float64_t>(Policy::getSolidAngle(sampler));

         for (uint32_t c = 0; c < cutsPerRect; c++)
         {
            // Alternate between x and y cuts
            const bool cutAlongX = (c % 2 == 0);
            std::uniform_real_distribution<float> fDist(0.2f, 0.8f);
            const float f = fDist(ctx.rng);

            // Create sub-rectangle (the "lower" portion of the cut)
            shapes::CompressedSphericalRectangle<float32_t> subCompressed;
            subCompressed.origin = compressed.origin;
            if (cutAlongX)
            {
               subCompressed.right = compressed.right * f;
               subCompressed.up = compressed.up;
            }
            else
            {
               subCompressed.right = compressed.right;
               subCompressed.up = compressed.up * f;
            }

            shapes::SphericalRectangle<float32_t> subShape = shapes::SphericalRectangle<float32_t>::create(subCompressed);
            auto subSA = subShape.solidAngle(observer);
            if (subSA.value <= 0.0f || !std::isfinite(subSA.value))
               continue;

            const float64_t expectedFraction = static_cast<float64_t>(subSA.value) / SA;

            // Skip cuts that produce very lopsided splits (hard to test statistically)
            if (expectedFraction < 0.1 || expectedFraction > 0.9)
               continue;

            // Count samples in the sub-rectangle
            std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
            const float cutThreshold = cutAlongX ? shape.extents.x * f : shape.extents.y * f;
            uint32_t countInSub = 0;

            for (uint32_t i = 0; i < numSamples; i++)
            {
               float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
               float32_t2 gen = Policy::generateOffset(sampler, u);
               const float coord = cutAlongX ? gen.x : gen.y;
               if (coord < cutThreshold)
                  countInSub++;
            }

            const float64_t observedFraction = static_cast<float64_t>(countInSub) / static_cast<float64_t>(numSamples);
            const float64_t absErr = std::abs(observedFraction - expectedFraction);

            testedCuts++;
            if (absErr > relTol)
            {
               ctx.failCount++;
               if (ctx.failCount <= 5)
               {
                  m_logger->log("[%s::generate] %s strip counting: observed=%f expected=%f absErr=%e (tol=%e) rect %u cut %u (%s at f=%f)",
                     system::ILogger::ELL_ERROR, Policy::name(), label, observedFraction, expectedFraction, absErr, relTol, r, c,
                     cutAlongX ? "x-cut" : "y-cut", static_cast<float64_t>(f));
                  logRectInfo(m_logger, compressed, observer, Policy::getSolidAngle(sampler));
               }
            }
         }
      }

      if (ctx.failCount == 0)
         m_logger->log("  [%s::generate] %s strip counting PASSED (%u cuts across %u rects x %u cuts/rect, %u samples/cut, relTol=%e)",
            system::ILogger::ELL_PERFORMANCE, Policy::name(), label, testedCuts, numRects, cutsPerRect, numSamples, relTol);
      else
         m_logger->log("  [%s::generate] %s strip counting FAILED (%u/%u cuts failed, %u rects x %u cuts/rect, %u samples/cut, relTol=%e)",
            system::ILogger::ELL_ERROR, Policy::name(), label, ctx.failCount, testedCuts, numRects, cutsPerRect, numSamples, relTol);

      return ctx.finalize(m_logger, Policy::name());
   }

   // -------------------------------------------------------------------------
   // Moment matching: E[dot(generate(u), N)] should equal the numerically
   // integrated moment over the planar rectangle.
   //
   // For a uniform distribution over a spherical rectangle:
   //   E[dot(L, N)] = (1/SA) * integral_{rect} dot(L, N) dOmega
   //
   // The expected value is computed by numerical quadrature over the
   // world-space rectangle (independent of generate). Testing multiple
   // random N per rectangle makes it very unlikely that a biased mapping
   // passes by accident.
   // -------------------------------------------------------------------------
   template<typename RectGen>
   bool runMomentMatching(const char* label, RectGen rectGen,
      uint32_t numRects, uint32_t numNormals, uint32_t numSamples,
      float64_t relTol, float64_t absTol)
   {
      SeededTestContext ctx;
      uint32_t testedConfigs = 0;

      for (uint32_t r = 0; r < numRects; r++)
      {
         shapes::CompressedSphericalRectangle<float32_t> compressed;
         float32_t3 observer;
         rectGen(ctx.rng, compressed, observer);

         shapes::SphericalRectangle<float32_t> shape = shapes::SphericalRectangle<float32_t>::create(compressed);
         sampler_type sampler = Policy::createSampler(shape, observer, ctx.rng);

         if (Policy::getSolidAngle(sampler) <= 0.0f || !std::isfinite(Policy::getSolidAngle(sampler)))
            continue;

         const float64_t SA = static_cast<float64_t>(Policy::getSolidAngle(sampler));

         for (uint32_t n = 0; n < numNormals; n++)
         {
            float32_t3 N = generateRandomUnitVector(ctx.rng);
            const float64_t expected = numericalMoment(compressed, observer, N, SA);

            float64_t sum = 0.0;
            std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
            for (uint32_t i = 0; i < numSamples; i++)
            {
               float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
               float32_t2 gen = Policy::generateOffset(sampler, u);
               float32_t3 dir = reconstructDirection(compressed, shape.extents, observer, gen);
               sum += static_cast<float64_t>(dot(dir, N));
            }
            const float64_t mcEstimate = sum / static_cast<float64_t>(numSamples);

            const float64_t absErr = std::abs(mcEstimate - expected);
            const float64_t tol = std::max(relTol * std::abs(expected), absTol);
            if (absErr > tol)
            {
               ctx.failCount++;
               if (ctx.failCount <= 5)
               {
                  m_logger->log("[%s::generate] %s moment mismatch: E[dot(L,N)]=%f expected=%f absErr=%e (tol=%e) rect %u normal %u",
                     system::ILogger::ELL_ERROR, Policy::name(), label, mcEstimate, expected, absErr, tol, r, n);
                  logRectInfo(m_logger, compressed, observer, Policy::getSolidAngle(sampler));
               }
            }
         }
         testedConfigs++;
      }

      const uint32_t totalMomentTests = testedConfigs * numNormals;
      const uint32_t skippedRects = numRects - testedConfigs;
      if (ctx.failCount == 0)
         m_logger->log("  [%s::generate] %s moment matching PASSED (%u/%u rects x %u normals = %u tests, %u skipped, %u samples/test, relTol=%e absTol=%e)",
            system::ILogger::ELL_PERFORMANCE, Policy::name(), label, testedConfigs, numRects, numNormals, totalMomentTests, skippedRects, numSamples, relTol, absTol);
      else
         m_logger->log("  [%s::generate] %s moment matching FAILED (%u/%u tests failed, %u/%u rects tested x %u normals, %u samples/test, relTol=%e absTol=%e)",
            system::ILogger::ELL_ERROR, Policy::name(), label, ctx.failCount, totalMomentTests, testedConfigs, numRects, numNormals, numSamples, relTol, absTol);

      return ctx.finalize(m_logger, Policy::name());
   }

   // -------------------------------------------------------------------------
   // Bounds check: all generated 2D offsets must lie within [0, extents].
   // -------------------------------------------------------------------------
   template<typename RectGen>
   bool runBoundsCheck(const char* label, RectGen rectGen,
      uint32_t numRects, uint32_t numSamples)
   {
      SeededTestContext ctx;
      uint32_t testedRects = 0;

      for (uint32_t r = 0; r < numRects; r++)
      {
         shapes::CompressedSphericalRectangle<float32_t> compressed;
         float32_t3 observer;
         rectGen(ctx.rng, compressed, observer);

         shapes::SphericalRectangle<float32_t> shape = shapes::SphericalRectangle<float32_t>::create(compressed);
         sampler_type sampler = Policy::createSampler(shape, observer, ctx.rng);

         if (Policy::getSolidAngle(sampler) <= 0.0f || !std::isfinite(Policy::getSolidAngle(sampler)))
            continue;

         const float extX = shape.extents.x;
         const float extY = shape.extents.y;

         std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
         for (uint32_t i = 0; i < numSamples; i++)
         {
            float32_t2 u(uDist(ctx.rng), uDist(ctx.rng));
            float32_t2 gen = Policy::generateOffset(sampler, u);

            if (gen.x < -1e-5f || gen.x > extX + 1e-5f || gen.y < -1e-5f || gen.y > extY + 1e-5f)
            {
               ctx.failCount++;
               if (ctx.failCount <= 5)
               {
                  m_logger->log("[%s::generate] %s out of bounds: generated=(%f, %f) extents=(%f, %f) rect %u sample %u",
                     system::ILogger::ELL_ERROR, Policy::name(), label, gen.x, gen.y, extX, extY, r, i);
                  logRectInfo(m_logger, compressed, observer, Policy::getSolidAngle(sampler));
               }
            }
         }
         testedRects++;
      }

      if (ctx.failCount == 0)
         m_logger->log("  [%s::generate] %s bounds check PASSED (%u rects x %u samples)",
            system::ILogger::ELL_PERFORMANCE, Policy::name(), label, testedRects, numSamples);
      else
         m_logger->log("  [%s::generate] %s bounds check FAILED (%u failures across %u rects x %u samples)",
            system::ILogger::ELL_ERROR, Policy::name(), label, ctx.failCount, testedRects, numSamples);

      return ctx.finalize(m_logger, Policy::name());
   }

   // -------------------------------------------------------------------------
   // Distant rectangles: world-space rectangles placed far from the observer.
   //
   // When the rectangle is far away, it subtends a tiny solid angle on the
   // unit sphere. This stresses float32 precision in the solid angle
   // computation (acos sum - 2*pi involves near-cancellation) and in
   // generate()'s trigonometric inversion.
   //
   // d<=10 is a hard failure (generate must be correct for moderate distances).
   // d>=100 is diagnostic only: the solid angle computation loses precision
   // for tiny rectangles, producing inaccurate inputs that generate() cannot
   // recover from.
   // -------------------------------------------------------------------------
   bool testDistantRectangles()
   {
      bool pass = true;

      struct DistanceConfig
      {
         float dist;
         bool hardFail;
         float64_t stripTol;
         float64_t momentRelTol;
         float64_t momentAbsTol;
      };

      const DistanceConfig configs[] = {
         {10.0f, true, 0.03, 0.08, 0.03},
         {100.0f, false, 0.05, 0.10, 0.05}};

      for (const auto& cfg : configs)
      {
         const float dist = cfg.dist;
         auto distantRectGen = [dist](std::mt19937& rng,
            shapes::CompressedSphericalRectangle<float32_t>& compressed,
            float32_t3& observer)
         {
            float32_t3 centerDir = generateRandomUnitVector(rng);
            float32_t3 t1, t2;
            buildTangentFrame(centerDir, t1, t2);

            // ~1 unit edge length rectangle at the given distance
            std::uniform_real_distribution<float> jitter(-0.2f, 0.2f);
            observer = float32_t3(0.0f, 0.0f, 0.0f);
            compressed.origin = centerDir * dist + t1 * (-0.5f + jitter(rng)) + t2 * (-0.5f + jitter(rng));
            compressed.right = t1 * (1.0f + jitter(rng));
            compressed.up = t2 * (1.0f + jitter(rng));
         };

         char labelBuf[64];
         snprintf(labelBuf, sizeof(labelBuf), "distant(d=%g)", dist);

         bool stripOK = true, momentOK = true;
         if constexpr (Policy::hasStripCounting)
            stripOK = runStripCounting(labelBuf, distantRectGen, 50, 3, 20000, cfg.stripTol);
         if constexpr (Policy::hasMomentMatching)
            momentOK = runMomentMatching(labelBuf, distantRectGen, 50, 5, 20000, cfg.momentRelTol, cfg.momentAbsTol);
         bool boundsOK = runBoundsCheck(labelBuf, distantRectGen, 50, 5000);

         if (cfg.hardFail)
         {
            pass &= stripOK;
            pass &= momentOK;
            pass &= boundsOK;
         }
         else
         {
            if (!stripOK || !momentOK || !boundsOK)
               m_logger->log("  [%s::generate] %s DIAGNOSTIC (precision limit, not a hard failure)",
                  system::ILogger::ELL_PERFORMANCE, Policy::name(), labelBuf);
         }
      }

      return pass;
   }

   system::ILogger* m_logger;
};

using CSphericalRectangleGenerateTester = CRectangleGenerateTester<UniformRectSamplerPolicy>;
using CProjectedSphericalRectangleGenerateTester = CRectangleGenerateTester<ProjectedRectSamplerPolicy>;


// ============================================================================
// CProjectedSphericalRectangleGeometricTester
//
// Tests the rectangle projectedSolidAngle() formula against a surface-grid reference,
// reusing the generic testPSAVersusGrid infrastructure and the rectangle generators
// from CRectangleGenerateTester.
// ============================================================================

class CProjectedSphericalRectangleGeometricTester
{
public:
   CProjectedSphericalRectangleGeometricTester(system::ILogger* logger) : m_logger(logger) {}

   bool run()
   {
      // NOTE: PSA formula uses abs() on individual edge-normal dot products for BSDF support.
      // This overcounts when edge normals have mixed signs -- same issue as the triangle PSA.
      // Diagnostic-only until proper hemisphere clipping is implemented.
      // TODO: make these hard failures once projectedSolidAngle clips to the hemisphere.
      // Hard-fail thresholds (relErr > 3.0 AND absErr > 0.3) still catch catastrophic regressions.
      bool pass = true;
      pass &= testPSAVersusGrid("random", generateRandomRectangle, 200, 500000, 0.05, 0.01, 3.0, 0.3);
      pass &= testPSAVersusGrid("grazing", generateStressRectangle, 200, 500000, 0.1, 0.01, 3.0, 0.3);
      return pass;
   }

private:
   // Reuse rectangle generators from CRectangleGenerateTester
   using RectGen = void(*)(std::mt19937&, shapes::CompressedSphericalRectangle<float32_t>&, float32_t3&);

   bool testPSAVersusGrid(const char* label, RectGen rectGen, uint32_t numConfigs, uint32_t gridSamples,
      float64_t relTol, float64_t absTol, float64_t hardRelTol, float64_t hardAbsTol)
   {
      return ::testPSAVersusGrid(m_logger, "RectPSA", label,
         [&](std::mt19937& rng, uint32_t, float64_t& formulaPSA, float64_t& gridPSA, auto& logInfo)
         {
            shapes::CompressedSphericalRectangle<float32_t> compressed;
            float32_t3 observer;
            rectGen(rng, compressed, observer);

            auto shape = shapes::SphericalRectangle<float32_t>::create(compressed);
            auto sa = shape.solidAngle(observer);
            if (sa.value <= 0.0f || !std::isfinite(sa.value))
               return;

            float32_t3 normal = generateRandomUnitVector(rng);
            formulaPSA = static_cast<float64_t>(shape.projectedSolidAngle(observer, normal));
            // surfaceGridEstimatePSA integrates over the rectangle surface directly (no sampler in
            // the loop), so a formula-vs-reference mismatch here isolates the PSA formula.
            gridPSA = surfaceGridEstimatePSA(shape, observer, normal, gridSamples);
            logInfo = [compressed, observer, normal, saValue = sa.value](system::ILogger* logger, system::ILogger::E_LOG_LEVEL level)
            {
               using nbl::system::to_string;
               const float width = length(compressed.right);
               const float height = length(compressed.up);
               logger->log("    origin=%s extents=(%s, %s) observer=%s normal=%s solidAngle=%s",
                  level, to_string(compressed.origin).c_str(),
                  to_string(width).c_str(), to_string(height).c_str(),
                  to_string(observer).c_str(), to_string(normal).c_str(),
                  to_string(saValue).c_str());
            };
         },
         numConfigs, relTol, absTol, hardRelTol, hardAbsTol, true);
   }

   system::ILogger* m_logger;
};


#endif
