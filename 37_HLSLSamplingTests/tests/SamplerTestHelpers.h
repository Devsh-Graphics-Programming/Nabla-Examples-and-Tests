#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_TEST_HELPERS_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_TEST_HELPERS_INCLUDED_

#include <nbl/builtin/hlsl/sampling/uniform_spheres.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_triangle.hlsl>

// ============================================================================
// Declarative field verification helpers
//
// FieldCheck and PdfCheck are aggregate types with C++20 CTAD.
// VERIFY_FIELDS and VERIFY_PDFS_POSITIVE are macros that expand fold
// expressions calling verifyTestValue/printTestFail (protected ITester members)
// within the calling class context.
// ============================================================================

template<typename R, typename T>
struct FieldCheck
{
   const char* name;
   T R::* field;
   float64_t relTol;
   float64_t absTol;
};

template<typename R>
struct PdfCheck
{
   const char* name;
   float32_t R::* field;
};

// Verify expected.*field vs actual.*field for each FieldCheck.
// Must be called from within a method that has access to verifyTestValue.
#define VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType, ...) \
   do \
   { \
      auto _checks = std::make_tuple(__VA_ARGS__); \
      std::apply([&](const auto&... c) { ((pass &= verifyTestValue(c.name, (expected).*c.field, (actual).*c.field, \
                                              iteration, seed, testType, c.relTol, c.absTol)), \
                                            ...); }, _checks); \
   } while (0)

// Check that each PDF field is positive and finite.
// Must be called from within a method that has access to printTestFail.
#define VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType, ...) \
   do \
   { \
      auto _pdfChecks = std::make_tuple(__VA_ARGS__); \
      std::apply([&](const auto&... c) { (([&] { \
                                            if (!((actual).*c.field > 0.0f) || !std::isfinite((actual).*c.field)) \
                                            { \
                                               pass = false; \
                                               printTestFail(std::string(c.name) + " (positive & finite)", \
                                                  1.0f, (actual).*c.field, iteration, seed, testType, 0.0, 0.0); \
                                            } \
                                         }()), \
                                            ...); }, _pdfChecks); \
   } while (0)

// ============================================================================
// Shared geometry helpers
//
// Used by CSphericalTriangleTester, CProjectedSphericalTriangleTester, and
// CSphericalRectangleTester. Previously duplicated across all three.
// ============================================================================

inline nbl::hlsl::float32_t3 generateRandomUnitVector(std::mt19937& rng)
{
   std::uniform_real_distribution<float> dist(0.0f, 1.0f);
   nbl::hlsl::float32_t2 u(dist(rng), dist(rng));
   nbl::hlsl::sampling::UniformSphere<float>::cache_type cache;
   return nbl::hlsl::sampling::UniformSphere<float>::generate(u, cache);
}

inline bool isValidSphericalTriangle(nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
{
   using namespace nbl::hlsl;

   constexpr float sinSqThreshold = 0.001f; // sin^2(theta) > 0.001, i.e. sin(theta) > 0.03, ~1.7 degrees

   const float d01 = dot(v0, v1);
   const float d12 = dot(v1, v2);
   const float d20 = dot(v2, v0);

   if ((1.f - d01 * d01) < sinSqThreshold)
      return false;
   if ((1.f - d12 * d12) < sinSqThreshold)
      return false;
   if ((1.f - d20 * d20) < sinSqThreshold)
      return false;

   constexpr float tripleThreshold = 0.08f;
   return abs(dot(v0, cross(v1, v2))) > tripleThreshold;
}

// ============================================================================
// Shared helpers for property test configs
// ============================================================================

// Uniform random domain in [0,1)^d
template<typename T>
inline T uniformRandomDomain(std::mt19937& rng);

template<>
inline nbl::hlsl::float32_t uniformRandomDomain<nbl::hlsl::float32_t>(std::mt19937& rng)
{
   return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
}

template<>
inline nbl::hlsl::float32_t2 uniformRandomDomain<nbl::hlsl::float32_t2>(std::mt19937& rng)
{
   std::uniform_real_distribution<float> d(0.0f, 1.0f);
   return nbl::hlsl::float32_t2(d(rng), d(rng));
}

template<>
inline nbl::hlsl::float32_t3 uniformRandomDomain<nbl::hlsl::float32_t3>(std::mt19937& rng)
{
   std::uniform_real_distribution<float> d(0.0f, 1.0f);
   return nbl::hlsl::float32_t3(d(rng), d(rng), d(rng));
}

// 1D grid integration of backwardPdf over [0,1]
inline float64_t gridIntegratePdf1D(const auto& sampler, uint32_t N = 100000)
{
   float64_t sum = 0.0;
   for (uint32_t i = 0; i < N; i++)
   {
      const float x = (static_cast<float>(i) + 0.5f) / static_cast<float>(N);
      sum += static_cast<float64_t>(sampler.backwardPdf(x));
   }
   return sum / static_cast<float64_t>(N);
}

// 2D grid integration of backwardPdf over [0,1]^2
inline float64_t gridIntegratePdf2D(const auto& sampler, uint32_t N = 1000)
{
   float64_t sum = 0.0;
   const float64_t cellArea = 1.0 / static_cast<float64_t>(N * N);
   for (uint32_t iy = 0; iy < N; iy++)
   {
      const float py = (static_cast<float>(iy) + 0.5f) / static_cast<float>(N);
      for (uint32_t ix = 0; ix < N; ix++)
      {
         const float px = (static_cast<float>(ix) + 0.5f) / static_cast<float>(N);
         sum += static_cast<float64_t>(sampler.backwardPdf(nbl::hlsl::float32_t2(px, py)));
      }
   }
   return sum * cellArea;
}

// True if at least one vertex has positive NdotL (triangle partially above horizon).
inline bool anyVertexAboveHorizon(nbl::hlsl::float32_t3 normal, nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
{
   return nbl::hlsl::dot(normal, v0) > 0.0f || nbl::hlsl::dot(normal, v1) > 0.0f || nbl::hlsl::dot(normal, v2) > 0.0f;
}

// True if all vertices have positive NdotL (entire triangle above horizon).
inline bool allVerticesAboveHorizon(nbl::hlsl::float32_t3 normal, nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
{
   return nbl::hlsl::dot(normal, v0) > 0.0f && nbl::hlsl::dot(normal, v1) > 0.0f && nbl::hlsl::dot(normal, v2) > 0.0f;
}

// Generate three random unit vectors forming a valid spherical triangle
inline void generateRandomTriangleVertices(std::mt19937& rng, nbl::hlsl::float32_t3& v0, nbl::hlsl::float32_t3& v1, nbl::hlsl::float32_t3& v2)
{
   do
   {
      v0 = generateRandomUnitVector(rng);
      v1 = generateRandomUnitVector(rng);
      v2 = generateRandomUnitVector(rng);
   } while (!isValidSphericalTriangle(v0, v1, v2));
}

// Build an orthonormal tangent frame from a direction vector.
// Returns t1 perpendicular to dir; t2 = cross(dir, t1).
inline void buildTangentFrame(nbl::hlsl::float32_t3 dir, nbl::hlsl::float32_t3& t1, nbl::hlsl::float32_t3& t2)
{
   using namespace nbl::hlsl;
   float32_t3 rawCross = cross(dir, float32_t3(0, 0, 1));
   if (length(rawCross) < 0.1f)
      rawCross = cross(dir, float32_t3(1, 0, 0));
   t1 = normalize(rawCross);
   t2 = normalize(cross(dir, t1));
}

// Generate a small equilateral triangle on the unit sphere around baseDir with given half-angle.
// Also generates a random normal with decent projection onto the triangle.
inline void generateSmallTriangle(std::mt19937& rng, float halfAngle,
   nbl::hlsl::float32_t3& v0, nbl::hlsl::float32_t3& v1, nbl::hlsl::float32_t3& v2,
   nbl::hlsl::float32_t3& baseDir, nbl::hlsl::float32_t3& normal)
{
   using namespace nbl::hlsl;
   baseDir = generateRandomUnitVector(rng);
   float32_t3 t1, t2;
   buildTangentFrame(baseDir, t1, t2);
   v0 = normalize(baseDir + t1 * halfAngle);
   v1 = normalize(baseDir - t1 * (halfAngle * 0.5f) + t2 * (halfAngle * 0.866f));
   v2 = normalize(baseDir - t1 * (halfAngle * 0.5f) - t2 * (halfAngle * 0.866f));
   normal = generateRandomUnitVector(rng);
   if (dot(normal, baseDir) < 0.1f)
      normal = normalize(normal + baseDir * 2.0f);
}

// Generate a stress-test triangle: thin/elongated, nearly coplanar, or one short edge.
// Falls back to a random triangle if the generated geometry is degenerate.
inline void generateStressTriangleVertices(std::mt19937& rng, nbl::hlsl::float32_t3& v0, nbl::hlsl::float32_t3& v1, nbl::hlsl::float32_t3& v2)
{
   using namespace nbl::hlsl;
   std::uniform_real_distribution<float> angleDist(0.0f, 1.0f);
   std::uniform_int_distribution<int> caseDist(0, 2);
   switch (caseDist(rng))
   {
      case 0: // Thin/elongated
         {
            float32_t3 base = generateRandomUnitVector(rng);
            float32_t3 t1, t2;
            buildTangentFrame(base, t1, t2);
            float spread = 0.15f + angleDist(rng) * 0.2f;
            v0 = normalize(base + t1 * spread);
            v1 = normalize(base - t1 * spread);
            float far_ = 0.8f + angleDist(rng) * 0.8f;
            v2 = normalize(base * std::cos(far_) + t2 * std::sin(far_));
            break;
         }
      case 1: // Nearly coplanar
         {
            float32_t3 pole = generateRandomUnitVector(rng);
            float32_t3 t1, t2;
            buildTangentFrame(pole, t1, t2);
            float offset = 0.05f + angleDist(rng) * 0.1f;
            float a1 = angleDist(rng) * 6.2832f;
            float a2 = a1 + 0.8f + angleDist(rng);
            float a3 = a2 + 0.8f + angleDist(rng);
            v0 = normalize(t1 * std::cos(a1) + t2 * std::sin(a1) + pole * offset);
            v1 = normalize(t1 * std::cos(a2) + t2 * std::sin(a2) - pole * offset * 0.5f);
            v2 = normalize(t1 * std::cos(a3) + t2 * std::sin(a3) + pole * offset * 0.3f);
            break;
         }
      default: // One short edge
         {
            float32_t3 base = generateRandomUnitVector(rng);
            float32_t3 t1, t2;
            buildTangentFrame(base, t1, t2);
            float shortAngle = 0.32f + angleDist(rng) * 0.1f;
            v0 = normalize(base + t1 * shortAngle * 0.5f);
            v1 = normalize(base - t1 * shortAngle * 0.5f);
            v2 = normalize(t2 + base * (0.3f + angleDist(rng) * 0.5f));
            break;
         }
   }
   if (!isValidSphericalTriangle(v0, v1, v2))
      generateRandomTriangleVertices(rng, v0, v1, v2);
}

// Generate an equilateral triangle on the unit sphere at colatitude theta from the north pole.
// Vertices are separated by 120 degrees in azimuth.
inline void makeEquilateralTriangle(float64_t theta, nbl::hlsl::float32_t3 verts[3])
{
   using namespace nbl::hlsl;
   const float32_t st = static_cast<float32_t>(std::sin(theta));
   const float32_t ct = static_cast<float32_t>(std::cos(theta));
   constexpr float64_t twoPiOver3 = 2.0 * numbers::pi<float64_t> / 3.0;
   verts[0] = float32_t3(st, 0.0f, ct);
   verts[1] = float32_t3(static_cast<float>(st * std::cos(twoPiOver3)),
      static_cast<float>(st * std::sin(twoPiOver3)), ct);
   verts[2] = float32_t3(static_cast<float>(st * std::cos(2.0 * twoPiOver3)),
      static_cast<float>(st * std::sin(2.0 * twoPiOver3)), ct);
}

// Monte Carlo estimate of projected solid angle: E[dot(L, normal)] * solidAngle.
// Samples L uniformly from the spherical triangle.
inline float64_t mcEstimatePSA(const nbl::hlsl::shapes::SphericalTriangle<nbl::hlsl::float32_t>& shape, nbl::hlsl::float32_t3 normal, uint32_t N, std::mt19937& rng)
{
   using namespace nbl::hlsl;
   auto sampler = sampling::SphericalTriangle<float32_t>::create(shape);
   std::uniform_real_distribution<float> uDist(0.0f, 1.0f);
   float64_t sum = 0.0;
   for (uint32_t i = 0; i < N; i++)
   {
      float32_t2 u(uDist(rng), uDist(rng));
      typename sampling::SphericalTriangle<float32_t>::cache_type cache;
      float32_t3 L = sampler.generate(u, cache);
      sum += static_cast<float64_t>(dot(normal, L));
   }
   return sum / static_cast<float64_t>(N) * static_cast<float64_t>(shape.solid_angle);
}

// Bundles seed + rng + failCount for randomized property tests.
// Use finalize() after logging your PASSED/FAILED summary to append
// the "reproduce with seed" line on failure and return the result.
struct SeededTestContext
{
   const uint32_t seed;
   std::mt19937 rng;
   uint32_t failCount = 0;

   SeededTestContext() : seed(std::random_device {}()), rng(seed) {}

   // Log "reproduce with seed" if failCount > 0, return failCount == 0
   bool finalize(nbl::system::ILogger* logger, const char* tag) const
   {
      if (failCount > 0)
      {
         logger->log("  [%s] reproduce with seed=%u", nbl::system::ILogger::ELL_ERROR, tag, seed);
         return false;
      }
      return true;
   }

   // Log "reproduce with seed" if !passed, return passed
   bool finalize(bool passed, nbl::system::ILogger* logger, const char* tag) const
   {
      if (!passed)
      {
         logger->log("  [%s] reproduce with seed=%u", nbl::system::ILogger::ELL_ERROR, tag, seed);
         return false;
      }
      return true;
   }
};

// Create a shapes::SphericalTriangle from three unit sphere vertices
inline nbl::hlsl::shapes::SphericalTriangle<nbl::hlsl::float32_t> createSphericalTriangleShape(nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
{
   const nbl::hlsl::float32_t3 verts[3] = {v0, v1, v2};
   return nbl::hlsl::shapes::SphericalTriangle<nbl::hlsl::float32_t>::createFromUnitSphereVertices(verts);
}

// ============================================================================
// Diagnostic logging on test failure
//
// Log the input configuration that caused a failed test iteration.
// Uses requires-clauses to detect available fields so a single call
// site works for all input types.
// ============================================================================

template<typename InputValues>
inline void logFailedInput(nbl::system::ILogger* logger, const InputValues& input)
{
   using namespace nbl::system;

   std::stringstream ss;
   ss << "  ";

   // Spherical triangle vertices
   if constexpr (requires { input.vertex0; input.vertex1; input.vertex2; })
   {
      ss << "v0=" << to_string(input.vertex0)
         << " v1=" << to_string(input.vertex1)
         << " v2=" << to_string(input.vertex2);

      auto shape = createSphericalTriangleShape(input.vertex0, input.vertex1, input.vertex2);
      ss << " solidAngle=" << to_string(shape.solid_angle)
         << " rcpSolidAngle=" << to_string(1.0f / shape.solid_angle);
   }

   // Receiver normal (projected spherical triangle)
   if constexpr (requires { input.receiverNormal; })
      ss << " normal=" << to_string(input.receiverNormal);

   // Spherical rectangle geometry
   if constexpr (requires { input.observer; input.rectOrigin; input.right; input.up; })
   {
      ss << "observer=" << to_string(input.observer)
         << " rectOrigin=" << to_string(input.rectOrigin)
         << " right=" << to_string(input.right)
         << " up=" << to_string(input.up);
   }

   // Linear/bilinear coefficients
   if constexpr (requires { input.coeffs; })
      ss << "coeffs=" << to_string(input.coeffs);
   if constexpr (requires { input.bilinearCoeffs; })
      ss << "bilinearCoeffs=" << to_string(input.bilinearCoeffs);

   // Domain sample (all inputs have u)
   if constexpr (requires { input.u; })
      ss << " u=" << to_string(input.u);

   logger->log("%s", ILogger::ELL_ERROR, ss.str().c_str());
}

// Log triangle geometry info on property test failure (raw vertices, no input struct)
inline void logTriangleInfo(nbl::system::ILogger* logger, nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2)
{
   using namespace nbl::system;
   auto shape = createSphericalTriangleShape(v0, v1, v2);
   logger->log("    v0=%s v1=%s v2=%s solidAngle=%s",
      ILogger::ELL_ERROR,
      to_string(v0).c_str(), to_string(v1).c_str(), to_string(v2).c_str(),
      to_string(shape.solid_angle).c_str());
}

inline void logTriangleInfo(nbl::system::ILogger* logger, nbl::hlsl::float32_t3 v0, nbl::hlsl::float32_t3 v1, nbl::hlsl::float32_t3 v2, nbl::hlsl::float32_t3 normal)
{
   using namespace nbl::system;
   auto shape = createSphericalTriangleShape(v0, v1, v2);
   logger->log("    v0=%s v1=%s v2=%s normal=%s solidAngle=%s",
      ILogger::ELL_ERROR,
      to_string(v0).c_str(), to_string(v1).c_str(), to_string(v2).c_str(),
      to_string(normal).c_str(), to_string(shape.solid_angle).c_str());
}

#endif
