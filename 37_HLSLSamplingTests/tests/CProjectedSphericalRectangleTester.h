#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_RECTANGLE_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_C_PROJECTED_SPHERICAL_RECTANGLE_TESTER_INCLUDED_

#include "nbl/examples/examples.hpp"
#include "app_resources/common/projected_spherical_rectangle.hlsl"
#include "nbl/examples/Tester/ITester.h"
#include "SamplerTestHelpers.h"

#include <nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl>
#include <nbl/builtin/hlsl/shapes/spherical_rectangle.hlsl>

class CProjectedSphericalRectangleTester final : public ITester<ProjectedSphericalRectangleInputValues, ProjectedSphericalRectangleTestResults, ProjectedSphericalRectangleTestExecutor>
{
   using base_t = ITester<ProjectedSphericalRectangleInputValues, ProjectedSphericalRectangleTestResults, ProjectedSphericalRectangleTestExecutor>;
   using R = ProjectedSphericalRectangleTestResults;

   public:
   CProjectedSphericalRectangleTester(const uint32_t testBatchCount) : base_t(testBatchCount, WORKGROUP_SIZE) {}

   private:
   ProjectedSphericalRectangleInputValues generateInputTestValues() override
   {
      std::uniform_real_distribution<float> uDist(0.0f, 1.0f);

      nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t> compressed;
      nbl::hlsl::float32_t3 observer;
      generateRandomRectangle(getRandomEngine(), compressed, observer);

      ProjectedSphericalRectangleInputValues input;
      input.observer = observer;
      input.rectOrigin = compressed.origin;
      input.right = compressed.right;
      input.up = compressed.up;

      auto shape = nbl::hlsl::shapes::SphericalRectangle<nbl::hlsl::float32_t>::create(compressed);

      // Ensure the receiver normal has positive projection onto at least one vertex,
      // otherwise the projected solid angle is zero and the bilinear patch is degenerate (NaN PDFs).
      do
      {
         input.receiverNormal = generateRandomUnitVector(getRandomEngine());
      } while (!anyRectCornerAboveHorizon(shape, input.observer, input.receiverNormal));
      input.receiverWasBSDF = 0u;
      input.u = nbl::hlsl::float32_t2(uDist(getRandomEngine()), uDist(getRandomEngine()));
      m_inputs.push_back(input);
      return input;
   }

   ProjectedSphericalRectangleTestResults determineExpectedResults(const ProjectedSphericalRectangleInputValues& input) override
   {
      ProjectedSphericalRectangleTestResults expected;
      ProjectedSphericalRectangleTestExecutor executor;
      executor(input, expected);
      return expected;
   }

   bool verifyTestResults(const ProjectedSphericalRectangleTestResults& expected, const ProjectedSphericalRectangleTestResults& actual,
      const size_t iteration, const uint32_t seed, TestType testType) override
   {
      bool pass = true;
      // `backwardWeight` takes a 3D direction; `surfaceOffset` is reconstructed in the executor
      // (bilinear warp + sphrect.generateLocalBasisXY - r0) so the [0, extents] bounds check and
      // the generate-vs-referenceDirection consistency check still apply.
      VERIFY_FIELDS(pass, expected, actual, iteration, seed, testType,
         FieldCheck {"ProjectedSphericalRectangle::generate",              &R::generated,     2e-2, 1e-2},
         FieldCheck {"ProjectedSphericalRectangle::generateSurfaceOffset", &R::surfaceOffset, 2e-2, 1e-2},
         FieldCheck {"ProjectedSphericalRectangle::forwardPdf",            &R::forwardPdf,    5e-2, 1e-4},
         FieldCheck {"ProjectedSphericalRectangle::forwardWeight",         &R::forwardWeight, 5e-2, 1e-4},
         FieldCheck {"ProjectedSphericalRectangle::backwardWeight",        &R::backwardWeight,5e-2, 1e-4});
      VERIFY_PDFS_POSITIVE(pass, actual, iteration, seed, testType,
         PdfCheck {"ProjectedSphericalRectangle::forwardPdf", &R::forwardPdf});
      VERIFY_JACOBIAN_OR_SKIP(pass, "ProjectedSphericalRectangle::jacobianProduct", 1.0f, actual.jacobianProduct, iteration, seed, testType, 5e-2, 5e-2);

      constexpr float boundsEps = 1e-5f;
      if (actual.surfaceOffset.x < -boundsEps || actual.surfaceOffset.x > actual.extents.x + boundsEps ||
         actual.surfaceOffset.y < -boundsEps || actual.surfaceOffset.y > actual.extents.y + boundsEps)
      {
         pass = false;
         printTestFail("ProjectedSphericalRectangle::generateSurfaceOffset (inside rect bounds)", actual.extents, actual.surfaceOffset, iteration, seed, testType, 0.0, boundsEps);
      }

      // generate must be unit length
      {
         const float dirLen = nbl::hlsl::length(actual.generated);
         pass &= verifyTestValue("ProjectedSphericalRectangle::generate (unit length)", dirLen, 1.0f, iteration, seed, testType, 1e-5, 1e-4);
      }

      // generate must agree with the reference direction reconstructed from the surface point
      pass &= verifyTestValue("ProjectedSphericalRectangle::generate vs generateSurfaceOffset", actual.generated, actual.referenceDirection, iteration, seed, testType, 5e-5, 5e-3);

      if (!pass && iteration < m_inputs.size())
         logFailedInput(m_logger.get(), m_inputs[iteration]);

      return pass;
   }

   core::vector<ProjectedSphericalRectangleInputValues> m_inputs;
};

// --- Property test configs ---

// Helper: create a ProjectedSphericalRectangle sampler from a random rectangle + normal
inline nbl::hlsl::sampling::ProjectedSphericalRectangle<nbl::hlsl::float32_t, false> createProjectedRectSampler(
   std::mt19937& rng,
   nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t>& compressed,
   nbl::hlsl::float32_t3& observer,
   nbl::hlsl::float32_t3& outNormal,
   void(*rectGen)(std::mt19937&, nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t>&, nbl::hlsl::float32_t3&))
{
   using namespace nbl::hlsl;
   rectGen(rng, compressed, observer);
   auto shape = shapes::SphericalRectangle<float32_t>::create(compressed);

   do
   {
      outNormal = generateRandomUnitVector(rng);
   } while (!anyRectCornerAboveHorizon(shape, observer, outNormal));

   return sampling::ProjectedSphericalRectangle<float32_t, false>::create(shape, observer, outNormal, false);
}

struct ProjectedSphericalRectanglePropertyConfig
{
   // UsePdfAsWeight=false so receiverNormal and projSolidAngle are populated for logSamplerInfo.
   using sampler_type = nbl::hlsl::sampling::ProjectedSphericalRectangle<nbl::hlsl::float32_t, false>;

   static constexpr uint32_t numConfigurations = 200;
   static constexpr uint32_t samplesPerConfig = 50000;
   static constexpr bool hasMCNormalization = true;
   static constexpr bool hasGridIntegration = false;
   static constexpr float64_t mcNormalizationRelTol = 0.08;
   static constexpr float64_t gridNormalizationAbsTol = 0.0;

   static const char* name() { return "ProjectedSphericalRectangle"; }

   static sampler_type createRandomSampler(std::mt19937& rng)
   {
      nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t> compressed;
      nbl::hlsl::float32_t3 observer, normal;
      return createProjectedRectSampler(rng, compressed, observer, normal, generateRandomRectangle);
   }

   static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

   // E[1/pdf] = solidAngle * E[1/bilinearPdf] = solidAngle * 1.0 = solidAngle
   static float64_t expectedCodomainMeasure(const sampler_type& s)
   {
      return static_cast<float64_t>(s.sphrect.solidAngle);
   }

   static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
   {
      using nbl::system::to_string;
      logger->log("    r0=%s extents=%s solidAngle=%s projSolidAngle=%s receiverNormal=%s",
         nbl::system::ILogger::ELL_ERROR,
         to_string(s.sphrect.r0).c_str(),
         to_string(s.sphrect.extents).c_str(),
         to_string(s.sphrect.solidAngle).c_str(),
         to_string(s.projSolidAngle).c_str(),
         to_string(s.receiverNormal).c_str());
   }
};

struct ProjectedSphericalRectangleGrazingConfig
{
   // UsePdfAsWeight=false so receiverNormal and projSolidAngle are populated for logSamplerInfo.
   using sampler_type = nbl::hlsl::sampling::ProjectedSphericalRectangle<nbl::hlsl::float32_t, false>;

   static constexpr uint32_t numConfigurations = 200;
   static constexpr uint32_t samplesPerConfig = 20000;
   static constexpr bool hasMCNormalization = true;
   static constexpr bool hasGridIntegration = false;
   // Single-corner bilinear patches (3/4 zero corners) have near-divergent
   // 1/bilinearPdf near zero-density regions, causing extreme MC variance.
   // 20k samples can't reliably converge closer than ~25% for these configs.
   static constexpr float64_t mcNormalizationRelTol = 0.25;
   static constexpr float64_t gridNormalizationAbsTol = 0.0;

   static const char* name() { return "ProjectedSphericalRectangle(grazing)"; }

   static sampler_type createRandomSampler(std::mt19937& rng)
   {
      nbl::hlsl::shapes::CompressedSphericalRectangle<nbl::hlsl::float32_t> compressed;
      nbl::hlsl::float32_t3 observer, normal;
      return createProjectedRectSampler(rng, compressed, observer, normal, generateStressRectangle);
   }

   static nbl::hlsl::float32_t2 randomDomain(std::mt19937& rng) { return uniformRandomDomain<nbl::hlsl::float32_t2>(rng); }

   static float64_t expectedCodomainMeasure(const sampler_type& s)
   {
      return static_cast<float64_t>(s.sphrect.solidAngle);
   }

   static void logSamplerInfo(nbl::system::ILogger* logger, const sampler_type& s)
   {
      using nbl::system::to_string;
      logger->log("    r0=%s extents=%s solidAngle=%s projSolidAngle=%s receiverNormal=%s",
         nbl::system::ILogger::ELL_ERROR,
         to_string(s.sphrect.r0).c_str(),
         to_string(s.sphrect.extents).c_str(),
         to_string(s.sphrect.solidAngle).c_str(),
         to_string(s.projSolidAngle).c_str(),
         to_string(s.receiverNormal).c_str());
   }
};

#endif
