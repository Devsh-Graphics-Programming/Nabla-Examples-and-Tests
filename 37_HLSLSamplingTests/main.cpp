#include <nabla.h>

#include <utility>

#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

using namespace nbl;
//using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace nbl::hlsl;
using namespace nbl::examples;

// sampling headers (HLSL/C++ compatible)
#include "nbl/builtin/hlsl/sampling/concentric_mapping.hlsl"
#include "nbl/builtin/hlsl/sampling/polar_mapping.hlsl"
#include "nbl/builtin/hlsl/sampling/linear.hlsl"
#include "nbl/builtin/hlsl/sampling/bilinear.hlsl"
#include "nbl/builtin/hlsl/sampling/uniform_spheres.hlsl"
#include "nbl/builtin/hlsl/sampling/cos_weighted_spheres.hlsl"
#include "nbl/builtin/hlsl/sampling/box_muller_transform.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/projected_spherical_triangle.hlsl"
#include "nbl/builtin/hlsl/sampling/projected_spherical_rectangle.hlsl"
#include "nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl"
#include "nbl/builtin/hlsl/sampling/alias_table.hlsl"
#include "nbl/builtin/hlsl/sampling/cumulative_probability.hlsl"

// concepts header — include AFTER sampler headers, and only in the test
#include "nbl/builtin/hlsl/sampling/concepts.hlsl"

// ITester-based testers
#include "tests/CLinearTester.h"
#include "tests/CBilinearTester.h"
#include "tests/CUniformHemisphereTester.h"
#include "tests/CUniformSphereTester.h"
#include "tests/CProjectedHemisphereTester.h"
#include "tests/CProjectedSphereTester.h"
#include "tests/CConcentricMappingTester.h"
#include "tests/CPolarMappingTester.h"
#include "tests/CSphericalTriangleTester.h"
#include "tests/CBoxMullerTransformTester.h"
#include "tests/CProjectedSphericalTriangleTester.h"
#include "tests/CProjectedSphericalRectangleTester.h"
#include "tests/CSphericalRectangleTester.h"
#include "tests/CDiscreteTableTester.h"
#include "tests/CAliasTableGPUTester.h"
#include "tests/CCumulativeProbabilityGPUTester.h"

#include "benchmarks/CSamplerBenchmark.h"
#include "benchmarks/CDiscreteSamplerBenchmark.h"
#include "tests/property/CSamplerPropertyTester.h"


class HLSLSamplingTests final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
   using device_base_t = application_templates::MonoDeviceApplication;
   using asset_base_t  = BuiltinResourcesApplication;

   public:
   HLSLSamplingTests(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
      : system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

   virtual SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
   {
      auto retval                   = device_base_t::getPreferredDeviceFeatures();
      retval.pipelineExecutableInfo = true;
      return retval;
   }

   inline bool onAppInitialized(core::smart_refctd_ptr<ISystem>&& system) override
   {
      if (!device_base_t::onAppInitialized(core::smart_refctd_ptr(system)))
         return false;

      if (!asset_base_t::onAppInitialized(std::move(system)))
         return false;

      // test compile with dxc
      {
         IAssetLoader::SAssetLoadParams lp = {};
         lp.logger                         = m_logger.get();
         lp.workingDirectory               = "app_resources";
         auto key                          = nbl::this_example::builtin::build::get_spirv_key<"shader">(m_device.get());
         auto bundle                       = m_assetMgr->getAsset(key.c_str(), lp);

         const auto assets = bundle.getContents();
         if (assets.empty())
         {
            m_logger->log("Could not load shader!", ILogger::ELL_ERROR);
            return false;
         }

         auto shader = IAsset::castDown<IShader>(assets[0]);
         if (!shader)
         {
            m_logger->log("compile shader test failed!", ILogger::ELL_ERROR);
            return false;
         }

         m_logger->log("Shader compilation test passed.", ILogger::ELL_INFO);
      }

      // ================================================================
      // Compile-time concept verification via static_assert
      // ================================================================

      // --- BasicSampler (level 1) --- generate(domain_type) -> codomain_type
      // Note: all samplers almost satisfy BasicSampler, but they have cache parameters in generate().
      static_assert(sampling::concepts::BasicSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::BasicSampler<sampling::PolarMapping<float32_t>>);
      static_assert(sampling::concepts::BasicSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::TRACKING>>);
      static_assert(sampling::concepts::BasicSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::YOLO>>);
      static_assert(sampling::concepts::BasicSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::EYTZINGER>>);
      static_assert(sampling::concepts::BasicSampler<sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, ReadOnlyAccessor<uint32_t>, ReadOnlyAccessor<float32_t>, 26>>);
      static_assert(sampling::concepts::BasicSampler<sampling::PackedAliasTableB<float32_t, float32_t, uint32_t, ArrayAccessor<sampling::PackedAliasEntryB<float>, 4>, ReadOnlyAccessor<float32_t>, 26>>);

      // --- TractableSampler (level 2) --- generate(domain_type, out cache_type) -> codomain_type, forwardPdf(domain_type, cache_type) -> density_type
      ;
      static_assert(sampling::concepts::TractableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::TRACKING>>);
      static_assert(sampling::concepts::TractableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::YOLO>>);
      static_assert(sampling::concepts::TractableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::EYTZINGER>>);
      static_assert(sampling::concepts::TractableSampler<sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, ReadOnlyAccessor<uint32_t>, ReadOnlyAccessor<float32_t>, 26>>);
      static_assert(sampling::concepts::TractableSampler<sampling::PackedAliasTableB<float32_t, float32_t, uint32_t, ArrayAccessor<sampling::PackedAliasEntryB<float>, 4>, ReadOnlyAccessor<float32_t>, 26>>);
      static_assert(sampling::concepts::TractableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedSphericalTriangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedSphericalRectangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::SphericalRectangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::BoxMullerTransform<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::TractableSampler<sampling::PolarMapping<float32_t>>);

      // --- ResamplableSampler (level 3, parallel) --- generate(domain_type, out cache_type) -> codomain_type, forwardWeight(domain_type, cache_type), backwardWeight(codomain_type)
      static_assert(sampling::concepts::ResamplableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::TRACKING>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::YOLO>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::CumulativeProbabilitySampler<float32_t, float32_t, uint32_t, ReadOnlyAccessor<float32_t>, sampling::EYTZINGER>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::PackedAliasTableA<float32_t, float32_t, uint32_t, ReadOnlyAccessor<uint32_t>, ReadOnlyAccessor<float32_t>, 26>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::PackedAliasTableB<float32_t, float32_t, uint32_t, ArrayAccessor<sampling::PackedAliasEntryB<float>, 4>, ReadOnlyAccessor<float32_t>, 26>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedSphericalTriangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedSphericalRectangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::BoxMullerTransform<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::SphericalRectangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::PolarMapping<float32_t>>);

      // --- BackwardTractableSampler (level 3) --- TractableSampler + backwardPdf(codomain_type), forwardWeight(domain_type, cache_type), backwardWeight(codomain_type)
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::SphericalTriangle<float>>);
      //static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedSphericalTriangle<float>>); // no backwardPdf
      //static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedSphericalRectangle<float>>);  // no backwardPdf
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::SphericalRectangle<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::BoxMullerTransform<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::PolarMapping<float32_t>>);

      // --- BijectiveSampler (level 4) --- BackwardTractableSampler + generateInverse(codomain_type) -> domain_type
      static_assert(sampling::concepts::BijectiveSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::BijectiveSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::BijectiveSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::BijectiveSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::BijectiveSampler<sampling::ConcentricMapping<float>>);
      static_assert(sampling::concepts::BijectiveSampler<sampling::PolarMapping<float>>);

      // --- GenericReadAccessor for discrete samplers ---
      static_assert(concepts::accessors::GenericReadAccessor<ReadOnlyAccessor<float>, float, uint32_t>);
      static_assert(concepts::accessors::GenericReadAccessor<ArrayAccessor<float, 4>, float, uint32_t>);
      static_assert(concepts::accessors::GenericReadAccessor<ArrayAccessor<uint32_t, 4>, uint32_t, uint32_t>);

      m_logger->log("All sampling concept tests passed.", ILogger::ELL_INFO);

      // ======================================================================
      // GPU throughput benchmarks
      // ======================================================================
      constexpr uint32_t testBatchCount = 4096;
      constexpr bool     DoBenchmark    = true;

      if constexpr (DoBenchmark)
      {
         constexpr uint32_t benchWorkgroupSize      = WORKGROUP_SIZE;
         constexpr uint32_t totalThreadsPerDispatch = testBatchCount * benchWorkgroupSize;
         constexpr uint32_t iterationsPerThread     = BENCH_ITERS;
         constexpr uint32_t benchSamplesPerDispatch = totalThreadsPerDispatch * iterationsPerThread;

         struct BenchEntry
         {
            CSamplerBenchmark bench;
            std::string       sampler;
            std::string       mode;
         };
         std::vector<BenchEntry> benchmarks;

         auto addBench = [&](const char* sampler, const char* mode, const std::string& shaderKey, size_t inputSize, size_t outputSize)
         {
            auto& entry   = benchmarks.emplace_back();
            entry.sampler = sampler;
            entry.mode    = mode;

            CSamplerBenchmark::SetupData data;
            data.device             = m_device;
            data.api                = m_api;
            data.assetMgr           = m_assetMgr;
            data.logger             = m_logger;
            data.physicalDevice     = m_physicalDevice;
            data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            data.shaderKey          = shaderKey;
            data.dispatchGroupCount = testBatchCount;
            data.samplesPerDispatch = benchSamplesPerDispatch;
            data.inputBufferBytes   = inputSize;
            data.outputBufferBytes  = outputSize;
            entry.bench.setup(data);
         };

         // Bench shaders don't read input (hardcoded values) and write a single uint32_t per thread via RWByteAddressBuffer
         if constexpr (true)
         {
            constexpr size_t benchInputBytes  = sizeof(uint32_t); // unused but binding must exist, didn't bother removing because some samplers need more complex inputs and it's easier to have a consistent buffer setup for all benchmarks
            constexpr size_t benchOutputBytes = sizeof(uint32_t) * totalThreadsPerDispatch;
            addBench("Linear", "1:1", nbl::this_example::builtin::build::get_spirv_key<"linear_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("Linear", "1:16", nbl::this_example::builtin::build::get_spirv_key<"linear_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("Bilinear", "1:1", nbl::this_example::builtin::build::get_spirv_key<"bilinear_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("Bilinear", "1:16", nbl::this_example::builtin::build::get_spirv_key<"bilinear_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("BoxMullerTransform", "1:1", nbl::this_example::builtin::build::get_spirv_key<"box_muller_transform_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("BoxMullerTransform", "1:16", nbl::this_example::builtin::build::get_spirv_key<"box_muller_transform_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("UniformHemisphere", "1:1", nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("UniformHemisphere", "1:16", nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("UniformSphere", "1:1", nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("UniformSphere", "1:16", nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ConcentricMapping", "1:1", nbl::this_example::builtin::build::get_spirv_key<"concentric_mapping_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ConcentricMapping", "1:16", nbl::this_example::builtin::build::get_spirv_key<"concentric_mapping_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("PolarMapping", "1:1", nbl::this_example::builtin::build::get_spirv_key<"polar_mapping_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("PolarMapping", "1:16", nbl::this_example::builtin::build::get_spirv_key<"polar_mapping_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedHemisphere", "1:1", nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedHemisphere", "1:16", nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphere", "1:1", nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphere", "1:16", nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:1  (shape,observer)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_1_shape_observer">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:16 (shape,observer)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_16_shape_observer">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:1  (sa,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_1_sa_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:16 (sa,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_16_sa_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:1  (r0,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_1_r0_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "1:16 (r0,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_1_16_r0_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "create-only (shape,observer)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_create_only_shape_observer">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "create-only (sa,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_create_only_sa_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalRectangle", "create-only (r0,extents)", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench_create_only_r0_extents">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalRectangle", "1:1", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_rectangle_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalRectangle", "1:16", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_rectangle_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalRectangle", "create-only", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_rectangle_bench_create_only">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalTriangle", "1:1", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalTriangle", "1:16", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("SphericalTriangle", "create-only", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_bench_create_only">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalTriangle", "1:1", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_bench_1_1">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalTriangle", "1:16", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_bench_1_16">(m_device.get()), benchInputBytes, benchOutputBytes);
            addBench("ProjectedSphericalTriangle", "create-only", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_bench_create_only">(m_device.get()), benchInputBytes, benchOutputBytes);
         }

         // Print all pipeline reports first
         for (auto& entry : benchmarks)
            entry.bench.logPipelineReport(entry.sampler + " (" + entry.mode + ")");

         // Discrete sampler benchmark: alias table vs cumulative probability (BDA)
         {
            CDiscreteSamplerBenchmark::SetupData dsData;
            dsData.device                    = m_device;
            dsData.api                       = m_api;
            dsData.assetMgr                  = m_assetMgr;
            dsData.logger                    = m_logger;
            dsData.physicalDevice            = m_physicalDevice;
            dsData.computeFamilyIndex        = getComputeQueue()->getFamilyIndex();
            dsData.packedAliasAShaderKey     = nbl::this_example::builtin::build::get_spirv_key<"packed_alias_a_bench">(m_device.get());
            dsData.packedAliasBShaderKey     = nbl::this_example::builtin::build::get_spirv_key<"packed_alias_b_bench">(m_device.get());
            dsData.cumProbShaderKey          = nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_bench">(m_device.get());
            dsData.cumProbYoloShaderKey      = nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_yolo_bench">(m_device.get());
            dsData.cumProbEytzingerShaderKey = nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_eytzinger_bench">(m_device.get());
            dsData.dispatchGroupCount        = testBatchCount;

            CDiscreteSamplerBenchmark discreteBench;
            discreteBench.setup(dsData);

            // Then run all benchmarks here so the reports are at the top of the log, followed by timings
            {
               constexpr uint32_t warmupDispatches = 300;
               constexpr uint32_t benchDispatches  = 1000;
               m_logger->log("=== GPU Sampler Benchmarks (%u dispatches, %u threads/dispatch, %u iters/thread, ps/sample is per all GPU threads) ===",
                  ILogger::ELL_PERFORMANCE, benchDispatches, totalThreadsPerDispatch, iterationsPerThread);
               m_logger->log("            %-28s | %-38s | %12s | %12s | %12s",
                  ILogger::ELL_PERFORMANCE, "Sampler", "Mode", "ps/sample", "GSamples/s", "ms total");
               for (auto& entry : benchmarks)
                  entry.bench.run(entry.sampler, entry.mode, warmupDispatches, benchDispatches);
            }

            {
               // If you change something here, better change kBenchTable below too
               const std::vector<uint32_t> discreteSizes = {
                  2u, 4u, 8u, 16u, 32u, 64u, 100u, 128u, 256u, 400u, 512u, 1024u, 2048u, 2049u, 3000u, 4096u, 7000u, 8192u, 10'000u, 16'384u, 32'768u,
                  65'536u, 131'072u, 262'144u, 524'288u, 1'000'000u, 1'048'576u, 2'097'152u, 16'777'216u, 20'971'520u, 25'165'824u, 33'554'432u};

               // Per-N dispatch counts calibrated from a prior measured run
               auto dispatchScheduler = [](uint32_t N) -> CDiscreteSamplerBenchmark::DispatchCounts
               {
                  static constexpr std::pair<uint32_t, uint32_t> kBenchTable[] = {
                     {2u, 7180u}, {4u, 5993u}, {8u, 4490u}, {16u, 4099u}, {32u, 3110u}, {64u, 3026u}, {100u, 2507u}, {128u, 2498u}, {256u, 2477u}, {400u, 2001u},
                     {512u, 1827u}, {1024u, 1372u}, {2048u, 1010u}, {2049u, 1010u}, {3000u, 859u}, {4096u, 962u}, {7000u, 742u}, {8192u, 833u}, {10'000u, 590u}, {16'384u, 786u}, {32'768u, 608u},
                     {65'536u, 283u}, {131'072u, 174u}, {262'144u, 160u}, {524'288u, 133u}, {1'000'000u, 77u}, {1'048'576u, 128u}, {2'097'152u, 106u}, {16'777'216u, 17u}, {20'971'520u, 17u}, {25'165'824u, 16u}, {33'554'432u, 14u}};
                  uint32_t bench = 10u; // fallback for any N not in the table
                  for (const auto& e : kBenchTable)
                     if (e.first == N)
                     {
                        bench = e.second;
                        break;
                     }
                  const uint32_t warmup = std::max(5u, bench / 10u);
                  return {warmup, bench};
               };

               discreteBench.runSweep(discreteSizes, dispatchScheduler);
            }
         }
      }

      // ================================================================
      // Runtime CPU/GPU comparison tests using ITester harness
      // ================================================================
      bool pass = true;

      // generic lambda to run a GPU sampler test
      auto runSamplerTest = [&]<typename Tester>(const char* testName, auto spirvKey, const char* logFile)
      {
         m_logger->log("Running %s tests...", ILogger::ELL_INFO, testName);
         typename Tester::PipelineSetupData data;
         data.device             = m_device;
         data.api                = m_api;
         data.assetMgr           = m_assetMgr;
         data.logger             = m_logger;
         data.physicalDevice     = m_physicalDevice;
         data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
         data.shaderKey          = std::move(spirvKey);
         Tester tester(testBatchCount);
         tester.setupPipeline(data);
         pass &= tester.performTestsAndVerifyResults(logFile);
      };

      // --- Sampler tests ---
      if constexpr (true)
      {
         runSamplerTest.operator()<CLinearTester>("Linear sampler", nbl::this_example::builtin::build::get_spirv_key<"linear_test">(m_device.get()), "LinearTestLog.txt");
         runSamplerTest.operator()<CBilinearTester>("Bilinear sampler", nbl::this_example::builtin::build::get_spirv_key<"bilinear_test">(m_device.get()), "BilinearTestLog.txt");
         runSamplerTest.operator()<CUniformHemisphereTester>("UniformHemisphere sampler", nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_test">(m_device.get()), "UniformHemisphereTestLog.txt");
         runSamplerTest.operator()<CUniformSphereTester>("UniformSphere sampler", nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_test">(m_device.get()), "UniformSphereTestLog.txt");
         runSamplerTest.operator()<CProjectedHemisphereTester>("ProjectedHemisphere sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_test">(m_device.get()), "ProjectedHemisphereTestLog.txt");
         runSamplerTest.operator()<CProjectedSphereTester>("ProjectedSphere sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_test">(m_device.get()), "ProjectedSphereTestLog.txt");
         runSamplerTest.operator()<CConcentricMappingTester>("ConcentricMapping sampler", nbl::this_example::builtin::build::get_spirv_key<"concentric_mapping_test">(m_device.get()), "ConcentricMappingTestLog.txt");
         runSamplerTest.operator()<CPolarMappingTester>("PolarMapping sampler", nbl::this_example::builtin::build::get_spirv_key<"polar_mapping_test">(m_device.get()), "PolarMappingTestLog.txt");
         runSamplerTest.operator()<CBoxMullerTransformTester>("BoxMullerTransform sampler", nbl::this_example::builtin::build::get_spirv_key<"box_muller_transform_test">(m_device.get()), "BoxMullerTransformTestLog.txt");
         runSamplerTest.operator()<CSphericalTriangleTester>("SphericalTriangle", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle">(m_device.get()), "SphericalTriangleTestLog.txt");
         runSamplerTest.operator()<CProjectedSphericalTriangleTester>("ProjectedSphericalTriangle sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_test">(m_device.get()), "ProjectedSphericalTriangleTestLog.txt");
         runSamplerTest.operator()<CSphericalRectangleTester>("SphericalRectangle sampler", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_test">(m_device.get()), "SphericalRectangleTestLog.txt");
         runSamplerTest.operator()<CProjectedSphericalRectangleTester>("ProjectedSphericalRectangle sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_rectangle_test">(m_device.get()), "ProjectedSphericalRectangleTestLog.txt");
      }

      if constexpr (DoBenchmark)
      {
         // --- Discrete table construction (CPU) ---
         {
            m_logger->log("Running discrete table builder tests (CPU)...", ILogger::ELL_INFO);
            CDiscreteTableTester tableTester(m_logger.get());
            pass &= tableTester.run();
         }

         // --- GPU table sampler tests ---
         runSamplerTest.operator()<CPackedAliasAGPUTester>("PackedAliasA GPU sampler", nbl::this_example::builtin::build::get_spirv_key<"packed_alias_a_test">(m_device.get()), "PackedAliasATestLog.txt");
         runSamplerTest.operator()<CPackedAliasBGPUTester>("PackedAliasB GPU sampler", nbl::this_example::builtin::build::get_spirv_key<"packed_alias_b_test">(m_device.get()), "PackedAliasBTestLog.txt");
         runSamplerTest.operator()<CCumulativeProbabilityGPUTester>("CumulativeProbability GPU sampler", nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_test">(m_device.get()), "CumulativeProbabilityTestLog.txt");
      }
      logJacobianSkipCounts(m_logger.get());
      if (pass)
         m_logger->log("All sampling tests PASSED.", ILogger::ELL_INFO);
      else
         m_logger->log("Some sampling tests FAILED. Check log files for details.", ILogger::ELL_ERROR);

      // ================================================================
      // CPU-only mathematical property tests (PDF normalization, consistency)
      // ================================================================
      if constexpr (true)
      {
         m_logger->log("Running sampler property tests (CPU)...", ILogger::ELL_INFO);
         m_logger->log("WARNING: CPU math may use higher intermediate precision than GPU shaders. Tolerances that pass here may be too tight for GPU.", ILogger::ELL_WARNING);

         CSamplerPropertyTester<LinearPropertyConfig> linearProps(m_logger.get());
         pass &= linearProps.run();

         CSamplerPropertyTester<BilinearPropertyConfig> bilinearProps(m_logger.get());
         pass &= bilinearProps.run();

         CSamplerPropertyTester<UniformHemispherePropertyConfig> uniformHemiProps(m_logger.get());
         pass &= uniformHemiProps.run();

         CSamplerPropertyTester<UniformSpherePropertyConfig> uniformSphereProps(m_logger.get());
         pass &= uniformSphereProps.run();

         CSamplerPropertyTester<ProjectedHemispherePropertyConfig> projHemiProps(m_logger.get());
         pass &= projHemiProps.run();

         CSamplerPropertyTester<ProjectedSpherePropertyConfig> projSphereProps(m_logger.get());
         pass &= projSphereProps.run();

         CSamplerPropertyTester<ConcentricMappingPropertyConfig> concentricProps(m_logger.get());
         pass &= concentricProps.run();

         CSamplerPropertyTester<PolarMappingPropertyConfig> polarProps(m_logger.get());
         pass &= polarProps.run();

         CSamplerPropertyTester<BoxMullerTransformPropertyConfig> boxMullerProps(m_logger.get());
         pass &= boxMullerProps.run();

         CSamplerPropertyTester<SphericalTrianglePropertyConfig> sphTriProps(m_logger.get());
         pass &= sphTriProps.run();

         CSamplerPropertyTester<ProjectedSphericalTrianglePropertyConfig> projSphTriProps(m_logger.get());
         pass &= projSphTriProps.run();

         CSamplerPropertyTester<SphericalRectanglePropertyConfig> sphRectProps(m_logger.get());
         pass &= sphRectProps.run();

         CSamplerPropertyTester<ProjectedSphericalRectanglePropertyConfig> projSphRectProps(m_logger.get());
         pass &= projSphRectProps.run();

         // Stress tests: extreme coefficient ratios
         CSamplerPropertyTester<LinearStressConfig> linearStress(m_logger.get());
         pass &= linearStress.run();

         CSamplerPropertyTester<BilinearStressConfig> bilinearStress(m_logger.get());
         pass &= bilinearStress.run();

         CSamplerPropertyTester<BilinearPSTPatternConfig> bilinearPST(m_logger.get());
         pass &= bilinearPST.run();

         CSamplerPropertyTester<SphericalTriangleStressConfig> sphTriStress(m_logger.get());
         pass &= sphTriStress.run();

         // Grazing angle tests
         CSamplerPropertyTester<ProjectedSphericalTriangleGrazingConfig> grazingProps(m_logger.get());
         pass &= grazingProps.run();

         if (pass)
            m_logger->log("All sampler property tests PASSED.", ILogger::ELL_INFO);
         else
            m_logger->log("Some sampler property tests FAILED.", ILogger::ELL_ERROR);
      }

      // ================================================================
      // Solid angle accuracy and small triangle convergence tests (CPU-only)
      // ================================================================
      if constexpr (true)
      {
         m_logger->log("Running geometry tests (CPU)...", ILogger::ELL_INFO);
         m_logger->log("WARNING: CPU math may use higher intermediate precision than GPU shaders. Tolerances that pass here may be too tight for GPU.", ILogger::ELL_WARNING);

         CSolidAngleAccuracyTester solidAngleTester(m_logger.get());
         pass &= solidAngleTester.run();

         CSphericalTriangleGenerateTester sphTriGenTester(m_logger.get());
         pass &= sphTriGenTester.run();

         CSphericalRectangleGenerateTester sphRectGenTester(m_logger.get());
         pass &= sphRectGenTester.run();

         CProjectedSphericalRectangleGenerateTester projRectGenTester(m_logger.get());
         pass &= projRectGenTester.run();

         CProjectedSphericalRectangleGeometricTester projRectGeoTester(m_logger.get());
         pass &= projRectGeoTester.run();

         CProjectedSphericalTriangleGeometricTester pstTester(m_logger.get());
         pass &= pstTester.run();

         if (pass)
            m_logger->log("All geometry tests PASSED.", ILogger::ELL_INFO);
         else
            m_logger->log("Some geometry tests FAILED.", ILogger::ELL_ERROR);
      }

      return pass;
   }

   void workLoopBody() override {}

   bool keepRunning() override { return false; }

   bool onAppTerminated() override
   {
      return device_base_t::onAppTerminated();
   }
};

NBL_MAIN_FUNC(HLSLSamplingTests)
