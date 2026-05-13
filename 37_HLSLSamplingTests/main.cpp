#include <nabla.h>

#include <chrono>
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
      constexpr uint32_t benchWorkgroupsCount = 4096;
      constexpr bool     DoBenchmark    = true;

      if constexpr (DoBenchmark)
      {
         constexpr uint32_t benchWorkgroupSize      = WORKGROUP_SIZE;
         constexpr uint32_t totalThreadsPerDispatch = benchWorkgroupsCount * benchWorkgroupSize;
         constexpr uint32_t iterationsPerThread     = BENCH_ITERS;
         constexpr uint32_t benchSamplesPerDispatch = totalThreadsPerDispatch * iterationsPerThread;
         constexpr uint32_t warmupDispatches        = 300;          // unmeasured warmup + cooldown around the timing window
         constexpr uint64_t targetBudgetMs          = 400;          // wall-clock per row; runTimedBudgeted sizes dispatches

         std::vector<CSamplerBenchmark> benchmarks;

         // Single Aggregator owns results, baselines, formatting, and reporting
         // for both bench classes. Passed by reference into each bench's ctor.
         Aggregator agg(m_logger, m_device, m_physicalDevice, getComputeQueue()->getFamilyIndex());
         const auto cli = agg.applyCli({
            .argv              = this->argv,
            .defaultOutputPath = "SamplerBench.json",
            .appName           = "37_HLSLSamplingTests",
         });

         // One context for the whole sampler-bench span: drives both the per-bench
         // shape/budget and the banner that runSessionAndReport prints.
         const RunContext samplerCtx = {
            .shape          = {
                       .workgroupSize      = {benchWorkgroupSize, 1u, 1u},
                       .dispatchGroupCount = {benchWorkgroupsCount, 1u, 1u},
                       .samplesPerDispatch = benchSamplesPerDispatch,
            },
            .targetBudgetMs = targetBudgetMs,
            .sectionLabel   = "GPU Sampler Benchmarks",
         };

         auto addBench = [&](const std::initializer_list<std::string> name, GPUBenchmarkHelper::ShaderVariant variant, size_t outputSize)
         {
            CSamplerBenchmark::SetupData data;
            data.assetMgr          = m_assetMgr;
            data.name              = name;
            data.variant           = std::move(variant);
            data.outputBufferBytes = outputSize;
            data.warmupDispatches  = warmupDispatches;
            data.shape             = samplerCtx.shape;
            data.targetBudgetMs    = samplerCtx.targetBudgetMs;

            benchmarks.emplace_back(agg, data);
         };

         // Convenience wrappers so the 35+ existing precompiled-key calls below stay
         // one line each, and adding a new runtime variant is also a one-liner without
         // CMake JSON edits. Both go through the same addBench, just construct the
         // ShaderVariant differently.
         auto addPrecompiled = [&]<nbl::core::StringLiteral ShaderKey>(std::initializer_list<std::string> name, size_t outputSize)
         {
            auto shader = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
            addBench(name, GPUBenchmarkHelper::ShaderVariant::Precompiled(std::move(shader)), outputSize);
         };
         auto addRuntime = [&](std::initializer_list<std::string> name, const char* sourcePath, std::vector<GPUBenchmarkHelper::ShaderVariant::Define> defines, size_t outputSize)
         {
            // Mirror CMake's COMMON_OPTIONS so runtime variants see the same baseline
            // as precompiled ones.
            std::vector<GPUBenchmarkHelper::ShaderVariant::Define> all = {
               {"WORKGROUP_SIZE", std::to_string(WORKGROUP_SIZE)},
               {"BENCH_ITERS", std::to_string(BENCH_ITERS)},
            };
            all.insert(all.end(), std::make_move_iterator(defines.begin()), std::make_move_iterator(defines.end()));
            addBench(name, GPUBenchmarkHelper::ShaderVariant::FromSource(sourcePath, std::move(all)), outputSize);
         };

         // Bench shaders don't read input -- output is BDA via push constants.
         if constexpr (true)
         {
            constexpr size_t benchOutputBytes = sizeof(uint32_t) * totalThreadsPerDispatch;
            addPrecompiled.operator()<"linear_bench_1_1">({"Linear", "Linear", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"linear_bench_1_16">({"Linear", "Linear", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"bilinear_bench_1_1">({"Linear", "Bilinear", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"bilinear_bench_1_16">({"Linear", "Bilinear", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"box_muller_transform_bench_1_1">({"Gaussian", "BoxMullerTransform", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"box_muller_transform_bench_1_16">({"Gaussian", "BoxMullerTransform", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"uniform_hemisphere_bench_1_1">({"SphereSampling", "UniformHemisphere", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"uniform_hemisphere_bench_1_16">({"SphereSampling", "UniformHemisphere", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"uniform_sphere_bench_1_1">({"SphereSampling", "UniformSphere", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"uniform_sphere_bench_1_16">({"SphereSampling", "UniformSphere", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_hemisphere_bench_1_1">({"SphereSampling", "ProjectedHemisphere", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_hemisphere_bench_1_16">({"SphereSampling", "ProjectedHemisphere", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_sphere_bench_1_1">({"SphereSampling", "ProjectedSphere", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_sphere_bench_1_16">({"SphereSampling", "ProjectedSphere", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"concentric_mapping_bench_1_1">({"DiskMappers", "ConcentricMapping", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"concentric_mapping_bench_1_16">({"DiskMappers", "ConcentricMapping", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"polar_mapping_bench_1_1">({"DiskMappers", "PolarMapping", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"polar_mapping_bench_1_16">({"DiskMappers", "PolarMapping", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_1_shape_observer">({"SphShapes", "SphRect", "1:1", "shape,observer"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_1_sa_extents">({"SphShapes", "SphRect", "1:1", "sa,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_1_r0_extents">({"SphShapes", "SphRect", "1:1", "r0,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_16_shape_observer">({"SphShapes", "SphRect", "1:16", "shape,observer"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_16_sa_extents">({"SphShapes", "SphRect", "1:16", "sa,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_1_16_r0_extents">({"SphShapes", "SphRect", "1:16", "r0,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_create_only_shape_observer">({"SphShapes", "SphRect", "create-only", "shape,observer"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_create_only_sa_extents">({"SphShapes", "SphRect", "create-only", "sa,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_rectangle_bench_create_only_r0_extents">({"SphShapes", "SphRect", "create-only", "r0,extents"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_rectangle_bench_1_1">({"SphShapes", "ProjSphRect", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_rectangle_bench_1_16">({"SphShapes", "ProjSphRect", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_rectangle_bench_create_only">({"SphShapes", "ProjSphRect", "create-only"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_triangle_bench_1_1">({"SphShapes", "SphTri", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_triangle_bench_1_16">({"SphShapes", "SphTri", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"spherical_triangle_bench_create_only">({"SphShapes", "SphTri", "create-only"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_triangle_bench_1_1">({"SphShapes", "ProjSphTri", "1:1"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_triangle_bench_1_16">({"SphShapes", "ProjSphTri", "1:16"}, benchOutputBytes);
            addPrecompiled.operator()<"projected_spherical_triangle_bench_create_only">({"SphShapes", "ProjSphTri", "create-only"}, benchOutputBytes);
            // ---- Runtime-compiled demo variants (no CMake JSON edit needed) ----
            // Same .hlsl source as the precompiled "linear_bench_1_*" entries, but with
            // a `BENCH_SAMPLES_PER_CREATE` value that has no JSON entry. Add as many
            // here as you want -- each is a one-liner, no reconfigure required.
            //addRuntime({"Linear", "Linear", "1:4 (rt)"}, "shaders/linear_test.comp.hlsl", {{"BENCH_SAMPLES_PER_CREATE", "4"}}, benchOutputBytes);
            //addRuntime({"Linear", "Linear", "1:8 (rt)"}, "shaders/linear_test.comp.hlsl", {{"BENCH_SAMPLES_PER_CREATE", "8"}}, benchOutputBytes);
         }

         // Discrete sampler benchmark: alias table vs cumulative probability (BDA)
         {
            CDiscreteSamplerBenchmark::SetupData dsData;
            dsData.assetMgr                = m_assetMgr;
            dsData.packedAliasAVariant     = GPUBenchmarkHelper::ShaderVariant::Precompiled(nbl::this_example::builtin::build::get_spirv_key<"packed_alias_a_bench">(m_device.get()));
            dsData.packedAliasBVariant     = GPUBenchmarkHelper::ShaderVariant::Precompiled(nbl::this_example::builtin::build::get_spirv_key<"packed_alias_b_bench">(m_device.get()));
            dsData.cumProbVariant          = GPUBenchmarkHelper::ShaderVariant::Precompiled(nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_bench">(m_device.get()));
            dsData.cumProbYoloVariant      = GPUBenchmarkHelper::ShaderVariant::Precompiled(nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_yolo_bench">(m_device.get()));
            dsData.cumProbEytzingerVariant = GPUBenchmarkHelper::ShaderVariant::Precompiled(nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_eytzinger_bench">(m_device.get()));
            dsData.dispatchGroupCount      = {benchWorkgroupsCount, 1u, 1u};
            dsData.targetBudgetMs          = targetBudgetMs;

            // Just the N values now -- runTimedBudgeted sizes dispatches per
            // row to hit the budget. The old per-N tuning table is gone.
            static constexpr uint32_t kSweepNs[] = {
               2u, 4u, 8u, 16u, 32u, 64u, 100u, 128u, 256u, 400u,
               512u, 1024u, 2048u, 2049u, 3000u, 4096u, 7000u, 8192u, 10'000u, 16'384u, 32'768u,
               65'536u, 131'072u, 262'144u, 524'288u, 1'000'000u, 1'048'576u, 2'097'152u, 16'777'216u, 20'971'520u, 25'165'824u, 33'554'432u};
            dsData.sweepNs                 = kSweepNs;

            CDiscreteSamplerBenchmark discreteBench(agg, dsData);

            const RunContext discreteCtx = {
               .shape          = CDiscreteSamplerBenchmark::shapeFor(dsData),
               .targetBudgetMs = targetBudgetMs,
               .sectionLabel   = "Discrete Sampler Sweep",
            };

            // Single call. Each span contributes its own focus rows first, then
            // every span's unfocused rows -- the aggregator iterates both packs
            // in each phase. CDiscrete's overridden run() does per-row filtering
            // against cli.focusVariants since its rows aren't a flat list.
            agg.runSessionAndReport(
               Aggregator::makeSpan(benchmarks,    samplerCtx),
               Aggregator::makeSpan(discreteBench, discreteCtx));
         }
      }

      // ================================================================
      // Runtime CPU/GPU comparison tests using ITester harness
      // ================================================================
      bool pass = true;
      constexpr uint32_t testWorkgroupsCount = 4096;
      // generic lambda to run a GPU sampler test
      auto runSamplerTest = [&]<typename Tester, core::StringLiteral ShaderKey>(const char* testName, const char* logFile)
      {
         m_logger->log("Running %s tests...", ILogger::ELL_INFO, testName);
         typename Tester::PipelineSetupData data;
         data.device             = m_device;
         data.api                = m_api;
         data.assetMgr           = m_assetMgr;
         data.logger             = m_logger;
         data.physicalDevice     = m_physicalDevice;
         data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
         data.shaderKey          = std::move(nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get()));
         Tester tester(testWorkgroupsCount);
         tester.setupPipeline(data);
         pass &= tester.performTestsAndVerifyResults(logFile);
      };

      // --- Sampler tests ---
      if constexpr (true)
      {
         runSamplerTest.operator()<CLinearTester, "linear_test">("Linear sampler", "LinearTestLog.txt");
         runSamplerTest.operator()<CBilinearTester, "bilinear_test">("Bilinear sampler", "BilinearTestLog.txt");
         runSamplerTest.operator()<CUniformHemisphereTester, "uniform_hemisphere_test">("UniformHemisphere sampler", "UniformHemisphereTestLog.txt");
         runSamplerTest.operator()<CUniformSphereTester, "uniform_sphere_test">("UniformSphere sampler", "UniformSphereTestLog.txt");
         runSamplerTest.operator()<CProjectedHemisphereTester, "projected_hemisphere_test">("ProjectedHemisphere sampler", "ProjectedHemisphereTestLog.txt");
         runSamplerTest.operator()<CProjectedSphereTester, "projected_sphere_test">("ProjectedSphere sampler", "ProjectedSphereTestLog.txt");
         runSamplerTest.operator()<CConcentricMappingTester, "concentric_mapping_test">("ConcentricMapping sampler", "ConcentricMappingTestLog.txt");
         runSamplerTest.operator()<CPolarMappingTester, "polar_mapping_test">("PolarMapping sampler", "PolarMappingTestLog.txt");
         runSamplerTest.operator()<CBoxMullerTransformTester, "box_muller_transform_test">("BoxMullerTransform sampler", "BoxMullerTransformTestLog.txt");
         runSamplerTest.operator()<CSphericalTriangleTester, "spherical_triangle">("SphericalTriangle", "SphericalTriangleTestLog.txt");
         runSamplerTest.operator()<CProjectedSphericalTriangleTester, "projected_spherical_triangle_test">("ProjectedSphericalTriangle sampler", "ProjectedSphericalTriangleTestLog.txt");
         runSamplerTest.operator()<CSphericalRectangleTester, "spherical_rectangle_test">("SphericalRectangle sampler", "SphericalRectangleTestLog.txt");
         runSamplerTest.operator()<CProjectedSphericalRectangleTester, "projected_spherical_rectangle_test">("ProjectedSphericalRectangle sampler", "ProjectedSphericalRectangleTestLog.txt");
      }

      if constexpr (true)
      {
         // --- Discrete table construction (CPU) ---
         {
            m_logger->log("Running discrete table builder tests (CPU)...", ILogger::ELL_INFO);
            CDiscreteTableTester tableTester(m_logger.get());
            pass &= tableTester.run();
         }

         // --- GPU table sampler tests ---
         runSamplerTest.operator()<CPackedAliasAGPUTester, "packed_alias_a_test">("PackedAliasA GPU sampler", "PackedAliasATestLog.txt");
         runSamplerTest.operator()<CPackedAliasBGPUTester, "packed_alias_b_test">("PackedAliasB GPU sampler", "PackedAliasBTestLog.txt");
         runSamplerTest.operator()<CCumulativeProbabilityGPUTester, "cumulative_probability_test">("CumulativeProbability GPU sampler", "CumulativeProbabilityTestLog.txt");
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

         auto check = [&]<typename Config>()
         {
            pass &= CSamplerPropertyTester<Config>(m_logger.get()).run();
         };

         check.operator()<LinearPropertyConfig>();
         check.operator()<BilinearPropertyConfig>();
         check.operator()<UniformHemispherePropertyConfig>();
         check.operator()<UniformSpherePropertyConfig>();
         check.operator()<ProjectedHemispherePropertyConfig>();
         check.operator()<ProjectedSpherePropertyConfig>();
         check.operator()<ConcentricMappingPropertyConfig>();
         check.operator()<PolarMappingPropertyConfig>();
         check.operator()<BoxMullerTransformPropertyConfig>();
         check.operator()<SphericalTrianglePropertyConfig>();
         check.operator()<ProjectedSphericalTrianglePropertyConfig>();
         check.operator()<SphericalRectanglePropertyConfig>();
         check.operator()<ProjectedSphericalRectanglePropertyConfig>();

         // Stress tests: extreme coefficient ratios
         check.operator()<LinearStressConfig>();
         check.operator()<BilinearStressConfig>();
         check.operator()<BilinearPSTPatternConfig>();
         check.operator()<SphericalTriangleStressConfig>();

         // Grazing angle tests
         check.operator()<ProjectedSphericalTriangleGrazingConfig>();

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

         auto check = [&]<typename Tester>()
         {
            pass &= Tester(m_logger.get()).run();
         };

         check.template operator()<CSolidAngleAccuracyTester>();
         check.template operator()<CSphericalTriangleGenerateTester>();
         check.template operator()<CSphericalRectangleGenerateTester>();
         check.template operator()<CProjectedSphericalRectangleGenerateTester>();
         check.template operator()<CProjectedSphericalRectangleGeometricTester>();
         check.template operator()<CProjectedSphericalTriangleGeometricTester>();

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
