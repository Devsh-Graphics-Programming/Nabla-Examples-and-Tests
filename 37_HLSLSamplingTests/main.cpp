#include <nabla.h>

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
#include "nbl/builtin/hlsl/sampling/spherical_rectangle.hlsl"
#include "nbl/builtin/hlsl/sampling/alias_table.hlsl"
#include "nbl/builtin/hlsl/sampling/cumulative_probability.hlsl"

// concepts header — include AFTER sampler headers, and only in the test
#include "nbl/builtin/hlsl/sampling/concepts.hlsl"

// ITester-based testers
#include "CLinearTester.h"
#include "CBilinearTester.h"
#include "CUniformHemisphereTester.h"
#include "CUniformSphereTester.h"
#include "CProjectedHemisphereTester.h"
#include "CProjectedSphereTester.h"
#include "CConcentricMappingTester.h"
#include "CPolarMappingTester.h"
#include "CSphericalTriangleTester.h"
#include "CBoxMullerTransformTester.h"
#include "CProjectedSphericalTriangleTester.h"
#include "CSphericalRectangleTester.h"
#include "CDiscreteTableTester.h"
#include "CAliasTableGPUTester.h"
#include "CCumulativeProbabilityGPUTester.h"

#include "CSamplerBenchmark.h"
#include "CDiscreteSamplerBenchmark.h"

constexpr bool DoBenchmark = true;

class HLSLSamplingTests final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
   using device_base_t = application_templates::MonoDeviceApplication;
   using asset_base_t = BuiltinResourcesApplication;

   // Helper to create pipeline setup data
   template<typename Tester>
   auto createSetupData(const std::string& shaderKey) -> typename Tester::PipelineSetupData
   {
      typename Tester::PipelineSetupData data;
      data.device = m_device;
      data.api = m_api;
      data.assetMgr = m_assetMgr;
      data.logger = m_logger;
      data.physicalDevice = m_physicalDevice;
      data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
      data.shaderKey = shaderKey;
      return data;
   }

   CSamplerBenchmark::SetupData createBenchmarkSetupData(const std::string& shaderKey, uint32_t dispatchGroupCount, uint32_t samplesPerDispatch, size_t inputBufferBytes, size_t outputBufferBytes) const
   {
      CSamplerBenchmark::SetupData data;
      data.device = m_device;
      data.api = m_api;
      data.assetMgr = m_assetMgr;
      data.logger = m_logger;
      data.physicalDevice = m_physicalDevice;
      data.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
      data.shaderKey = shaderKey;
      data.dispatchGroupCount = dispatchGroupCount;
      data.samplesPerDispatch = samplesPerDispatch;
      data.inputBufferBytes = inputBufferBytes;
      data.outputBufferBytes = outputBufferBytes;
      return data;
   }

   public:
   HLSLSamplingTests(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
      : system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

   virtual SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
   {
      auto retval = device_base_t::getPreferredDeviceFeatures();
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
         lp.logger = m_logger.get();
         lp.workingDirectory = "app_resources";
         auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(m_device.get());
         auto bundle = m_assetMgr->getAsset(key.c_str(), lp);

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
      static_assert(sampling::concepts::BasicSampler<TestAliasTable>);
      static_assert(sampling::concepts::BasicSampler<TestCumulativeProbabilitySampler>);

      // --- TractableSampler (level 2) --- generate(domain_type, out cache_type) -> codomain_type, forwardPdf(cache_type) -> density_type
      static_assert(sampling::concepts::TractableSampler<TestAliasTable>);
      static_assert(sampling::concepts::TractableSampler<TestCumulativeProbabilitySampler>);
      static_assert(sampling::concepts::TractableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ProjectedSphericalTriangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::SphericalRectangle<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::BoxMullerTransform<float>>);
      static_assert(sampling::concepts::TractableSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::TractableSampler<sampling::PolarMapping<float32_t>>);

      // --- ResamplableSampler (level 3, parallel) --- generate(domain_type, out cache_type) -> codomain_type, forwardWeight(cache_type), backwardWeight(codomain_type)
      static_assert(sampling::concepts::ResamplableSampler<TestAliasTable>);
      static_assert(sampling::concepts::ResamplableSampler<TestCumulativeProbabilitySampler>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ProjectedSphericalTriangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::BoxMullerTransform<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::SphericalRectangle<float>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::ConcentricMapping<float32_t>>);
      static_assert(sampling::concepts::ResamplableSampler<sampling::PolarMapping<float32_t>>);

      // --- BackwardTractableSampler (level 3) --- TractableSampler + backwardPdf(codomain_type), forwardWeight(cache_type), backwardWeight(codomain_type)
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::Linear<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::Bilinear<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::UniformHemisphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::UniformSphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedHemisphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedSphere<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::SphericalTriangle<float>>);
      static_assert(sampling::concepts::BackwardTractableSampler<sampling::ProjectedSphericalTriangle<float>>);
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

      // ================================================================
      // Runtime CPU/GPU comparison tests using ITester harness
      // ================================================================
      bool pass = true;
      const uint32_t workgroupSize = 64;
      const uint32_t testBatchCount = 64; // 64 * workgroupSize = 4096 tests per sampler


      // generic lambda to run a GPU sampler test
      auto runSamplerTest = [&]<typename Tester>(const char* testName, auto spirvKey, const char* logFile)
      {
         m_logger->log("Running %s tests...", ILogger::ELL_INFO, testName);
         auto data = createSetupData<Tester>(spirvKey);
         Tester tester(testBatchCount, workgroupSize);
         tester.setupPipeline(data);
         pass &= tester.performTestsAndVerifyResults(logFile);
      };

      // --- Sampler tests ---
      runSamplerTest.operator()<CLinearTester>("Linear sampler", nbl::this_example::builtin::build::get_spirv_key<"linear_test">(m_device.get()), "LinearTestLog.txt");
      runSamplerTest.operator()<CBilinearTester>("Bilinear sampler", nbl::this_example::builtin::build::get_spirv_key<"bilinear_test">(m_device.get()), "BilinearTestLog.txt");
      runSamplerTest.operator()<CUniformHemisphereTester>("UniformHemisphere sampler", nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_test">(m_device.get()), "UniformHemisphereTestLog.txt");
      runSamplerTest.operator()<CUniformSphereTester>("UniformSphere sampler", nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_test">(m_device.get()), "UniformSphereTestLog.txt");
      runSamplerTest.operator()<CProjectedHemisphereTester>("ProjectedHemisphere sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_test">(m_device.get()), "ProjectedHemisphereTestLog.txt");
      runSamplerTest.operator()<CProjectedSphereTester>("ProjectedSphere sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_test">(m_device.get()), "ProjectedSphereTestLog.txt");
      runSamplerTest.operator()<CConcentricMappingTester>("ConcentricMapping sampler", nbl::this_example::builtin::build::get_spirv_key<"concentric_mapping_test">(m_device.get()), "ConcentricMappingTestLog.txt");
      runSamplerTest.operator()<CPolarMappingTester>("PolarMapping sampler", nbl::this_example::builtin::build::get_spirv_key<"polar_mapping_test">(m_device.get()), "PolarMappingTestLog.txt");
      runSamplerTest.operator()<CBoxMullerTransformTester>("BoxMullerTransform sampler", nbl::this_example::builtin::build::get_spirv_key<"box_muller_transform_test">(m_device.get()), "BoxMullerTransformTestLog.txt");
      runSamplerTest.operator()<CProjectedSphericalTriangleTester>("ProjectedSphericalTriangle sampler", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_test">(m_device.get()), "ProjectedSphericalTriangleTestLog.txt");
      runSamplerTest.operator()<CSphericalRectangleTester>("SphericalRectangle sampler", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_test">(m_device.get()), "SphericalRectangleTestLog.txt");
      runSamplerTest.operator()<CSphericalTriangleTester>("SphericalTriangle", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle">(m_device.get()), "SphericalTriangleTestLog.txt");

      // --- Discrete table construction (CPU) ---
      {
         m_logger->log("Running discrete table builder tests (CPU)...", ILogger::ELL_INFO);
         CDiscreteTableTester tableTester(m_logger.get());
         pass &= tableTester.run();
      }

      // --- GPU table sampler tests ---
      runSamplerTest.operator()<CAliasTableGPUTester>("AliasTable GPU sampler", nbl::this_example::builtin::build::get_spirv_key<"alias_table_test">(m_device.get()), "AliasTableTestLog.txt");
      runSamplerTest.operator()<CCumulativeProbabilityGPUTester>("CumulativeProbability GPU sampler", nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_test">(m_device.get()), "CumulativeProbabilityTestLog.txt");

      if (pass)
         m_logger->log("All sampling tests PASSED.", ILogger::ELL_INFO);
      else
         m_logger->log("Some sampling tests FAILED. Check log files for details.", ILogger::ELL_ERROR);

      // ======================================================================
      // GPU throughput benchmarks (1000 warmup + 20000 timed dispatches each)
      // ======================================================================
      if constexpr (DoBenchmark)
      {
         m_logger->log("=== GPU Sampler Benchmarks ===", ILogger::ELL_PERFORMANCE);
         constexpr uint32_t totalSamplesPerWorkgroup = testBatchCount * workgroupSize;
         constexpr uint32_t iteratationsPerThread = 4096; // internal to shader, set in CMakeLists.txt
         constexpr uint32_t benchSamplesPerDispatch = totalSamplesPerWorkgroup * iteratationsPerThread;

         struct BenchEntry
         {
            CSamplerBenchmark bench;
            std::string name;
         };
         std::vector<BenchEntry> benchmarks;

         auto addBench = [&](const char* name, const std::string& shaderKey, size_t inputSize, size_t outputSize)
         {
            auto& entry = benchmarks.emplace_back();
            entry.name = name;
            entry.bench.setup(createBenchmarkSetupData(shaderKey, testBatchCount, benchSamplesPerDispatch, inputSize, outputSize));
         };

         addBench("Linear", nbl::this_example::builtin::build::get_spirv_key<"linear_bench">(m_device.get()), sizeof(LinearInputValues) * totalSamplesPerWorkgroup, sizeof(LinearTestResults) * totalSamplesPerWorkgroup);
         addBench("Bilinear", nbl::this_example::builtin::build::get_spirv_key<"bilinear_bench">(m_device.get()), sizeof(BilinearInputValues) * totalSamplesPerWorkgroup, sizeof(BilinearTestResults) * totalSamplesPerWorkgroup);
         addBench("BoxMullerTransform", nbl::this_example::builtin::build::get_spirv_key<"box_muller_transform_bench">(m_device.get()), sizeof(BoxMullerTransformInputValues) * totalSamplesPerWorkgroup, sizeof(BoxMullerTransformTestResults) * totalSamplesPerWorkgroup);
         addBench("UniformHemisphere", nbl::this_example::builtin::build::get_spirv_key<"uniform_hemisphere_bench">(m_device.get()), sizeof(UniformHemisphereInputValues) * totalSamplesPerWorkgroup, sizeof(UniformHemisphereTestResults) * totalSamplesPerWorkgroup);
         addBench("UniformSphere", nbl::this_example::builtin::build::get_spirv_key<"uniform_sphere_bench">(m_device.get()), sizeof(UniformSphereInputValues) * totalSamplesPerWorkgroup, sizeof(UniformSphereTestResults) * totalSamplesPerWorkgroup);
         addBench("ConcentricMapping", nbl::this_example::builtin::build::get_spirv_key<"concentric_mapping_bench">(m_device.get()), sizeof(ConcentricMappingInputValues) * totalSamplesPerWorkgroup, sizeof(ConcentricMappingTestResults) * totalSamplesPerWorkgroup);
         addBench("PolarMapping", nbl::this_example::builtin::build::get_spirv_key<"polar_mapping_bench">(m_device.get()), sizeof(PolarMappingInputValues) * totalSamplesPerWorkgroup, sizeof(PolarMappingTestResults) * totalSamplesPerWorkgroup);
         addBench("ProjectedHemisphere", nbl::this_example::builtin::build::get_spirv_key<"projected_hemisphere_bench">(m_device.get()), sizeof(ProjectedHemisphereInputValues) * totalSamplesPerWorkgroup, sizeof(ProjectedHemisphereTestResults) * totalSamplesPerWorkgroup);
         addBench("ProjectedSphere", nbl::this_example::builtin::build::get_spirv_key<"projected_sphere_bench">(m_device.get()), sizeof(ProjectedSphereInputValues) * totalSamplesPerWorkgroup, sizeof(ProjectedSphereTestResults) * totalSamplesPerWorkgroup);
         addBench("SphericalRectangle", nbl::this_example::builtin::build::get_spirv_key<"spherical_rectangle_bench">(m_device.get()), sizeof(SphericalRectangleInputValues) * totalSamplesPerWorkgroup, sizeof(SphericalRectangleTestResults) * totalSamplesPerWorkgroup);
         addBench("SphericalTriangle", nbl::this_example::builtin::build::get_spirv_key<"spherical_triangle_bench">(m_device.get()), sizeof(SphericalTriangleInputValues) * totalSamplesPerWorkgroup, sizeof(SphericalTriangleTestResults) * totalSamplesPerWorkgroup);
         addBench("ProjectedSphericalTriangle", nbl::this_example::builtin::build::get_spirv_key<"projected_spherical_triangle_bench">(m_device.get()), sizeof(ProjectedSphericalTriangleInputValues) * totalSamplesPerWorkgroup, sizeof(ProjectedSphericalTriangleTestResults) * totalSamplesPerWorkgroup);

         // Print all pipeline reports first
         for (auto& entry : benchmarks)
            entry.bench.logPipelineReport(entry.name);

         // Discrete sampler benchmark: alias table vs cumulative probability (BDA)
         {
            CDiscreteSamplerBenchmark::SetupData dsData;
            dsData.device = m_device;
            dsData.api = m_api;
            dsData.assetMgr = m_assetMgr;
            dsData.logger = m_logger;
            dsData.physicalDevice = m_physicalDevice;
            dsData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            dsData.aliasShaderKey = nbl::this_example::builtin::build::get_spirv_key<"alias_table_bench">(m_device.get());
            dsData.cumProbShaderKey = nbl::this_example::builtin::build::get_spirv_key<"cumulative_probability_bench">(m_device.get());
            dsData.dispatchGroupCount = testBatchCount;
            dsData.tableSize = 1024;

            CDiscreteSamplerBenchmark discreteBench;
            discreteBench.setup(dsData);

            // Then run all benchmarks here so the reports are at the top of the log, followed by timings
            for (auto& entry : benchmarks)
               entry.bench.run(entry.name);

            discreteBench.run();
         }
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
