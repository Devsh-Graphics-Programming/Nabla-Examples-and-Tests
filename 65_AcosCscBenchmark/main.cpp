// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "nbl/examples/examples.hpp"

#include <nabla.h>
#include <array>
#include <span>
#include <assert.h>
#include <cfenv>

#include "app_resources/benchmark/common.hlsl"
#include "nbl/examples/Benchmark/IBenchmark.h"
#include "nbl/examples/Benchmark/GPUBenchmarkHelper.h"

using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;

class AcosCscBenchmark : public GPUBenchmark
{
   public:
   static constexpr const char* kSectionLabel = "AcosCsc Benchmarks";

   struct SetupData
   {
      smart_refctd_ptr<IAssetManager>     assetMgr;
      core::vector<core::string>          name; // hierarchical row name
      BENCHMARK_MODE                      mode; // pushed each run() via PC
      GPUBenchmarkHelper::ShaderVariant   variant; // precompiled "benchmark" SPIRV
      uint32_t                            warmupDispatches;
      uint64_t                            targetBudgetMs;
   };

   // Shape is fixed by the BENCHMARK_WORKGROUP_* macros; expose it so the
   // caller uses the same shape both to construct the bench and to build the
   // RunContext for its span.
   static WorkloadShape shape()
   {
      const hlsl::uint32_t3 wg = {
         BENCHMARK_WORKGROUP_DIMENSION_SIZE_X,
         BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y,
         BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z};
      const hlsl::uint32_t3 dgc = {BENCHMARK_WORKGROUP_COUNT, 1u, 1u};
      // Shader writes one float64 per thread per dispatch;
      const uint64_t samplesPerDispatch = uint64_t(dgc.x) * dgc.y * dgc.z * wg.x * wg.y * wg.z * BENCHMARK_SAMPLE_PER_THREAD;
      return {.workgroupSize = wg, .dispatchGroupCount = dgc, .samplesPerDispatch = samplesPerDispatch};
   }

   AcosCscBenchmark(Aggregator& aggregator, const SetupData& data)
      : GPUBenchmark(aggregator, GPUBenchmark::SetupData{
                                    .name             = data.name,
                                    .warmupDispatches = data.warmupDispatches,
                                    .shape            = shape(),
                                    .targetBudgetMs   = data.targetBudgetMs,
                                 })
      , m_mode(data.mode)
   {
      // Buffer needs one float64 per thread, not per sample (BENCHMARK_SAMPLE_PER_THREAD iters collapse to one output)
      m_buffer = createOutputBuffer(uint64_t(BENCHMARK_WORKGROUP_COUNT) * BENCHMARK_WORKGROUP_DIMENSION_SIZE_X * sizeof(float64_t));

      // One SSBO at set 0 / binding 0. createSingleBindingDS wires the
      // layout + pool + DS + write descriptor in one call.
      auto ds       = createSingleBindingDS(m_buffer);
      m_dsLayout    = std::move(ds.layout);
      m_ds          = std::move(ds.set);
      m_pipelineIdx = createPipeline(data.variant, data.assetMgr, sizeof(BenchmarkPushConstants), joinName(data.name), m_dsLayout);
   }

   void doRun() override
   {
      const PipelineEntry*   pe = getPipelineEntry(m_pipelineIdx, joinName(m_name));
      if (!pe)
         return;
      BenchmarkPushConstants pc = {};
      pc.benchmarkMode          = m_mode;

      const TimingResult t = runTimedBudgeted(getWarmupDispatches(), getTargetBudgetMs(),
         [&](IGPUCommandBuffer* cb)
         {
            cb->bindDescriptorSets(EPBP_COMPUTE, pe->layout.get(), 0, 1, &m_ds.get());
            defaultBindAndPush(cb, *pe, pc);
         },
         [this](IGPUCommandBuffer* cb) { defaultDispatch(cb); },
         samplesForCurrentRow());

      record(m_name, t, pe->stats);
   }

   private:
   BENCHMARK_MODE                            m_mode = BM_EXACT;
   smart_refctd_ptr<IGPUBuffer>              m_buffer;
   smart_refctd_ptr<IGPUDescriptorSetLayout> m_dsLayout;
   smart_refctd_ptr<IGPUDescriptorSet>       m_ds;
   uint32_t                                  m_pipelineIdx = 0;
};

class AcosCscBenchmarkApp final : public MonoDeviceApplication, public BuiltinResourcesApplication
{
   using device_base_t = MonoDeviceApplication;
   using asset_base_t  = BuiltinResourcesApplication;

   public:
   AcosCscBenchmarkApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

   virtual SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
   {
      auto retval                   = device_base_t::getPreferredDeviceFeatures();
      retval.pipelineExecutableInfo = true;
      return retval;
   }

   bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
   {
      if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
         return false;
      if (!asset_base_t::onAppInitialized(std::move(system)))
         return false;

      return true;
   }

   void onAppTerminated_impl() override
   {
      m_device->waitIdle();
   }

   void workLoopBody() override
   {
      runBenchmarks();
      m_keepRunning = false;
   }

   bool keepRunning() override
   {
      return m_keepRunning;
   }


   private:
   bool m_keepRunning = true;

   void runBenchmarks()
   {
      constexpr uint32_t WarmupDispatches = 1000;
      constexpr uint64_t TargetBudgetMs   = 400; // ~400ms per row

      auto runBenchmark = [&](const std::string& outputPath, const std::string& shaderKey, const std::string& name, std::span<const std::pair<BENCHMARK_MODE, const char*>> modes)
      {
          Aggregator agg(m_logger, m_device, m_physicalDevice, getComputeQueue()->getFamilyIndex());
          agg.applyCli({
             .argv              = this->argv,
             .defaultOutputPath = outputPath,
             .appName           = "65_AcosCscBenchmark",
          });

          auto       shaderVariant = GPUBenchmarkHelper::ShaderVariant::Precompiled(shaderKey);

          std::vector<AcosCscBenchmark> benches;
          benches.reserve(modes.size());
          for (size_t i = 0; i < modes.size(); ++i)
          {
             const auto& [mode, leaf] = modes[i];
             benches.emplace_back(agg, AcosCscBenchmark::SetupData{
                .assetMgr         = m_assetMgr,
                .name             = {name, leaf},
                .mode             = mode,
                .variant          = shaderVariant,
                .warmupDispatches = WarmupDispatches,
                .targetBudgetMs   = TargetBudgetMs,
             });
          }

          const RunContext ctx = {
             .shape          = AcosCscBenchmark::shape(),
             .targetBudgetMs = TargetBudgetMs,
             .sectionLabel   = AcosCscBenchmark::kSectionLabel,
          };
          agg.runSessionAndReport(Aggregator::makeSpan(benches, ctx));

      };
      constexpr std::pair<BENCHMARK_MODE, const char*> kModes[] = {
        {BM_SETUP, "setup"},
        {BM_EXACT, "baseline"},
        {BM_ORDER1, "order1"},
        {BM_ORDER2, "order2"},
        {BM_ORDER3, "order3"},
        {BM_SIGN_FLIP, "sign_flip"},
      };
      runBenchmark("AcosCscBench.json", nbl::this_example::builtin::build::get_spirv_key<"acos_csc_benchmark">(m_device.get()), "AcosCsc", std::span(kModes, BM_ORDER3 + 1));
      
      runBenchmark("IntegrateEdge.json", nbl::this_example::builtin::build::get_spirv_key<"integrate_edge_benchmark">(m_device.get()), "IntegrateEdge", std::span(kModes));
   }


   template<typename... Args>
   inline bool logFail(const char* msg, Args&&... args)
   {
      m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
      return false;
   }

   std::ofstream m_logFile;
};

NBL_MAIN_FUNC(AcosCscBenchmarkApp)
