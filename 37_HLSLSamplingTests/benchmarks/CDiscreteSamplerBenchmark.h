#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
#include "app_resources/common/discrete_sampler_bench.hlsl"

#include <random>

using namespace nbl;

// Benchmarks alias table vs cumulative probability sampler on the GPU using BDA.
// Builds pipelines once, then sweeps a list of table sizes. For each N it builds
// both tables from the same weight distribution, uploads via BDA buffers, and
// measures GPU throughput using timestamp queries. The cumulative probability
// sampler is run in two variants: the stateful-comparator cache population
// (default) and the "YOLO re-read" variant (cumulative_probability.hlsl).
class CDiscreteSamplerBenchmark
{
   public:
   struct SetupData
   {
      core::smart_refctd_ptr<video::ILogicalDevice> device;
      core::smart_refctd_ptr<video::CVulkanConnection> api;
      core::smart_refctd_ptr<asset::IAssetManager> assetMgr;
      core::smart_refctd_ptr<system::ILogger> logger;
      video::IPhysicalDevice* physicalDevice;
      std::string aliasShaderKey;
      std::string cumProbShaderKey;
      std::string cumProbYoloShaderKey;
      uint32_t computeFamilyIndex;
      uint32_t dispatchGroupCount;
   };

   void setup(const SetupData& data)
   {
      m_device = data.device;
      m_logger = data.logger;
      m_assetMgr = data.assetMgr;
      m_dispatchGroupCount = data.dispatchGroupCount;
      m_physicalDevice = data.physicalDevice;

      m_queue = m_device->getQueue(data.computeFamilyIndex, 0);

      // Command pool + buffers
      m_cmdpool = m_device->createCommandPool(data.computeFamilyIndex, video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_benchCmdbuf);

      // Timestamp query pool
      {
         video::IQueryPool::SCreationParams qp = {};
         qp.queryType = video::IQueryPool::TYPE::TIMESTAMP;
         qp.queryCount = 2;
         qp.pipelineStatisticsFlags = video::IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
         m_queryPool = m_device->createQueryPool(qp);
      }

      const uint32_t totalThreads = m_dispatchGroupCount * WORKGROUP_SIZE;

      // Shared output buffer (size only depends on thread count)
      {
         video::IGPUBuffer::SCreationParams bp = {};
         bp.size = totalThreads * sizeof(uint32_t);
         bp.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) |
            video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
         m_outputBuf = m_device->createBuffer(std::move(bp));
         video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_outputBuf->getMemoryReqs();
         reqs.memoryTypeBits &= data.physicalDevice->getHostVisibleMemoryTypeBits();
         m_device->allocate(reqs, m_outputBuf.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
      }

      // Pipelines (N-independent; only push constants change per run)
      m_aliasPipeline = createPipeline<AliasTablePushConstants>(data.aliasShaderKey, m_aliasPplnLayout, "alias");
      m_cumProbPipeline = createPipeline<CumProbPushConstants>(data.cumProbShaderKey, m_cumProbPplnLayout, "cumprob-comparator");
      m_cumProbYoloPipeline = createPipeline<CumProbPushConstants>(data.cumProbYoloShaderKey, m_cumProbYoloPplnLayout, "cumprob-yolo");
   }

   // DispatchScheduler: uint32_t N -> std::pair<uint32_t warmup, uint32_t bench>.
   // Lets the caller trade wall-clock for statistical stability per size:
   // big-N runs are DRAM-bound and need fewer dispatches to hit the same total sample count.
   struct DispatchCounts { uint32_t warmup; uint32_t bench; };

   // Sweep a list of table sizes. For each N: build tables from a fresh weight
   // distribution (deterministic seed = 42 + N so different N's get distinct
   // distributions but runs are reproducible), upload via BDA, then run all
   // three samplers with the dispatch counts chosen by `scheduler`.
   template<typename DispatchScheduler>
   void runSweep(const std::vector<uint32_t>& tableSizes, DispatchScheduler scheduler)
   {
      const uint32_t totalThreads = m_dispatchGroupCount * WORKGROUP_SIZE;
      m_logger->log("=== GPU Discrete Sampler Benchmark sweep (%u threads * %u iters/thread; wg=%u; dispatches chosen per-N) ===",
         system::ILogger::ELL_PERFORMANCE, totalThreads, BENCH_ITERS, WORKGROUP_SIZE);
      m_logger->log("%12s | %-28s | %12s | %12s | %12s | %10s",
         system::ILogger::ELL_PERFORMANCE, "N", "Sampler", "ps/sample", "GSamples/s", "ms total", "dispatches");

      for (uint32_t N : tableSizes)
      {
         const DispatchCounts dc = scheduler(N);
         buildAndUpload(N);
         runSingle(N, "AliasTable",                    m_aliasPipeline,       m_aliasPplnLayout,       SamplerKind::Alias,           dc.warmup, dc.bench);
         runSingle(N, "CumulativeProbability",         m_cumProbPipeline,     m_cumProbPplnLayout,     SamplerKind::CumProbCompare,  dc.warmup, dc.bench);
         runSingle(N, "CumulativeProbability (YOLO)",  m_cumProbYoloPipeline, m_cumProbYoloPplnLayout, SamplerKind::CumProbYolo,     dc.warmup, dc.bench);
         releaseTables();
      }
   }

   // Convenience: sweep with fixed dispatch counts for every size.
   void runSweep(const std::vector<uint32_t>& tableSizes, uint32_t warmupIterations = 500, uint32_t benchmarkIterations = 5000)
   {
      runSweep(tableSizes, [warmupIterations, benchmarkIterations](uint32_t) -> DispatchCounts {
         return {warmupIterations, benchmarkIterations};
      });
   }

   private:
   enum class SamplerKind { Alias, CumProbCompare, CumProbYolo };

   template<typename PushConstantT>
   core::smart_refctd_ptr<video::IGPUComputePipeline> createPipeline(const std::string& shaderKey, core::smart_refctd_ptr<video::IGPUPipelineLayout>& outLayout, const char* tag)
   {
      const asset::SPushConstantRange pcRange = {
         .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
         .offset = 0,
         .size = sizeof(PushConstantT)};
      auto layout = m_device->createPipelineLayout({&pcRange, 1});
      if (!layout)
         m_logger->log("CDiscreteSamplerBenchmark: failed to create %s pipeline layout", system::ILogger::ELL_ERROR, tag);

      asset::IAssetLoader::SAssetLoadParams lp = {};
      lp.logger = m_logger.get();
      lp.workingDirectory = "app_resources";
      auto bundle = m_assetMgr->getAsset(shaderKey, lp);
      auto source = asset::IAsset::castDown<asset::IShader>(bundle.getContents()[0]);
      auto shader = m_device->compileShader({.source = source.get()});
      if (!shader)
         m_logger->log("CDiscreteSamplerBenchmark: failed to load %s shader", system::ILogger::ELL_ERROR, tag);

      video::IGPUComputePipeline::SCreationParams pp = {};
      pp.layout = layout.get();
      pp.shader.shader = shader.get();
      pp.shader.entryPoint = "main";
      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
      {
         pp.flags |= video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
      }

      core::smart_refctd_ptr<video::IGPUComputePipeline> pipeline;
      if (!m_device->createComputePipelines(nullptr, {&pp, 1}, &pipeline))
         m_logger->log("CDiscreteSamplerBenchmark: failed to create %s compute pipeline", system::ILogger::ELL_ERROR, tag);

      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
      {
         auto report = system::to_string(pipeline->getExecutableInfo());
         m_logger->log("%s Sampling Pipeline Executable Report:\n%s", system::ILogger::ELL_PERFORMANCE, tag, report.c_str());
      }
      outLayout = std::move(layout);
      return pipeline;
   }

   core::smart_refctd_ptr<video::IGPUBuffer> createBdaBuffer(const void* srcData, size_t bytes)
   {
      video::IGPUBuffer::SCreationParams bp = {};
      bp.size = bytes;
      bp.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) |
         video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
      auto buf = m_device->createBuffer(std::move(bp));

      video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buf->getMemoryReqs();
      reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();
      auto alloc = m_device->allocate(reqs, buf.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

      const auto allocSize = alloc.memory->getAllocationSize();
      if (alloc.memory->map({0ull, allocSize}, video::IDeviceMemoryAllocation::EMCAF_WRITE))
      {
         std::memcpy(alloc.memory->getMappedPointer(), srcData, bytes);
         video::ILogicalDevice::MappedMemoryRange flushRange(alloc.memory.get(), 0ull, allocSize);
         m_device->flushMappedMemoryRanges(1u, &flushRange);
         alloc.memory->unmap();
      }
      return buf;
   }

   void buildAndUpload(uint32_t N)
   {
      m_currentN = N;

      std::vector<float> weights(N);
      std::mt19937 rng(42u + N);
      std::uniform_real_distribution<float> dist(0.001f, 100.0f);
      for (uint32_t i = 0; i < N; i++)
         weights[i] = dist(rng);

      // Alias table
      std::vector<float> aliasProb(N);
      std::vector<uint32_t> aliasIdx(N);
      std::vector<float> aliasPdf(N);
      std::vector<uint32_t> workspace(N);
      nbl::hlsl::sampling::AliasTableBuilder<float>::build({weights}, aliasProb.data(), aliasIdx.data(), aliasPdf.data(), workspace.data());

      // Cumulative probability (N-1 entries, last bucket implicitly 1.0)
      std::vector<float> cumProb(N > 0 ? N - 1 : 0);
      nbl::hlsl::sampling::computeNormalizedCumulativeHistogram({weights}, cumProb.data());

      m_aliasProbBuf = createBdaBuffer(aliasProb.data(), N * sizeof(float));
      m_aliasIdxBuf  = createBdaBuffer(aliasIdx.data(), N * sizeof(uint32_t));
      m_aliasPdfBuf  = createBdaBuffer(aliasPdf.data(), N * sizeof(float));
      const size_t cumProbBytes = (N > 0 ? (N - 1) : 0) * sizeof(float);
      m_cumProbBuf = cumProbBytes ? createBdaBuffer(cumProb.data(), cumProbBytes) : nullptr;
   }

   void releaseTables()
   {
      m_aliasProbBuf = nullptr;
      m_aliasIdxBuf  = nullptr;
      m_aliasPdfBuf  = nullptr;
      m_cumProbBuf   = nullptr;
   }

   void runSingle(
      uint32_t N,
      const char* name,
      const core::smart_refctd_ptr<video::IGPUComputePipeline>& pipeline,
      const core::smart_refctd_ptr<video::IGPUPipelineLayout>& layout,
      SamplerKind kind,
      uint32_t warmupIterations,
      uint32_t benchmarkIterations)
   {
      m_device->waitIdle();

      // Everything (warmup, timestamped bench, cooldown) goes into ONE cmdbuf and ONE
      // submit. Serial submissions with semaphore waits between them would add sync cost
      // to every dispatch and prevent the driver from overlapping adjacent dispatches.
      // With a single cmdbuf the driver pipelines freely, and GPU memory latency is
      // hidden by warp hyperthreading rather than by cross-submit overlap.
      //
      // Layout: [warmup dispatches] [ts 0] [bench dispatches] [ts 1] [cooldown dispatches]
      // Warmup brings clocks + caches to steady state before ts 0. Cooldown keeps the
      // same steady-state context alive across ts 1 so the trailing bench dispatches
      // don't measure a tail where the GPU is already winding down.
      const uint32_t cooldownIterations = warmupIterations;

      m_benchCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
      m_benchCmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      m_benchCmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);
      m_benchCmdbuf->bindComputePipeline(pipeline.get());

      if (kind == SamplerKind::Alias)
      {
         AliasTablePushConstants pc = {};
         pc.probAddress  = m_aliasProbBuf->getDeviceAddress();
         pc.aliasAddress = m_aliasIdxBuf->getDeviceAddress();
         pc.pdfAddress   = m_aliasPdfBuf->getDeviceAddress();
         pc.outputAddress = m_outputBuf->getDeviceAddress();
         pc.tableSize = N;
         m_benchCmdbuf->pushConstants(layout.get(), asset::IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }
      else
      {
         CumProbPushConstants pc = {};
         pc.cumProbAddress = m_cumProbBuf ? m_cumProbBuf->getDeviceAddress() : 0ull;
         pc.outputAddress  = m_outputBuf->getDeviceAddress();
         pc.tableSize = N;
         m_benchCmdbuf->pushConstants(layout.get(), asset::IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }

      for (uint32_t i = 0u; i < warmupIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 0);
      for (uint32_t i = 0u; i < benchmarkIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 1);
      for (uint32_t i = 0u; i < cooldownIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->end();

      auto semaphore = m_device->createSemaphore(0u);
      const video::IQueue::SSubmitInfo::SCommandBufferInfo benchCmds[] = {{.cmdbuf = m_benchCmdbuf.get()}};
      const video::IQueue::SSubmitInfo::SSemaphoreInfo signalSem[] = {
         {.semaphore = semaphore.get(), .value = 1u, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
      video::IQueue::SSubmitInfo submit = {};
      submit.commandBuffers = benchCmds;
      submit.signalSemaphores = signalSem;
      m_queue->submit({&submit, 1u});

      m_device->waitIdle();

      uint64_t timestamps[2] = {};
      const auto flags = core::bitflag(video::IQueryPool::RESULTS_FLAGS::_64_BIT) |
         core::bitflag(video::IQueryPool::RESULTS_FLAGS::WAIT_BIT);
      m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, timestamps, sizeof(uint64_t), flags);

      constexpr uint32_t benchIters = BENCH_ITERS;
      const float64_t timestampPeriod = float64_t(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
      const float64_t elapsed_ns = float64_t(timestamps[1] - timestamps[0]) * timestampPeriod;
      const uint64_t totalThreads = uint64_t(m_dispatchGroupCount) * uint64_t(WORKGROUP_SIZE);
      const uint64_t totalSamples = uint64_t(benchmarkIterations) * totalThreads * uint64_t(benchIters);
      const float64_t ps_per_sample = elapsed_ns * 1e3 / float64_t(totalSamples);
      const float64_t gsamples_per_s = float64_t(totalSamples) / elapsed_ns;
      const float64_t elapsed_ms = elapsed_ns * 1e-6;

      m_logger->log("%12u | %-28s | %12.3f | %12.3f | %12.3f | %10u",
         system::ILogger::ELL_PERFORMANCE, N, name, ps_per_sample, gsamples_per_s, elapsed_ms, benchmarkIterations);
   }

   core::smart_refctd_ptr<video::ILogicalDevice> m_device;
   core::smart_refctd_ptr<system::ILogger> m_logger;
   core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;
   core::smart_refctd_ptr<video::IGPUCommandPool> m_cmdpool;
   core::smart_refctd_ptr<video::IGPUCommandBuffer> m_benchCmdbuf;
   core::smart_refctd_ptr<video::IQueryPool> m_queryPool;

   // Pipelines (set up once)
   core::smart_refctd_ptr<video::IGPUPipelineLayout> m_aliasPplnLayout;
   core::smart_refctd_ptr<video::IGPUComputePipeline> m_aliasPipeline;
   core::smart_refctd_ptr<video::IGPUPipelineLayout> m_cumProbPplnLayout;
   core::smart_refctd_ptr<video::IGPUComputePipeline> m_cumProbPipeline;
   core::smart_refctd_ptr<video::IGPUPipelineLayout> m_cumProbYoloPplnLayout;
   core::smart_refctd_ptr<video::IGPUComputePipeline> m_cumProbYoloPipeline;

   // Per-N data buffers (rebuilt each sweep step)
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasProbBuf;
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasIdxBuf;
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasPdfBuf;
   core::smart_refctd_ptr<video::IGPUBuffer> m_cumProbBuf;

   // Shared
   core::smart_refctd_ptr<video::IGPUBuffer> m_outputBuf;
   video::IQueue* m_queue = nullptr;
   video::IPhysicalDevice* m_physicalDevice = nullptr;
   uint32_t m_dispatchGroupCount = 0;
   uint32_t m_currentN = 0;
};

#endif
