#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_DISCRETE_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"
#include <nbl/builtin/hlsl/sampling/alias_table_builder.h>
#include <nbl/builtin/hlsl/sampling/cumulative_probability_builder.h>
#include "app_resources/common/discrete_sampler_bench.hlsl"

#include <random>

using namespace nbl;

class CDiscreteSamplerBenchmark
{
   public:
   struct SetupData
   {
      core::smart_refctd_ptr<ILogicalDevice>    device;
      core::smart_refctd_ptr<CVulkanConnection> api;
      core::smart_refctd_ptr<IAssetManager>     assetMgr;
      core::smart_refctd_ptr<ILogger>          logger;
      IPhysicalDevice*                          physicalDevice;
      std::string                                      packedAliasAShaderKey;
      std::string                                      packedAliasBShaderKey;
      std::string                                      cumProbShaderKey;
      std::string                                      cumProbYoloShaderKey;
      std::string                                      cumProbEytzingerShaderKey;
      uint32_t                                         computeFamilyIndex;
      uint32_t                                         dispatchGroupCount;
   };

   void setup(const SetupData& data)
   {
      m_device             = data.device;
      m_logger             = data.logger;
      m_assetMgr           = data.assetMgr;
      m_dispatchGroupCount = data.dispatchGroupCount;
      m_physicalDevice     = data.physicalDevice;

      m_queue = m_device->getQueue(data.computeFamilyIndex, 0);

      // Staging-upload utility. Without this, BDA buffers land in host-visible (system RAM)
      // and every sampler load becomes a PCIe round-trip instead of hitting VRAM/L2.
      m_utils = IUtilities::create(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr(m_logger));

      // Command pool + buffers
      m_cmdpool = m_device->createCommandPool(data.computeFamilyIndex, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_benchCmdbuf);

      // Timestamp query pool
      {
         IQueryPool::SCreationParams qp = {};
         qp.queryType                          = IQueryPool::TYPE::TIMESTAMP;
         qp.queryCount                         = 2;
         qp.pipelineStatisticsFlags            = IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
         m_queryPool                           = m_device->createQueryPool(qp);
      }

      const uint32_t totalThreads = m_dispatchGroupCount * WORKGROUP_SIZE;

      // Shared output buffer (size only depends on thread count). GPU writes via BDA and
      // nothing reads it on the CPU, so pin it to device-local VRAM.
      {
         IGPUBuffer::SCreationParams bp                      = {};
         bp.size                                                    = totalThreads * sizeof(uint32_t);
         bp.usage                                                   = core::bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
         m_outputBuf                                                = m_device->createBuffer(std::move(bp));
         IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_outputBuf->getMemoryReqs();
         reqs.memoryTypeBits &= data.physicalDevice->getDeviceLocalMemoryTypeBits();
         m_device->allocate(reqs, m_outputBuf.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
      }

      // Pipelines (N-independent; only push constants change per run)
      m_packedAliasAPipeline     = createPipeline<PackedAliasABPushConstants>(data.packedAliasAShaderKey, m_packedAliasAPplnLayout, "alias-packed-A");
      m_packedAliasBPipeline     = createPipeline<PackedAliasABPushConstants>(data.packedAliasBShaderKey, m_packedAliasBPplnLayout, "alias-packed-B");
      m_cumProbPipeline          = createPipeline<CumProbPushConstants>(data.cumProbShaderKey, m_cumProbPplnLayout, "cumprob-comparator");
      m_cumProbYoloPipeline      = createPipeline<CumProbPushConstants>(data.cumProbYoloShaderKey, m_cumProbYoloPplnLayout, "cumprob-yolo");
      m_cumProbEytzingerPipeline = createPipeline<CumProbPushConstants>(data.cumProbEytzingerShaderKey, m_cumProbEytzingerPplnLayout, "cumprob-eytzinger");
   }

   // DispatchScheduler: uint32_t N -> std::pair<uint32_t warmup, uint32_t bench>.
   // Lets the caller trade wall-clock for statistical stability per size:
   // big-N runs are DRAM-bound and need fewer dispatches to hit the same total sample count.
   struct DispatchCounts
   {
      uint32_t warmup;
      uint32_t bench;
   };

   template<typename DispatchScheduler>
   void runSweep(const std::vector<uint32_t>& tableSizes, DispatchScheduler scheduler)
   {
      const uint32_t totalThreads = m_dispatchGroupCount * WORKGROUP_SIZE;
      m_logger->log("=== GPU Discrete Sampler Benchmark sweep (%u threads * %u iters/thread; wg=%u; dispatches chosen per-N) ===",
         ILogger::ELL_PERFORMANCE, totalThreads, BENCH_ITERS, WORKGROUP_SIZE);
      m_logger->log("%12s | %-34s | %12s | %12s | %12s | %10s", ILogger::ELL_PERFORMANCE,
         "N", "Sampler", "ps/sample", "GSamples/s", "ms total", "dispatches");

      for (uint32_t N : tableSizes)
      {
         const DispatchCounts dc = scheduler(N);
         buildAndUpload(N);
         // Packed A wins N<=16k; Packed B wins N>=32k. SoA and Packed C were dominated
         // across every N measured, removed from the sweep.
         runSingle(N, "AliasTable (packed A, 4 B)", m_packedAliasAPipeline, m_packedAliasAPplnLayout, SamplerKind::AliasPackedA, dc.warmup, dc.bench);
         runSingle(N, "AliasTable (packed B, 8 B)", m_packedAliasBPipeline, m_packedAliasBPplnLayout, SamplerKind::AliasPackedB, dc.warmup, dc.bench);
         runSingle(N, "CumulativeProbability", m_cumProbPipeline, m_cumProbPplnLayout, SamplerKind::CumProbCompare, dc.warmup, dc.bench);
         runSingle(N, "CumulativeProbability (YOLO)", m_cumProbYoloPipeline, m_cumProbYoloPplnLayout, SamplerKind::CumProbYolo, dc.warmup, dc.bench);
         runSingle(N, "CumulativeProbability (Eytzinger)", m_cumProbEytzingerPipeline, m_cumProbEytzingerPplnLayout, SamplerKind::CumProbEytzinger, dc.warmup, dc.bench);
         releaseTables();
      }
   }

   // Convenience: sweep with fixed dispatch counts for every size.
   void runSweep(const std::vector<uint32_t>& tableSizes, uint32_t warmupIterations = 500, uint32_t benchmarkIterations = 5000)
   {
      runSweep(tableSizes, [warmupIterations, benchmarkIterations](uint32_t) -> DispatchCounts
         { return {warmupIterations, benchmarkIterations}; });
   }

   private:
   enum class SamplerKind
   {
      AliasPackedA,
      AliasPackedB,
      CumProbCompare,
      CumProbYolo,
      CumProbEytzinger
   };

   template<typename PushConstantT>
   core::smart_refctd_ptr<IGPUComputePipeline> createPipeline(const std::string& shaderKey, core::smart_refctd_ptr<IGPUPipelineLayout>& outLayout, const char* tag)
   {
      const SPushConstantRange pcRange = {
         .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
         .offset     = 0,
         .size       = sizeof(PushConstantT)};
      auto layout = m_device->createPipelineLayout({&pcRange, 1});
      if (!layout)
         m_logger->log("CDiscreteSamplerBenchmark: failed to create %s pipeline layout", ILogger::ELL_ERROR, tag);

      IAssetLoader::SAssetLoadParams lp = {};
      lp.logger                                = m_logger.get();
      lp.workingDirectory                      = "app_resources";
      auto bundle                              = m_assetMgr->getAsset(shaderKey, lp);
      auto source                              = IAsset::castDown<IShader>(bundle.getContents()[0]);
      auto shader                              = m_device->compileShader({.source = source.get()});
      if (!shader)
         m_logger->log("CDiscreteSamplerBenchmark: failed to load %s shader", ILogger::ELL_ERROR, tag);

      IGPUComputePipeline::SCreationParams pp = {};
      pp.layout                                      = layout.get();
      pp.shader.shader                               = shader.get();
      pp.shader.entryPoint                           = "main";
      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
      {
         pp.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
      }

      core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
      if (!m_device->createComputePipelines(nullptr, {&pp, 1}, &pipeline))
         m_logger->log("CDiscreteSamplerBenchmark: failed to create %s compute pipeline", ILogger::ELL_ERROR, tag);

      if (m_device->getEnabledFeatures().pipelineExecutableInfo)
      {
         auto report = system::to_string(pipeline->getExecutableInfo());
         m_logger->log("%s Sampling Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, tag, report.c_str());
      }
      outLayout = std::move(layout);
      return pipeline;
   }

   core::smart_refctd_ptr<IGPUBuffer> createBdaBuffer(const void* srcData, size_t bytes)
   {
      IGPUBuffer::SCreationParams bp = {};
      bp.size                               = bytes;
      bp.usage                              = core::bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) |
         IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT |
         IGPUBuffer::EUF_TRANSFER_DST_BIT;

      core::smart_refctd_ptr<IGPUBuffer> buf;
      auto                                      future = m_utils->createFilledDeviceLocalBufferOnDedMem(
         SIntendedSubmitInfo {.queue = m_queue}, std::move(bp), srcData);
      future.move_into(buf);
      return buf;
   }

   void buildAndUpload(uint32_t N)
   {
      m_currentN = N;

      std::vector<float>                    weights(N);
      std::mt19937                          rng(42u + N);
      std::uniform_real_distribution<float> dist(0.001f, 100.0f);
      for (uint32_t i = 0; i < N; i++)
         weights[i] = dist(rng);

      // Build the alias table SoA (intermediate form), then pack it for variants A and B.
      // Builder may pad PoT N to N+1 for cache-friendly stride; returned size drives
      // every downstream buffer / push-constant value.
      std::vector<float>    aliasProb;
      std::vector<uint32_t> aliasIdx;
      std::vector<float>    aliasPdf;
      m_aliasTableN = sampling::AliasTableBuilder<float>::build({weights}, aliasProb, aliasIdx, aliasPdf);

      constexpr uint32_t                                         kPackedLog2N = 26u;
      std::vector<uint32_t>                                      packedA(m_aliasTableN);
      std::vector<sampling::PackedAliasEntryB<float>> packedB(m_aliasTableN);
      sampling::AliasTableBuilder<float>::packA<kPackedLog2N>({aliasProb}, {aliasIdx}, packedA.data());
      sampling::AliasTableBuilder<float>::packB<kPackedLog2N>({aliasProb}, {aliasIdx}, {aliasPdf}, packedB.data());

      // Cumulative probability (N-1 entries, last bucket implicitly 1.0)
      std::vector<float> cumProb(N - 1u);
      sampling::computeNormalizedCumulativeHistogram({weights}, cumProb.data());

      // Eytzinger level-order tree: 2*P entries where P = nextPot(N)
      const uint32_t     eytzingerP        = sampling::eytzingerLeafCount(N);
      const uint32_t     eytzingerTreeSize = 2u * eytzingerP;
      std::vector<float> cumProbEytzinger(eytzingerTreeSize);
      sampling::buildEytzinger({weights}, cumProbEytzinger.data());

      m_aliasPdfBuf         = createBdaBuffer(aliasPdf.data(), m_aliasTableN * sizeof(float));
      m_packedAliasABuf     = createBdaBuffer(packedA.data(), m_aliasTableN * sizeof(uint32_t));
      m_packedAliasBBuf     = createBdaBuffer(packedB.data(), m_aliasTableN * sizeof(sampling::PackedAliasEntryB<float>));
      m_cumProbBuf          = createBdaBuffer(cumProb.data(), (N - 1u) * sizeof(float));
      m_cumProbEytzingerBuf = createBdaBuffer(cumProbEytzinger.data(), eytzingerTreeSize * sizeof(float));
   }

   void releaseTables()
   {
      m_aliasPdfBuf         = nullptr;
      m_packedAliasABuf     = nullptr;
      m_packedAliasBBuf     = nullptr;
      m_cumProbBuf          = nullptr;
      m_cumProbEytzingerBuf = nullptr;
   }

   void runSingle(uint32_t N, const char* name, const core::smart_refctd_ptr<IGPUComputePipeline>& pipeline, const core::smart_refctd_ptr<IGPUPipelineLayout>& layout, SamplerKind kind, uint32_t warmupIterations, uint32_t benchmarkIterations)
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

      m_benchCmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
      m_benchCmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      m_benchCmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);
      m_benchCmdbuf->bindComputePipeline(pipeline.get());

      if (kind == SamplerKind::AliasPackedA || kind == SamplerKind::AliasPackedB)
      {
         PackedAliasABPushConstants pc = {};
         pc.entriesAddress             = (kind == SamplerKind::AliasPackedA ? m_packedAliasABuf : m_packedAliasBBuf)->getDeviceAddress();
         pc.pdfAddress                 = m_aliasPdfBuf->getDeviceAddress();
         pc.outputAddress              = m_outputBuf->getDeviceAddress();
         pc.tableSize                  = m_aliasTableN;
         m_benchCmdbuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }
      else
      {
         CumProbPushConstants pc  = {};
         const auto&          buf = (kind == SamplerKind::CumProbEytzinger) ? m_cumProbEytzingerBuf : m_cumProbBuf;
         pc.cumProbAddress        = buf->getDeviceAddress();
         pc.outputAddress         = m_outputBuf->getDeviceAddress();
         pc.tableSize             = N;
         m_benchCmdbuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }

      for (uint32_t i = 0u; i < warmupIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 0);
      for (uint32_t i = 0u; i < benchmarkIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, m_queryPool.get(), 1);
      for (uint32_t i = 0u; i < cooldownIterations; ++i)
         m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->end();

      auto                                                 semaphore   = m_device->createSemaphore(0u);
      const IQueue::SSubmitInfo::SCommandBufferInfo benchCmds[] = {{.cmdbuf = m_benchCmdbuf.get()}};
      const IQueue::SSubmitInfo::SSemaphoreInfo     signalSem[] = {
         {.semaphore = semaphore.get(), .value = 1u, .stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
      IQueue::SSubmitInfo submit = {};
      submit.commandBuffers             = benchCmds;
      submit.signalSemaphores           = signalSem;
      m_queue->submit({&submit, 1u});

      m_device->waitIdle();

      uint64_t   timestamps[2] = {};
      const auto flags         = core::bitflag(IQueryPool::RESULTS_FLAGS::_64_BIT) |
         core::bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT);
      m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, timestamps, sizeof(uint64_t), flags);

      constexpr uint32_t benchIters      = BENCH_ITERS;
      const float64_t    timestampPeriod = float64_t(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
      const float64_t    elapsed_ns      = float64_t(timestamps[1] - timestamps[0]) * timestampPeriod;
      const uint64_t     totalThreads    = uint64_t(m_dispatchGroupCount) * uint64_t(WORKGROUP_SIZE);
      const uint64_t     totalSamples    = uint64_t(benchmarkIterations) * totalThreads * uint64_t(benchIters);
      const float64_t    ps_per_sample   = elapsed_ns * 1e3 / float64_t(totalSamples);
      const float64_t    gsamples_per_s  = float64_t(totalSamples) / elapsed_ns;
      const float64_t    elapsed_ms      = elapsed_ns * 1e-6;

      m_logger->log("%12u | %-34s | %12.3f | %12.3f | %12.3f | %10u",
         ILogger::ELL_PERFORMANCE, N, name, ps_per_sample, gsamples_per_s, elapsed_ms, benchmarkIterations);
   }

   core::smart_refctd_ptr<ILogicalDevice>    m_device;
   core::smart_refctd_ptr<ILogger>          m_logger;
   core::smart_refctd_ptr<IAssetManager>     m_assetMgr;
   core::smart_refctd_ptr<IUtilities>        m_utils;
   core::smart_refctd_ptr<IGPUCommandPool>   m_cmdpool;
   core::smart_refctd_ptr<IGPUCommandBuffer> m_benchCmdbuf;
   core::smart_refctd_ptr<IQueryPool>        m_queryPool;

   // Pipelines (set up once)
   core::smart_refctd_ptr<IGPUPipelineLayout>  m_packedAliasAPplnLayout;
   core::smart_refctd_ptr<IGPUComputePipeline> m_packedAliasAPipeline;
   core::smart_refctd_ptr<IGPUPipelineLayout>  m_packedAliasBPplnLayout;
   core::smart_refctd_ptr<IGPUComputePipeline> m_packedAliasBPipeline;
   core::smart_refctd_ptr<IGPUPipelineLayout>  m_cumProbPplnLayout;
   core::smart_refctd_ptr<IGPUComputePipeline> m_cumProbPipeline;
   core::smart_refctd_ptr<IGPUPipelineLayout>  m_cumProbYoloPplnLayout;
   core::smart_refctd_ptr<IGPUComputePipeline> m_cumProbYoloPipeline;
   core::smart_refctd_ptr<IGPUPipelineLayout>  m_cumProbEytzingerPplnLayout;
   core::smart_refctd_ptr<IGPUComputePipeline> m_cumProbEytzingerPipeline;

   // Per-N data buffers (rebuilt each sweep step). pdf[] is shared between A and B.
   core::smart_refctd_ptr<IGPUBuffer> m_aliasPdfBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_packedAliasABuf;
   core::smart_refctd_ptr<IGPUBuffer> m_packedAliasBBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_cumProbBuf;
   core::smart_refctd_ptr<IGPUBuffer> m_cumProbEytzingerBuf;

   // Shared
   core::smart_refctd_ptr<IGPUBuffer> m_outputBuf;
   IQueue*                            m_queue              = nullptr;
   IPhysicalDevice*                   m_physicalDevice     = nullptr;
   uint32_t                                  m_dispatchGroupCount = 0;
   uint32_t                                  m_currentN           = 0;
   uint32_t                                  m_aliasTableN        = 0;
};

#endif
