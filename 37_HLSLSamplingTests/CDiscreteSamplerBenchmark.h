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
// Builds both tables from the same weight distribution, uploads via BDA buffers,
// and measures GPU throughput using timestamp queries.
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
      uint32_t computeFamilyIndex;
      uint32_t dispatchGroupCount;
      uint32_t tableSize;
   };

   void setup(const SetupData& data)
   {
      m_device = data.device;
      m_logger = data.logger;
      m_dispatchGroupCount = data.dispatchGroupCount;
      m_tableSize = data.tableSize;
      m_physicalDevice = data.physicalDevice;

      m_queue = m_device->getQueue(data.computeFamilyIndex, 0);

      // Command pool + buffers
      m_cmdpool = m_device->createCommandPool(data.computeFamilyIndex, video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
      m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_benchCmdbuf);
      m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampBeforeCmdbuf);
      m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampAfterCmdbuf);

      // Timestamp query pool
      {
         video::IQueryPool::SCreationParams qp = {};
         qp.queryType = video::IQueryPool::TYPE::TIMESTAMP;
         qp.queryCount = 2;
         qp.pipelineStatisticsFlags = video::IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
         m_queryPool = m_device->createQueryPool(qp);
      }

      // Generate random weights
      const uint32_t N = m_tableSize;
      std::vector<float> weights(N);
      std::mt19937 rng(42);
      std::uniform_real_distribution<float> dist(0.001f, 100.0f);
      for (uint32_t i = 0; i < N; i++)
         weights[i] = dist(rng);

      // Build alias table
      std::vector<float> aliasProb(N);
      std::vector<uint32_t> aliasIdx(N);
      std::vector<float> aliasPdf(N);
      std::vector<uint32_t> workspace(N);
      nbl::hlsl::sampling::AliasTableBuilder<float>::build(weights.data(), N, aliasProb.data(), aliasIdx.data(), aliasPdf.data(), workspace.data());

      // Build cumulative probability table
      std::vector<float> cumProb(N - 1);
      nbl::hlsl::sampling::computeNormalizedCumulativeHistogram(weights.data(), N, cumProb.data());

      // Create BDA buffers and upload data
      auto createBdaBuffer = [&](const void* srcData, size_t bytes) -> core::smart_refctd_ptr<video::IGPUBuffer>
      {
         video::IGPUBuffer::SCreationParams bp = {};
         bp.size = bytes;
         bp.usage = core::bitflag(video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT) |
            video::IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
         auto buf = m_device->createBuffer(std::move(bp));

         video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = buf->getMemoryReqs();
         reqs.memoryTypeBits &= data.physicalDevice->getHostVisibleMemoryTypeBits();
         auto alloc = m_device->allocate(reqs, buf.get(), video::IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

         const auto allocSize = alloc.memory->getAllocationSize();
         if (alloc.memory->map({0ull, allocSize}, video::IDeviceMemoryAllocation::EMCAF_WRITE))
         {
            std::memcpy(alloc.memory->getMappedPointer(), srcData, bytes);
            // Flush so GPU can see the written data
            video::ILogicalDevice::MappedMemoryRange flushRange(alloc.memory.get(), 0ull, allocSize);
            m_device->flushMappedMemoryRanges(1u, &flushRange);
            alloc.memory->unmap();
         }
         return buf;
      };

      const uint32_t totalThreads = m_dispatchGroupCount * WorkgroupSize;

      // Alias table buffers
      m_aliasProbBuf = createBdaBuffer(aliasProb.data(), N * sizeof(float));
      m_aliasIdxBuf = createBdaBuffer(aliasIdx.data(), N * sizeof(uint32_t));
      m_aliasPdfBuf = createBdaBuffer(aliasPdf.data(), N * sizeof(float));

      // CDF buffer
      m_cumProbBuf = createBdaBuffer(cumProb.data(), (N - 1) * sizeof(float));

      // Shared output buffer
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

      // Create pipelines (push constants only, no descriptor sets)
      auto loadShader = [&](const std::string& key)
      {
         asset::IAssetLoader::SAssetLoadParams lp = {};
         lp.logger = m_logger.get();
         lp.workingDirectory = "app_resources";
         auto bundle = data.assetMgr->getAsset(key, lp);
         auto source = asset::IAsset::castDown<asset::IShader>(bundle.getContents()[0]);
         return m_device->compileShader({.source = source.get()});
      };

      // Alias table pipeline
      {
         const asset::SPushConstantRange pcRange = {
            .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
            .offset = 0,
            .size = sizeof(AliasTablePushConstants)};
         auto layout = m_device->createPipelineLayout({&pcRange, 1});
         if (!layout)
            m_logger->log("CDiscreteSamplerBenchmark: failed to create alias pipeline layout", system::ILogger::ELL_ERROR);
         video::IGPUComputePipeline::SCreationParams pp = {};
         pp.layout = layout.get();
         auto shader = loadShader(data.aliasShaderKey);
         if (!shader)
            m_logger->log("CDiscreteSamplerBenchmark: failed to load alias shader", system::ILogger::ELL_ERROR);
         pp.shader.shader = shader.get();
         pp.shader.entryPoint = "main";

         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         {
            pp.flags |= video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
         }

         if (!m_device->createComputePipelines(nullptr, {&pp, 1}, &m_aliasPipeline))
            m_logger->log("CDiscreteSamplerBenchmark: failed to create alias compute pipeline", system::ILogger::ELL_ERROR);

         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         {
            auto report = system::to_string(m_aliasPipeline->getExecutableInfo());
            m_logger->log("Alias Table Sampling Pipeline Executable Report:\n%s", system::ILogger::ELL_PERFORMANCE, report.c_str());
         }
         m_aliasPplnLayout = std::move(layout);
      }

      // CDF pipeline
      {
         const asset::SPushConstantRange pcRange = {
            .stageFlags = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE,
            .offset = 0,
            .size = sizeof(CumProbPushConstants)};
         auto layout = m_device->createPipelineLayout({&pcRange, 1});
         if (!layout)
            m_logger->log("CDiscreteSamplerBenchmark: failed to create cumprob pipeline layout", system::ILogger::ELL_ERROR);
         video::IGPUComputePipeline::SCreationParams pp = {};
         pp.layout = layout.get();
         auto shader = loadShader(data.cumProbShaderKey);
         if (!shader)
            m_logger->log("CDiscreteSamplerBenchmark: failed to load cumprob shader", system::ILogger::ELL_ERROR);
         pp.shader.shader = shader.get();
         pp.shader.entryPoint = "main";
         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         {
            pp.flags |= video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | video::IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
         }
         if (!m_device->createComputePipelines(nullptr, {&pp, 1}, &m_cumProbPipeline))
            m_logger->log("CDiscreteSamplerBenchmark: failed to create cumprob compute pipeline", system::ILogger::ELL_ERROR);
         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         {
            auto report = system::to_string(m_cumProbPipeline->getExecutableInfo());
            m_logger->log("Cumulative Probability Sampling Pipeline Executable Report:\n%s", system::ILogger::ELL_PERFORMANCE, report.c_str());
         }
         m_cumProbPplnLayout = std::move(layout);
      }
   }

   void run(uint32_t warmupIterations = 1000, uint32_t benchmarkIterations = 20000)
   {
      m_logger->log("=== GPU Discrete Sampler Benchmark (N=%u) ===", system::ILogger::ELL_PERFORMANCE, m_tableSize);

      runSingle("AliasTable", m_aliasPipeline, m_aliasPplnLayout, true, warmupIterations, benchmarkIterations);
      runSingle("CumulativeProbability", m_cumProbPipeline, m_cumProbPplnLayout, false, warmupIterations, benchmarkIterations);
   }

   private:
   void runSingle(const char* name, const core::smart_refctd_ptr<video::IGPUComputePipeline>& pipeline, const core::smart_refctd_ptr<video::IGPUPipelineLayout>& layout, bool isAlias, uint32_t warmupIterations, uint32_t benchmarkIterations)
   {
      m_device->waitIdle();

      // Record benchmark command buffer
      m_benchCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
      m_benchCmdbuf->begin(video::IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT);
      m_benchCmdbuf->bindComputePipeline(pipeline.get());

      if (isAlias)
      {
         AliasTablePushConstants pc = {};
         pc.probAddress = m_aliasProbBuf->getDeviceAddress();
         pc.aliasAddress = m_aliasIdxBuf->getDeviceAddress();
         pc.pdfAddress = m_aliasPdfBuf->getDeviceAddress();
         pc.outputAddress = m_outputBuf->getDeviceAddress();
         pc.tableSize = m_tableSize;
         m_benchCmdbuf->pushConstants(layout.get(), asset::IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }
      else
      {
         CumProbPushConstants pc = {};
         pc.cumProbAddress = m_cumProbBuf->getDeviceAddress();
         pc.outputAddress = m_outputBuf->getDeviceAddress();
         pc.tableSize = m_tableSize;
         m_benchCmdbuf->pushConstants(layout.get(), asset::IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
      }

      m_benchCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
      m_benchCmdbuf->end();

      // Record timestamp command buffers
      m_timestampBeforeCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
      m_timestampBeforeCmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      m_timestampBeforeCmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);
      m_timestampBeforeCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 0);
      m_timestampBeforeCmdbuf->end();

      m_timestampAfterCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
      m_timestampAfterCmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
      m_timestampAfterCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 1);
      m_timestampAfterCmdbuf->end();

      auto semaphore = m_device->createSemaphore(0u);
      uint64_t semCounter = 0u;

      const video::IQueue::SSubmitInfo::SCommandBufferInfo benchCmds[] = {{.cmdbuf = m_benchCmdbuf.get()}};
      const video::IQueue::SSubmitInfo::SCommandBufferInfo beforeCmds[] = {{.cmdbuf = m_timestampBeforeCmdbuf.get()}};
      const video::IQueue::SSubmitInfo::SCommandBufferInfo afterCmds[] = {{.cmdbuf = m_timestampAfterCmdbuf.get()}};

      auto submitSerial = [&](const video::IQueue::SSubmitInfo::SCommandBufferInfo* cmds, uint32_t count)
      {
         const video::IQueue::SSubmitInfo::SSemaphoreInfo waitSem[] = {
            {.semaphore = semaphore.get(), .value = semCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
         const video::IQueue::SSubmitInfo::SSemaphoreInfo signalSem[] = {
            {.semaphore = semaphore.get(), .value = ++semCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
         video::IQueue::SSubmitInfo submit = {};
         submit.commandBuffers = {cmds, count};
         submit.waitSemaphores = waitSem;
         submit.signalSemaphores = signalSem;
         m_queue->submit({&submit, 1u});
      };

      for (uint32_t i = 0u; i < warmupIterations; ++i)
         submitSerial(benchCmds, 1u);

      submitSerial(beforeCmds, 1u);
      for (uint32_t i = 0u; i < benchmarkIterations; ++i)
         submitSerial(benchCmds, 1u);
      submitSerial(afterCmds, 1u);

      m_device->waitIdle();

      uint64_t timestamps[2] = {};
      const auto flags = core::bitflag(video::IQueryPool::RESULTS_FLAGS::_64_BIT) |
         core::bitflag(video::IQueryPool::RESULTS_FLAGS::WAIT_BIT);
      m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, timestamps, sizeof(uint64_t), flags);

      constexpr uint32_t benchIters = 4096; // must match -DBENCH_ITERS in CMakeLists.txt
      const float64_t timestampPeriod = float64_t(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
      const float64_t elapsed_ns = float64_t(timestamps[1] - timestamps[0]) * timestampPeriod;
      const uint64_t totalThreads = uint64_t(m_dispatchGroupCount) * uint64_t(WorkgroupSize);
      const uint64_t totalSamples = uint64_t(benchmarkIterations) * totalThreads * uint64_t(benchIters);
      const float64_t ns_per_sample = elapsed_ns / float64_t(totalSamples);
      const float64_t msamples_per_s = (float64_t(totalSamples) / elapsed_ns) * 1e3;
      const float64_t elapsed_ms = elapsed_ns * 1e-6;

      m_logger->log("[Benchmark] %s: %.5f ns/sample  |  %.2f MSamples/s  |  %.3f ms total", system::ILogger::ELL_PERFORMANCE, name, ns_per_sample, msamples_per_s, elapsed_ms);
   }

   core::smart_refctd_ptr<video::ILogicalDevice> m_device;
   core::smart_refctd_ptr<system::ILogger> m_logger;
   core::smart_refctd_ptr<video::IGPUCommandPool> m_cmdpool;
   core::smart_refctd_ptr<video::IGPUCommandBuffer> m_benchCmdbuf;
   core::smart_refctd_ptr<video::IGPUCommandBuffer> m_timestampBeforeCmdbuf;
   core::smart_refctd_ptr<video::IGPUCommandBuffer> m_timestampAfterCmdbuf;
   core::smart_refctd_ptr<video::IQueryPool> m_queryPool;

   // Alias table
   core::smart_refctd_ptr<video::IGPUPipelineLayout> m_aliasPplnLayout;
   core::smart_refctd_ptr<video::IGPUComputePipeline> m_aliasPipeline;
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasProbBuf;
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasIdxBuf;
   core::smart_refctd_ptr<video::IGPUBuffer> m_aliasPdfBuf;

   // Cumulative probability
   core::smart_refctd_ptr<video::IGPUPipelineLayout> m_cumProbPplnLayout;
   core::smart_refctd_ptr<video::IGPUComputePipeline> m_cumProbPipeline;
   core::smart_refctd_ptr<video::IGPUBuffer> m_cumProbBuf;

   // Shared
   core::smart_refctd_ptr<video::IGPUBuffer> m_outputBuf;
   video::IQueue* m_queue = nullptr;
   video::IPhysicalDevice* m_physicalDevice = nullptr;
   uint32_t m_dispatchGroupCount = 0;
   uint32_t m_tableSize = 0;
};

#endif
