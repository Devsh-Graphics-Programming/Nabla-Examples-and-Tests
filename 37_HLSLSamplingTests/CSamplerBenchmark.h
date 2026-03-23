// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_BENCHMARK_INCLUDED_
#define _NBL_EXAMPLES_TESTS_37_SAMPLING_SAMPLER_BENCHMARK_INCLUDED_

#include <nabla.h>
#include "nbl/examples/examples.hpp"

using namespace nbl;

// Measures GPU execution time of a sampler shader using GPU timestamp queries.
class CSamplerBenchmark
{
public:
	struct SetupData
	{
		core::smart_refctd_ptr<video::ILogicalDevice> device;
		core::smart_refctd_ptr<video::CVulkanConnection> api;
		core::smart_refctd_ptr<asset::IAssetManager> assetMgr;
		core::smart_refctd_ptr<system::ILogger> logger;
		video::IPhysicalDevice* physicalDevice;
		uint32_t computeFamilyIndex;
		std::string shaderKey;
		uint32_t dispatchGroupCount;  // workgroup count = testBatchCount
		uint32_t samplesPerDispatch;  // dispatchGroupCount * WorkgroupSize (BenchIters is internal to the shader)
		size_t inputBufferBytes;      // sizeof(InputType) * samplesPerDispatch
		size_t outputBufferBytes;     // sizeof(ResultType) * samplesPerDispatch
	};

	void setup(const SetupData& data)
	{
		m_device = data.device;
		m_logger = data.logger;
		m_dispatchGroupCount = data.dispatchGroupCount;

		// Command pool + 3 command buffers: benchmark (multi-submit), before/after timestamp
		m_cmdpool = m_device->createCommandPool(data.computeFamilyIndex, video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		if (!m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_benchmarkCmdbuf))
			m_logger->log("CSamplerBenchmark: failed to create benchmark cmdbuf", system::ILogger::ELL_ERROR);
		if (!m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampBeforeCmdbuf))
			m_logger->log("CSamplerBenchmark: failed to create timestamp-before cmdbuf", system::ILogger::ELL_ERROR);
		if (!m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampAfterCmdbuf))
			m_logger->log("CSamplerBenchmark: failed to create timestamp-after cmdbuf", system::ILogger::ELL_ERROR);

		// Timestamp query pool (2 queries: before and after)
		{
			video::IQueryPool::SCreationParams qparams = {};
			qparams.queryType = video::IQueryPool::TYPE::TIMESTAMP;
			qparams.queryCount = 2;
			qparams.pipelineStatisticsFlags = video::IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
			m_queryPool = m_device->createQueryPool(qparams);
			if (!m_queryPool)
				m_logger->log("CSamplerBenchmark: failed to create query pool", system::ILogger::ELL_ERROR);
		}

		// Load and compile shader
		core::smart_refctd_ptr<asset::IShader> shader;
		{
			asset::IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = "app_resources";
			auto bundle = data.assetMgr->getAsset(data.shaderKey, lp);
			const auto assets = bundle.getContents();
			if (assets.empty())
			{
				m_logger->log("CSamplerBenchmark: failed to load shader", system::ILogger::ELL_ERROR);
				return;
			}
			auto source = asset::IAsset::castDown<asset::IShader>(assets[0]);
			shader = m_device->compileShader({ source.get() });
		}

		// Descriptor set layout: binding 0 = input SSBO, binding 1 = output SSBO
		video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
			{ .binding = 0, .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
			  .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			  .stageFlags = ShaderStage::ESS_COMPUTE, .count = 1 },
			{ .binding = 1, .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
			  .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
			  .stageFlags = ShaderStage::ESS_COMPUTE, .count = 1 }
		};
		auto dsLayout = m_device->createDescriptorSetLayout(bindings);

		m_pplnLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(dsLayout));

		{
			video::IGPUComputePipeline::SCreationParams pparams = {};
			pparams.layout = m_pplnLayout.get();
			pparams.shader.entryPoint = "main";
			pparams.shader.shader = shader.get();
         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
         {
            pparams.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS | IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
         }
			if (!m_device->createComputePipelines(nullptr, { &pparams, 1 }, &m_pipeline))
				m_logger->log("CSamplerBenchmark: failed to create compute pipeline", system::ILogger::ELL_ERROR);

         if (m_device->getEnabledFeatures().pipelineExecutableInfo)
               m_executableReport = system::to_string(m_pipeline->getExecutableInfo());
		}

		// Allocate input buffer (host-visible, zero-filled, correctness irrelevant for benchmarking)
		core::smart_refctd_ptr<video::IGPUBuffer> inputBuf;
		{
			video::IGPUBuffer::SCreationParams bparams = {};
			bparams.size = data.inputBufferBytes;
			bparams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			inputBuf = m_device->createBuffer(std::move(bparams));
			video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = inputBuf->getMemoryReqs();
			reqs.memoryTypeBits &= data.physicalDevice->getHostVisibleMemoryTypeBits();
			m_inputAlloc = m_device->allocate(reqs, inputBuf.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_inputAlloc.isValid())
				m_logger->log("CSamplerBenchmark: failed to allocate input buffer memory", system::ILogger::ELL_ERROR);
			if (m_inputAlloc.memory->map({ 0ull, m_inputAlloc.memory->getAllocationSize() }, video::IDeviceMemoryAllocation::EMCAF_READ))
			{
				std::memset(m_inputAlloc.memory->getMappedPointer(), 0, m_inputAlloc.memory->getAllocationSize());
				m_inputAlloc.memory->unmap();
			}
		}

		// Allocate output buffer (host-visible, GPU writes garbage, never read back)
		core::smart_refctd_ptr<video::IGPUBuffer> outputBuf;
		{
			video::IGPUBuffer::SCreationParams bparams = {};
			bparams.size = data.outputBufferBytes;
			bparams.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			outputBuf = m_device->createBuffer(std::move(bparams));
			video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuf->getMemoryReqs();
			reqs.memoryTypeBits &= data.physicalDevice->getHostVisibleMemoryTypeBits();
			m_outputAlloc = m_device->allocate(reqs, outputBuf.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_outputAlloc.isValid())
				m_logger->log("CSamplerBenchmark: failed to allocate output buffer memory", system::ILogger::ELL_ERROR);
		}

		// Descriptor set: bind both buffers
		auto pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, { &dsLayout.get(), 1 });
		m_ds = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
		{
			video::IGPUDescriptorSet::SDescriptorInfo info[2];
			info[0].desc = core::smart_refctd_ptr(inputBuf);
			info[0].info.buffer = { .offset = 0, .size = data.inputBufferBytes };
			info[1].desc = core::smart_refctd_ptr(outputBuf);
			info[1].info.buffer = { .offset = 0, .size = data.outputBufferBytes };
			video::IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
				{ .dstSet = m_ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &info[0] },
				{ .dstSet = m_ds.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &info[1] }
			};
			m_device->updateDescriptorSets(writes, {});
		}

		m_queue = m_device->getQueue(data.computeFamilyIndex, 0);
		m_samplesPerDispatch = data.samplesPerDispatch;
		m_physicalDevice = data.physicalDevice;
	}

	void logPipelineReport(const std::string& name) const
   {
		if (!m_executableReport.empty())
			m_logger->log("%s Sampler Benchmark Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, name.c_str(), m_executableReport.c_str());
	}

	// Runs warmupIterations submits (unclocked), then benchmarkIterations submits under GPU timestamps.
	void run(const std::string& samplerName, uint32_t warmupIterations = 1000, uint32_t benchmarkIterations = 20000)
	{
		m_device->waitIdle();
		recordBenchmarkCmdBuf();
		recordTimestampCmdBufs();

		auto semaphore = m_device->createSemaphore(0u);
		uint64_t semCounter = 0u;

		const video::IQueue::SSubmitInfo::SCommandBufferInfo benchCmds[] = { {.cmdbuf = m_benchmarkCmdbuf.get()} };
		const video::IQueue::SSubmitInfo::SCommandBufferInfo beforeCmds[] = { {.cmdbuf = m_timestampBeforeCmdbuf.get()} };
		const video::IQueue::SSubmitInfo::SCommandBufferInfo afterCmds[] = { {.cmdbuf = m_timestampAfterCmdbuf.get()} };

		// Chains submissions via a timeline semaphore so they execute strictly in order
		auto submitSerial = [&](const video::IQueue::SSubmitInfo::SCommandBufferInfo* cmds, uint32_t count)
		{
			const video::IQueue::SSubmitInfo::SSemaphoreInfo waitSem[] = {
				{.semaphore = semaphore.get(), .value = semCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}
			};
			const video::IQueue::SSubmitInfo::SSemaphoreInfo signalSem[] = {
				{.semaphore = semaphore.get(), .value = ++semCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}
			};
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

		const float64_t timestampPeriod = float64_t(m_physicalDevice->getLimits().timestampPeriodInNanoSeconds);
		const float64_t elapsed_ns      = float64_t(timestamps[1] - timestamps[0]) * timestampPeriod;
		const uint64_t total_samples    = uint64_t(benchmarkIterations) * uint64_t(m_samplesPerDispatch);
		const float64_t ns_per_sample   = elapsed_ns / float64_t(total_samples);
		const float64_t msamples_per_s = (float64_t(total_samples) / elapsed_ns) * 1e3;
		const float64_t elapsed_ms = elapsed_ns * 1e-6;

		m_logger->log("[Benchmark] %s: %.5f ns/sample  |  %.2f MSamples/s  |  %.3f ms total",
			system::ILogger::ELL_PERFORMANCE,
			samplerName.c_str(), ns_per_sample, msamples_per_s, elapsed_ms);
	}

private:
	void recordBenchmarkCmdBuf()
	{
		m_benchmarkCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
		m_benchmarkCmdbuf->begin(video::IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT);
		m_benchmarkCmdbuf->bindComputePipeline(m_pipeline.get());
		m_benchmarkCmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
		m_benchmarkCmdbuf->dispatch(m_dispatchGroupCount, 1, 1);
		m_benchmarkCmdbuf->end();
	}

	void recordTimestampCmdBufs()
	{
		m_timestampBeforeCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
		m_timestampBeforeCmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		m_timestampBeforeCmdbuf->resetQueryPool(m_queryPool.get(), 0, 2);
		m_timestampBeforeCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 0);
		m_timestampBeforeCmdbuf->end();

		m_timestampAfterCmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
		m_timestampAfterCmdbuf->begin(video::IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		m_timestampAfterCmdbuf->writeTimestamp(asset::PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 1);
		m_timestampAfterCmdbuf->end();
	}

	core::smart_refctd_ptr<video::ILogicalDevice>       m_device;
	core::smart_refctd_ptr<system::ILogger>             m_logger;
	core::smart_refctd_ptr<video::IGPUCommandPool>      m_cmdpool;
	core::smart_refctd_ptr<video::IGPUCommandBuffer>    m_benchmarkCmdbuf;
	core::smart_refctd_ptr<video::IGPUCommandBuffer>    m_timestampBeforeCmdbuf;
	core::smart_refctd_ptr<video::IGPUCommandBuffer>    m_timestampAfterCmdbuf;
	core::smart_refctd_ptr<video::IQueryPool>           m_queryPool;
	core::smart_refctd_ptr<video::IGPUPipelineLayout>   m_pplnLayout;
	core::smart_refctd_ptr<video::IGPUComputePipeline>  m_pipeline;
	core::smart_refctd_ptr<video::IGPUDescriptorSet>    m_ds;
	video::IDeviceMemoryAllocator::SAllocation          m_inputAlloc  = {};
	video::IDeviceMemoryAllocator::SAllocation          m_outputAlloc = {};
	video::IQueue*                                      m_queue              = nullptr;
	video::IPhysicalDevice*                             m_physicalDevice     = nullptr;
	uint32_t                                            m_dispatchGroupCount = 0;
	uint32_t                                            m_samplesPerDispatch = 0;
	std::string                                         m_executableReport;
};

#endif
