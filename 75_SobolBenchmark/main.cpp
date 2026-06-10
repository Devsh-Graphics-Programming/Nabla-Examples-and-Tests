// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nabla.h>

#include "nbl/examples/examples.hpp"
#include "nbl/examples/common/CCachedOwenScrambledSequence.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "benchmarks/CSobolBenchmark.h"

#include <bit>


using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::examples;


class SobolBenchmarkApp final : public application_templates::BasicMultiQueueApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t  = BuiltinResourcesApplication;

	// Seed image: 512x512 R32G32_UINT.
	static constexpr uint32_t SeedDim   = 512u;
	static constexpr size_t SeedEntries = size_t(SeedDim) * size_t(SeedDim);

public:
	SobolBenchmarkApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		: IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
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

		// Smaller dispatch than 37's light-sampler bench since each thread does Depth*Triplets*Components
		// matrix-muls per outer iteration (so the inner work is much heavier per "sample").
		constexpr uint32_t testBatchCount          = 256;
		constexpr uint32_t benchWorkgroupSize      = WORKGROUP_SIZE;
		constexpr uint32_t totalThreadsPerDispatch = testBatchCount * benchWorkgroupSize;
		constexpr uint32_t iterationsPerThread     = BENCH_ITERS;
		constexpr uint32_t benchSamplesPerDispatch = totalThreadsPerDispatch * iterationsPerThread;

		// BDA sample buffer holds (Depth * Triplets) quantized atoms (each packing 3
		// 1D Sobol dimensions), with BENCH_ITERS samples per atom.
		static_assert((BENCH_ITERS & (BENCH_ITERS - 1)) == 0, "BENCH_ITERS must be a power of two");
		constexpr uint32_t Triplets            = 2u;
		constexpr uint32_t Components          = 3u;
		constexpr uint32_t SampleDimensions    = uint32_t(SOBOL_DEPTH) * Triplets;
		constexpr uint32_t TotalSobolDims      = SampleDimensions * Components;
		constexpr uint32_t SequenceSamplesLog2 = std::countr_zero(uint32_t(BENCH_ITERS));

		constexpr size_t benchInputBytes  = sizeof(uint32_t); // Generic input slot, sized minimally since no current variant reads it.
		constexpr size_t benchOutputBytes = sizeof(uint32_t) * totalThreadsPerDispatch;

		// Match example 40: random-fill the per-path xoroshiro seed buffer on the CPU,
		// build the Owen-scrambled quantized Sobol sequence via the cache helper, and
		// stage-upload both via IUtilities.
		auto* const queue = getComputeQueue();

		// 512x512 R32G32_UINT storage image holding xoroshiro seeds, modelled after
		// the path tracer's gScrambleKey image (ex 31, ex 40). Tiled memory layout +
		// texture cache, vs the linear SSBO path we had before.
		smart_refctd_ptr<IGPUImageView> seedImageView;
		{
			IGPUImage::SCreationParams imgParams = {};
			imgParams.type        = IGPUImage::E_TYPE::ET_2D;
			imgParams.samples     = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			imgParams.format      = E_FORMAT::EF_R32G32_UINT;
			imgParams.extent      = { SeedDim, SeedDim, 1 };
			imgParams.mipLevels   = 1;
			imgParams.arrayLayers = 1;
			imgParams.usage       = bitflag(IGPUImage::EUF_TRANSFER_DST_BIT) | IGPUImage::EUF_STORAGE_BIT;
			imgParams.viewFormats.set(E_FORMAT::EF_R32G32_UINT);
			auto seedImage = m_device->createImage(std::move(imgParams));
			auto reqs      = seedImage->getMemoryReqs();
			reqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			if (!m_device->allocate(reqs, seedImage.get(), IDeviceMemoryAllocation::EMAF_NONE).isValid())
			{
				m_logger->log("Failed to allocate seed image", ILogger::ELL_ERROR);
				return false;
			}
			seedImage->setObjectDebugName("Sobol Bench Scramble Seeds");

			core::vector<hlsl::uint32_t2> seedData(SeedEntries);
			{
				core::RandomSampler rng(0xbadc0ffeu);
				for (auto& el : seedData)
					el = { rng.nextSample(), rng.nextSample() };
			}

			auto sem = m_device->createSemaphore(0);
			auto cmdpool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			smart_refctd_ptr<IGPUCommandBuffer> scratch;
			cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &scratch);
			scratch->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			using image_barrier_t = IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t;
			const IGPUImage::SSubresourceRange subRange = {
				.aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
				.baseMipLevel = 0, .levelCount = 1,
				.baseArrayLayer = 0, .layerCount = 1
			};
			{
				const image_barrier_t toGeneral = {
					.barrier = { .dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, .srcAccessMask = ACCESS_FLAGS::NONE,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
					} },
					.image = seedImage.get(), .subresourceRange = subRange,
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED, .newLayout = IGPUImage::LAYOUT::GENERAL
				};
				scratch->pipelineBarrier(EDF_NONE, { .imgBarriers = {&toGeneral, 1} });
			}

			IQueue::SSubmitInfo::SCommandBufferInfo scratchCmds[] = { {.cmdbuf = scratch.get()} };
			SIntendedSubmitInfo submitInfo = {
				.queue = queue,
				.waitSemaphores = {},
				.prevCommandBuffers = {},
				.scratchCommandBuffers = scratchCmds,
				.scratchSemaphore = { .semaphore = sem.get(), .value = 0, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS }
			};

			const IImage::SBufferCopy region = {
				.bufferOffset      = 0,
				.bufferRowLength   = SeedDim,
				.bufferImageHeight = SeedDim,
				.imageSubresource  = { .aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
				.imageExtent       = { SeedDim, SeedDim, 1 }
			};
			m_utils->updateImageViaStagingBuffer(submitInfo, seedData.data(), E_FORMAT::EF_R32G32_UINT, seedImage.get(), IGPUImage::LAYOUT::GENERAL, { &region, 1 });

			{
				const image_barrier_t toShader = {
					.barrier = { .dep = {
						.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, .dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
					} },
					.image = seedImage.get(), .subresourceRange = subRange,
					.oldLayout = IGPUImage::LAYOUT::GENERAL, .newLayout = IGPUImage::LAYOUT::GENERAL
				};
				scratch->pipelineBarrier(EDF_NONE, { .imgBarriers = {&toShader, 1} });
			}

			scratch->end();
			const IQueue::SSubmitInfo::SSemaphoreInfo signalSem[] = { { .semaphore = sem.get(), .value = 1, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS } };
			IQueue::SSubmitInfo finalSubmit = { .commandBuffers = scratchCmds, .signalSemaphores = signalSem };
			queue->submit({ &finalSubmit, 1u });
			m_device->waitIdle();

			IGPUImageView::SCreationParams viewParams = {};
			viewParams.image            = seedImage;
			viewParams.viewType         = IGPUImageView::E_TYPE::ET_2D;
			viewParams.format           = E_FORMAT::EF_R32G32_UINT;
			viewParams.subresourceRange = subRange;
			viewParams.subUsages        = IGPUImage::EUF_STORAGE_BIT;
			seedImageView               = m_device->createImageView(std::move(viewParams));
			if (!seedImageView)
			{
				m_logger->log("Failed to create seed image view", ILogger::ELL_ERROR);
				return false;
			}
		}

		smart_refctd_ptr<IGPUBuffer> sampleBuffer;
		{
			auto sequence = nbl::examples::CCachedOwenScrambledSequence::create({
				.cachePath = (sharedOutputCWD / nbl::examples::CCachedOwenScrambledSequence::SCreationParams::DefaultFilename).string(),
				.assMan    = m_assetMgr.get(),
				.header    = {
					.maxSamplesLog2 = SequenceSamplesLog2,
					.maxDimensions  = TotalSobolDims
				}
			});
			if (!sequence)
			{
				m_logger->log("Failed to build cached Owen-scrambled Sobol sequence", ILogger::ELL_ERROR);
				return false;
			}
			auto* const seqBufferCPU = sequence->getBuffer();
			m_utils->createFilledDeviceLocalBufferOnDedMem(SIntendedSubmitInfo{ .queue = queue }, IGPUBuffer::SCreationParams{ seqBufferCPU->getCreationParams() }, seqBufferCPU->getPointer()).move_into(sampleBuffer);
			if (!sampleBuffer)
			{
				m_logger->log("Failed to allocate/fill sample BDA buffer", ILogger::ELL_ERROR);
				return false;
			}
			sampleBuffer->setObjectDebugName("Low Discrepancy Sequence");
		}

		const uint64_t pSampleBuffer = sampleBuffer->getDeviceAddress();

		struct BenchEntry
		{
			CSobolBenchmark bench;
			std::string     name;
		};
		std::vector<BenchEntry> benchmarks;

		auto addBench = [&](const char* name, const std::string& shaderKey)
		{
			auto& entry = benchmarks.emplace_back();
			entry.name  = name;

			CSobolBenchmark::SetupData data;
			data.device              = m_device;
			data.api                 = m_api;
			data.assetMgr            = m_assetMgr;
			data.logger              = m_logger;
			data.physicalDevice      = m_physicalDevice;
			data.computeFamilyIndex  = getComputeQueue()->getFamilyIndex();
			data.shaderKey           = shaderKey;
			data.dispatchGroupCount  = testBatchCount;
			data.samplesPerDispatch  = benchSamplesPerDispatch;
			data.inputBufferBytes    = benchInputBytes;
			data.outputBufferBytes   = benchOutputBytes;
			data.seedImageView       = seedImageView;
			data.pSampleBuffer       = pSampleBuffer;
			data.sequenceSamplesLog2 = SequenceSamplesLog2;
			// Slight stagger so lanes within a subgroup pull two distinct matrix
			// indices instead of one broadcast load, modelling SER/wavefront packing
			// without the wild full-divergence cost of a larger mask. RR break left
			// off so the bench measures the matrix-mul cost end-to-end.
			data.depthStaggerMask    = 0x1u;
			data.probAtDepth0        = 0.0f;
			data.probAtDepthMax      = 0.0f;
			entry.bench.setup(data);
		};

		addBench("Sobol Quantized BDA", nbl::this_example::builtin::build::get_spirv_key<"sobol_bench_quantized">(m_device.get()));
		addBench("Sobol RowMajor",      nbl::this_example::builtin::build::get_spirv_key<"sobol_bench_row_major">(m_device.get()));
		addBench("Sobol ColMajor",      nbl::this_example::builtin::build::get_spirv_key<"sobol_bench_col_major">(m_device.get()));

		for (auto& entry : benchmarks)
			entry.bench.logPipelineReport(entry.name);

		constexpr uint32_t warmupDispatches = 100;
		constexpr uint32_t benchDispatches  = 500;
		m_logger->log("=== GPU Sobol Benchmarks (%u dispatches, %u threads/dispatch, %u outer iters/thread, inner work = DEPTH * 6 ops) ===",
			ILogger::ELL_PERFORMANCE, benchDispatches, totalThreadsPerDispatch, iterationsPerThread);
		m_logger->log("            %-28s | %12s | %12s | %12s",
			ILogger::ELL_PERFORMANCE, "Variant", "ps/sample", "GSamples/s", "ms total");
		for (auto& entry : benchmarks)
			entry.bench.run(entry.name, warmupDispatches, benchDispatches);

		return true;
	}

	bool onAppTerminated() override
	{
		m_device->waitIdle();
		return device_base_t::onAppTerminated();
	}

	void workLoopBody() override {}

	bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(SobolBenchmarkApp)
