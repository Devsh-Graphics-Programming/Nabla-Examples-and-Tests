#include "nbl/examples/examples.hpp"
#include <chrono>
#include <thread>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::examples;

class ImageUploadBenchmarkApp final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = BuiltinResourcesApplication;

public:
	ImageUploadBenchmarkApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		constexpr uint32_t TILE_SIZE = 128;
		constexpr uint32_t TILE_BYTES_PER_PIXEL = 4;
		constexpr uint32_t TILE_SIZE_BYTES = TILE_SIZE * TILE_SIZE * TILE_BYTES_PER_PIXEL;
		constexpr uint32_t STAGING_BUFFER_SIZE = 64 * 1024 * 1024;
		constexpr uint32_t FRAMES_IN_FLIGHT = 4;
		constexpr uint32_t TILES_PER_FRAME = STAGING_BUFFER_SIZE / (TILE_SIZE_BYTES * FRAMES_IN_FLIGHT);
		constexpr uint32_t TOTAL_FRAMES = 1000;

		m_logger->log("GPU Memory Transfer Benchmark", ILogger::ELL_PERFORMANCE);
		m_logger->log("Tile size: %ux%u (%u KB)", ILogger::ELL_PERFORMANCE, TILE_SIZE, TILE_SIZE, TILE_SIZE_BYTES / 1024);
		m_logger->log("Staging buffer: %u MB", ILogger::ELL_PERFORMANCE, STAGING_BUFFER_SIZE / (1024 * 1024));
		m_logger->log("Tiles per frame: %u", ILogger::ELL_PERFORMANCE, TILES_PER_FRAME);
		m_logger->log("Frames in flight: %u", ILogger::ELL_PERFORMANCE, FRAMES_IN_FLIGHT);

		uint32_t hostVisibleBits = m_physicalDevice->getHostVisibleMemoryTypeBits();
		uint32_t deviceLocalBits = m_physicalDevice->getDeviceLocalMemoryTypeBits();
		uint32_t hostCachedBits = m_physicalDevice->getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_CACHED_BIT);

		uint32_t hostVisibleOnlyBits = hostVisibleBits & ~deviceLocalBits & ~hostCachedBits;

		uint32_t hostVisibleDeviceLocalBits = hostVisibleBits & deviceLocalBits;

		m_logger->log("Memory type bits - HostVisible: 0x%X, DeviceLocal: 0x%X, HostCached: 0x%X",
			ILogger::ELL_PERFORMANCE, hostVisibleBits, deviceLocalBits, hostCachedBits);
		m_logger->log("System RAM (non-cached): 0x%X, VRAM: 0x%X",
			ILogger::ELL_PERFORMANCE, hostVisibleOnlyBits, hostVisibleDeviceLocalBits);

		if (!hostVisibleOnlyBits)
		{
			m_logger->log("HOST_VISIBLE non-cached memory types not found!", ILogger::ELL_ERROR);
			return false;
		}

		if (!deviceLocalBits)
		{
			m_logger->log("DEVICE_LOCAL memory types not found!", ILogger::ELL_ERROR);
			return false;
		}

		IQueue* queue = getQueue(IQueue::FAMILY_FLAGS::GRAPHICS_BIT);
		smart_refctd_ptr<IGPUImage> destinationImage;
		{
			IGPUImage::SCreationParams imgParams{};
			imgParams.type = IImage::E_TYPE::ET_2D;
			uint32_t tilePerRow = (uint32_t)std::sqrt(TILES_PER_FRAME);
			imgParams.extent.width = TILE_SIZE * tilePerRow;
			imgParams.extent.height = TILE_SIZE * tilePerRow;
			imgParams.extent.depth = 1u;
			imgParams.format = asset::E_FORMAT::EF_R8G8B8A8_UNORM;
			imgParams.mipLevels = 1u;
			imgParams.flags = IImage::ECF_NONE;
			imgParams.arrayLayers = 1u;
			imgParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			imgParams.tiling = video::IGPUImage::TILING::OPTIMAL;
			imgParams.usage = asset::IImage::EUF_TRANSFER_DST_BIT;
			imgParams.preinitialized = false;

			destinationImage = m_device->createImage(std::move(imgParams));
			if (!destinationImage)
				return logFail("Failed to create destination image!\n");

			destinationImage->setObjectDebugName("Destination Image");

			auto reqs = destinationImage->getMemoryReqs();
			reqs.memoryTypeBits &= deviceLocalBits;

			auto allocation = m_device->allocate(reqs, destinationImage.get(), IDeviceMemoryAllocation::EMAF_NONE);
			if (!allocation.isValid())
				return logFail("Failed to allocate DEVICE_LOCAL memory for destination image!\n");
		}

		m_logger->log("\nStrategy 1: System RAM", ILogger::ELL_PERFORMANCE);

		double throughputSystemRAM = 0.0;
		{
			smart_refctd_ptr<IGPUBuffer> stagingBuffer;
			IDeviceMemoryAllocator::SAllocation stagingAlloc;
			void* mappedPtr = nullptr;

			if (!createStagingBuffer(STAGING_BUFFER_SIZE, hostVisibleOnlyBits,
				"Staging Buffer - System RAM", stagingBuffer, stagingAlloc, mappedPtr))
			{
				return false;
			}

			throughputSystemRAM = runBenchmark(
				"System RAM",
				stagingBuffer.get(),
				stagingAlloc,
				mappedPtr,
				destinationImage.get(),
				TILE_SIZE,
				TILE_SIZE_BYTES,
				TILES_PER_FRAME,
				FRAMES_IN_FLIGHT,
				TOTAL_FRAMES,
				queue
			);

			stagingAlloc.memory->unmap();
		}

		m_logger->log("System RAM throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, throughputSystemRAM);

		m_device->waitIdle();

		if (hostVisibleDeviceLocalBits)
		{
			m_logger->log("\nStrategy 2: VRAM", ILogger::ELL_PERFORMANCE);

			double throughputVRAM = 0.0;
			{
				smart_refctd_ptr<IGPUBuffer> stagingBuffer;
				IDeviceMemoryAllocator::SAllocation stagingAlloc;
				void* mappedPtr = nullptr;

				if (!createStagingBuffer(STAGING_BUFFER_SIZE, hostVisibleDeviceLocalBits,
					"Staging Buffer - VRAM", stagingBuffer, stagingAlloc, mappedPtr))
				{
					return false;
				}

				throughputVRAM = runBenchmark(
					"VRAM",
					stagingBuffer.get(),
					stagingAlloc,
					mappedPtr,
					destinationImage.get(),
					TILE_SIZE,
					TILE_SIZE_BYTES,
					TILES_PER_FRAME,
					FRAMES_IN_FLIGHT,
					TOTAL_FRAMES,
					queue
				);

				stagingAlloc.memory->unmap();
			}

			m_logger->log("VRAM throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, throughputVRAM);

			double speedup = throughputVRAM / throughputSystemRAM;
			m_logger->log("\nVRAM is %.2fx faster than System RAM", ILogger::ELL_PERFORMANCE, speedup);
		}

		m_logger->log("\nWaiting 5 seconds before exit...", ILogger::ELL_PERFORMANCE);
		std::this_thread::sleep_for(std::chrono::seconds(5));

		return true;
	}

	bool keepRunning() override { return false; }
	void workLoopBody() override {}
	bool onAppTerminated() override { return true; }

protected:
	core::vector<queue_req_t> getQueueRequirements() const override
	{
		using flags_t = IQueue::FAMILY_FLAGS;
		return { {
			.requiredFlags = flags_t::GRAPHICS_BIT,
			.disallowedFlags = flags_t::NONE,
			.queueCount = 1,
			.maxImageTransferGranularity = {1, 1, 1}
		} };
	}

private:
	void generateTileCopyRegions(
		IImage::SBufferCopy* outRegions,
		uint32_t tilesPerFrame,
		uint32_t tileSize,
		uint32_t tileSizeBytes,
		uint32_t imageWidth,
		uint32_t bufferBaseOffset)
	{
		uint32_t tilesPerRow = imageWidth / tileSize;
		for (size_t i = 0; i < tilesPerFrame; i++)
		{
			uint32_t tileX = (i % tilesPerRow) * tileSize;
			uint32_t tileY = (i / tilesPerRow) * tileSize;

			outRegions[i].bufferOffset = bufferBaseOffset + (i * tileSizeBytes);
			outRegions[i].bufferRowLength = tileSize;
			outRegions[i].bufferImageHeight = tileSize;
			outRegions[i].imageOffset = { tileX, tileY, 0 };
			outRegions[i].imageExtent = { tileSize, tileSize, 1 };
			outRegions[i].imageSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			outRegions[i].imageSubresource.mipLevel = 0;
			outRegions[i].imageSubresource.baseArrayLayer = 0;
			outRegions[i].imageSubresource.layerCount = 1;
		}
	}

	double runBenchmark(
		const char* strategyName,
		IGPUBuffer* stagingBuffer,
		IDeviceMemoryAllocator::SAllocation& stagingAlloc,
		void* mappedPtr,
		IGPUImage* destinationImage,
		uint32_t tileSize,
		uint32_t tileSizeBytes,
		uint32_t tilesPerFrame,
		uint32_t framesInFlight,
		uint32_t totalFrames,
		IQueue* queue)
	{
		smart_refctd_ptr<ISemaphore> timelineSemaphore = m_device->createSemaphore(0);

		smart_refctd_ptr<IQueryPool> queryPool;
		{
			IQueryPool::SCreationParams queryPoolParams = {};
			queryPoolParams.queryType = IQueryPool::TYPE::TIMESTAMP;
			queryPoolParams.queryCount = framesInFlight * 2;  
			queryPoolParams.pipelineStatisticsFlags = IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
			queryPool = m_device->createQueryPool(queryPoolParams);
		}
		
		std::vector<smart_refctd_ptr<IGPUCommandPool>> commandPools(framesInFlight);
		for (uint32_t i = 0; i < framesInFlight; i++)
		{
			commandPools[i] = m_device->createCommandPool(
				queue->getFamilyIndex(),
				IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT
			);
		}
		std::vector<smart_refctd_ptr<IGPUCommandBuffer>> commandBuffers(framesInFlight);
		for (uint32_t i = 0; i < framesInFlight; i++)
		{
			commandPools[i]->createCommandBuffers(
				IGPUCommandPool::BUFFER_LEVEL::PRIMARY,
				1,
				&commandBuffers[i]
			);
		}

		uint64_t timelineValue = 0;

		commandBuffers[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> initBarrier = {};
			initBarrier.oldLayout = IImage::LAYOUT::UNDEFINED;
			initBarrier.newLayout = IImage::LAYOUT::GENERAL;
			initBarrier.image = destinationImage;
			initBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			initBarrier.subresourceRange.baseMipLevel = 0;
			initBarrier.subresourceRange.levelCount = 1;
			initBarrier.subresourceRange.baseArrayLayer = 0;
			initBarrier.subresourceRange.layerCount = 1;
			initBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
			initBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			initBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
			initBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			commandBuffers[0]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.imgBarriers = {&initBarrier, 1}});
		}
		commandBuffers[0]->end();

		IQueue::SSubmitInfo submitInfo = {};
		IQueue::SSubmitInfo::SCommandBufferInfo cmdBufInfo = { .cmdbuf = commandBuffers[0].get() };
		submitInfo.commandBuffers = { &cmdBufInfo, 1 };

		IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
			.semaphore = timelineSemaphore.get(),
			.value = ++timelineValue,
			.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
		};
		submitInfo.signalSemaphores = { &signalInfo, 1 };

		queue->submit({ &submitInfo, 1 });

		ISemaphore::SWaitInfo waitInfo = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({ &waitInfo, 1 });

		uint32_t imageWidth = destinationImage->getCreationParameters().extent.width;
		uint32_t partitionSize = tilesPerFrame * tileSizeBytes;

		std::vector<uint8_t> cpuSourceData(partitionSize);
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::mt19937 g(seed);
			uint32_t* data = reinterpret_cast<uint32_t*>(cpuSourceData.data());
			for (uint32_t i = 0; i < partitionSize / sizeof(uint32_t); i++)
				data[i] = g();
		}
		std::vector<std::vector<IImage::SBufferCopy>> regionsPerFrame(framesInFlight);
		for (uint32_t i = 0; i < framesInFlight; i++)
		{
			regionsPerFrame[i].resize(tilesPerFrame);
			uint32_t bufferOffset = i * partitionSize;
			generateTileCopyRegions(regionsPerFrame[i].data(), tilesPerFrame, tileSize, tileSizeBytes, imageWidth, bufferOffset);
		}

		double totalWaitTime = 0.0;
		double totalMemcpyTime = 0.0;
		double totalRecordTime = 0.0;
		double totalSubmitTime = 0.0;

		auto startTime = std::chrono::high_resolution_clock::now();

		for (uint32_t frame = 0; frame < totalFrames; frame++)
		{
			uint32_t cmdBufIndex = frame % framesInFlight;

			auto t1 = std::chrono::high_resolution_clock::now();
			if (frame >= framesInFlight)
			{
				ISemaphore::SWaitInfo frameWaitInfo = {
					.semaphore = timelineSemaphore.get(),
					.value = timelineValue - framesInFlight + 1
				};
				m_device->blockForSemaphores({&frameWaitInfo, 1});
			}
			auto t2 = std::chrono::high_resolution_clock::now();

			commandPools[cmdBufIndex]->reset();

			uint32_t bufferOffset = cmdBufIndex * partitionSize;
			void* targetPtr = static_cast<uint8_t*>(mappedPtr) + bufferOffset;
			memcpy(targetPtr, cpuSourceData.data(), partitionSize);

			if (!stagingAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			{
				ILogicalDevice::MappedMemoryRange range(stagingAlloc.memory.get(), bufferOffset, partitionSize);
				m_device->flushMappedMemoryRanges(1, &range);
			}

			auto t3 = std::chrono::high_resolution_clock::now();

			commandBuffers[cmdBufIndex]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			uint32_t queryStartIndex = cmdBufIndex * 2;
			commandBuffers[cmdBufIndex]->resetQueryPool(queryPool.get(), queryStartIndex, 2);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> barrier = {};
			barrier.oldLayout = IImage::LAYOUT::GENERAL;
			barrier.newLayout = IImage::LAYOUT::GENERAL;
			barrier.image = destinationImage;
			barrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.imgBarriers = {&barrier, 1}});

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COPY_BIT, queryPool.get(), queryStartIndex + 0);

			commandBuffers[cmdBufIndex]->copyBufferToImage(
				stagingBuffer,
				destinationImage,
				IImage::LAYOUT::GENERAL,
				tilesPerFrame,
				regionsPerFrame[cmdBufIndex].data()
			);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> afterBarrier = {};
			afterBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			afterBarrier.newLayout = IImage::LAYOUT::GENERAL;
			afterBarrier.image = destinationImage;
			afterBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			afterBarrier.subresourceRange.baseMipLevel = 0;
			afterBarrier.subresourceRange.levelCount = 1;
			afterBarrier.subresourceRange.baseArrayLayer = 0;
			afterBarrier.subresourceRange.layerCount = 1;
			afterBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			afterBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			afterBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			afterBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.imgBarriers = {&afterBarrier, 1}});

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COPY_BIT, queryPool.get(), queryStartIndex + 1);

			commandBuffers[cmdBufIndex]->end();
			auto t4 = std::chrono::high_resolution_clock::now();

			IQueue::SSubmitInfo frameSubmitInfo = {};
			IQueue::SSubmitInfo::SCommandBufferInfo frameCmdBufInfo = {.cmdbuf = commandBuffers[cmdBufIndex].get()};
			frameSubmitInfo.commandBuffers = {&frameCmdBufInfo, 1};

			IQueue::SSubmitInfo::SSemaphoreInfo frameSignalInfo = {
				.semaphore = timelineSemaphore.get(),
				.value = ++timelineValue,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			frameSubmitInfo.signalSemaphores = {&frameSignalInfo, 1};

			queue->submit({&frameSubmitInfo, 1});
			auto t5 = std::chrono::high_resolution_clock::now();

			totalWaitTime += std::chrono::duration<double>(t2 - t1).count();
			totalMemcpyTime += std::chrono::duration<double>(t3 - t2).count();
			totalRecordTime += std::chrono::duration<double>(t4 - t3).count();
			totalSubmitTime += std::chrono::duration<double>(t5 - t4).count();
		}

		// Wait for all remaining frames to complete
		ISemaphore::SWaitInfo finalWait = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({&finalWait, 1});

		auto endTime = std::chrono::high_resolution_clock::now();

		std::vector<uint64_t> timestamps(framesInFlight * 2);
		const core::bitflag flags = core::bitflag(IQueryPool::RESULTS_FLAGS::_64_BIT) | core::bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT);
		m_device->getQueryPoolResults(queryPool.get(), 0, framesInFlight * 2, timestamps.data(), sizeof(uint64_t), flags);
		uint64_t totalGpuTicks = 0;
		for (uint32_t i = 0; i < framesInFlight; i++) {
			uint64_t startTick = timestamps[i * 2 + 0];
			uint64_t endTick = timestamps[i * 2 + 1];
			totalGpuTicks += (endTick - startTick);
		}
		float timestampPeriod = m_physicalDevice->getLimits().timestampPeriodInNanoSeconds;
		double sampledGpuTimeSeconds = (totalGpuTicks * timestampPeriod) / 1e9;

		double avgGpuTimePerFrame = sampledGpuTimeSeconds / framesInFlight;
		double totalGpuTimeSeconds = avgGpuTimePerFrame * totalFrames;


		double elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();
		uint64_t totalBytes = (uint64_t)totalFrames * tilesPerFrame * tileSizeBytes;

		double throughputGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsedSeconds;

		m_logger->log("    GPU time: %.3f s", ILogger::ELL_PERFORMANCE, totalGpuTimeSeconds);
		m_logger->log("    GPU throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, throughputGBps);

		m_logger->log("  Timing breakdown for %s:", ILogger::ELL_PERFORMANCE, strategyName);
		m_logger->log("    Wait time:   %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalWaitTime, 100.0 * totalWaitTime / elapsedSeconds);
		m_logger->log("    Memcpy time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalMemcpyTime, 100.0 * totalMemcpyTime / elapsedSeconds);
		m_logger->log("    Record time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalRecordTime, 100.0 * totalRecordTime / elapsedSeconds);
		m_logger->log("    Submit time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalSubmitTime, 100.0 * totalSubmitTime / elapsedSeconds);
		m_logger->log("    Memcpy speed: %.2f GB/s", ILogger::ELL_PERFORMANCE, (totalBytes / (1024.0 * 1024.0 * 1024.0)) / totalMemcpyTime);

		return throughputGBps;
	}

	bool createStagingBuffer(
		uint32_t bufferSize,
		uint32_t memoryTypeBits,
		const char* debugName,
		smart_refctd_ptr<IGPUBuffer>& outBuffer,
		IDeviceMemoryAllocator::SAllocation& outAllocation,
		void*& outMappedPtr)
	{
		IGPUBuffer::SCreationParams params;
		params.size = bufferSize;
		params.usage = IGPUBuffer::EUF_TRANSFER_SRC_BIT;
		outBuffer = m_device->createBuffer(std::move(params));
		if (!outBuffer)
			return logFail("Failed to create GPU buffer of size %d!\n", bufferSize);

		outBuffer->setObjectDebugName(debugName);

		auto reqs = outBuffer->getMemoryReqs();
		reqs.memoryTypeBits &= memoryTypeBits;

		outAllocation = m_device->allocate(reqs, outBuffer.get(), IDeviceMemoryAllocation::EMAF_NONE);
		if (!outAllocation.isValid())
			return logFail("Failed to allocate Device Memory!\n");

		outMappedPtr = outAllocation.memory->map({0ull, outAllocation.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_WRITE);
		if (!outMappedPtr)
			return logFail("Failed to map Device Memory!\n");

		return true;
	}
};

NBL_MAIN_FUNC(ImageUploadBenchmarkApp)
