#include "nbl/examples/examples.hpp"
#include <chrono>

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
		constexpr uint32_t TILES_PER_FRAME = STAGING_BUFFER_SIZE / TILE_SIZE_BYTES;
		constexpr uint32_t FRAMES_IN_FLIGHT = 4;
		constexpr uint32_t TOTAL_FRAMES = 1000;

		m_logger->log("GPU Memory Transfer Benchmark", ILogger::ELL_INFO);
		m_logger->log("Tile size: %ux%u (%u KB)", ILogger::ELL_INFO, TILE_SIZE, TILE_SIZE, TILE_SIZE_BYTES / 1024);
		m_logger->log("Staging buffer: %u MB", ILogger::ELL_INFO, STAGING_BUFFER_SIZE / (1024 * 1024));
		m_logger->log("Tiles per frame: %u", ILogger::ELL_INFO, TILES_PER_FRAME);
		m_logger->log("Frames in flight: %u", ILogger::ELL_INFO, FRAMES_IN_FLIGHT);

		uint32_t hostVisibleBits = m_physicalDevice->getHostVisibleMemoryTypeBits();
		uint32_t deviceLocalBits = m_physicalDevice->getDeviceLocalMemoryTypeBits();
		uint32_t hostVisibleOnlyBits = hostVisibleBits & ~deviceLocalBits;
		uint32_t hostVisibleDeviceLocalBits = hostVisibleBits & deviceLocalBits;

		if (!hostVisibleOnlyBits)
		{
			m_logger->log("HOST_VISIBLE memory types not found!", ILogger::ELL_ERROR);
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
			imgParams.extent.width = TILE_SIZE * 32;
			imgParams.extent.height = TILE_SIZE * 32;
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

		m_logger->log("\nTesting Strategy 1: System RAM", ILogger::ELL_INFO);

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
			m_logger->log("\nTesting Strategy 2: VRAM (ReBAR)", ILogger::ELL_INFO);

			double throughputVRAM = 0.0;
			{
				smart_refctd_ptr<IGPUBuffer> stagingBuffer;
				IDeviceMemoryAllocator::SAllocation stagingAlloc;
				void* mappedPtr = nullptr;

				if (!createStagingBuffer(STAGING_BUFFER_SIZE, hostVisibleDeviceLocalBits,
					"Staging Buffer - VRAM (ReBAR)", stagingBuffer, stagingAlloc, mappedPtr))
				{
					return false;
				}

				throughputVRAM = runBenchmark(
					"VRAM (ReBAR)",
					stagingBuffer.get(),
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
	void transitionImageLayout(
		IGPUCommandBuffer* cmdBuf,
		IGPUImage* image,
		IImage::LAYOUT oldLayout,
		IImage::LAYOUT newLayout)
	{
		IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> barrier = {};
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
		barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
		barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
		barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
		cmdBuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });
	}

	void generateTileCopyRegions(
		IImage::SBufferCopy* outRegions,
		uint32_t tilesPerFrame,
		uint32_t tileSize,
		uint32_t tileSizeBytes,
		uint32_t imageWidth)
	{
		uint32_t tilesPerRow = imageWidth / tileSize;
		for (size_t i = 0; i < tilesPerFrame; i++)
		{
			uint32_t tileX = (i % tilesPerRow) * tileSize;
			uint32_t tileY = (i / tilesPerRow) * tileSize;

			outRegions[i].bufferOffset = i * tileSizeBytes;
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

	void generateRandomTileData(void* mappedPtr, uint32_t sizeBytes)
	{
		uint32_t* data = (uint32_t*)mappedPtr;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);
		const uint32_t valueCount = sizeBytes / sizeof(uint32_t);

		auto bufferData = new uint32_t[valueCount];

		for (uint32_t i = 0; i < valueCount; i++)
		{
			bufferData[i] = g();
		}
		memcpy(mappedPtr, bufferData, sizeBytes);
		delete[] bufferData;
	}

	double runBenchmark(
		const char* strategyName,
		IGPUBuffer* stagingBuffer,
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

		auto commandPools = new smart_refctd_ptr<IGPUCommandPool>[framesInFlight];
		for (uint32_t i = 0; i < framesInFlight; i++)
		{
			commandPools[i] = m_device->createCommandPool(
				queue->getFamilyIndex(),
				IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT
			);
		}

		auto commandBuffers = new smart_refctd_ptr<IGPUCommandBuffer>[framesInFlight];
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
		transitionImageLayout(
			commandBuffers[0].get(),
			destinationImage,
			IImage::LAYOUT::UNDEFINED,
			IImage::LAYOUT::TRANSFER_DST_OPTIMAL
		);
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

		auto regions = new IImage::SBufferCopy[tilesPerFrame];

		generateRandomTileData(mappedPtr, tilesPerFrame * tileSizeBytes);

		uint32_t imageWidth = destinationImage->getCreationParameters().extent.width;
		generateTileCopyRegions(regions, tilesPerFrame, tileSize, tileSizeBytes, imageWidth);

		auto startTime = std::chrono::high_resolution_clock::now();

		for (uint32_t frame = 0; frame < totalFrames; frame++)
		{
			uint32_t cmdBufIndex = frame % framesInFlight;

			commandBuffers[cmdBufIndex]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			commandBuffers[cmdBufIndex]->copyBufferToImage(
				stagingBuffer,
				destinationImage,
				IImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				tilesPerFrame,
				regions
			);

			commandBuffers[cmdBufIndex]->end();

			// Create submit info for THIS frame
			IQueue::SSubmitInfo frameSubmitInfo = {};
			IQueue::SSubmitInfo::SCommandBufferInfo frameCmdBufInfo = {.cmdbuf = commandBuffers[cmdBufIndex].get()};
			frameSubmitInfo.commandBuffers = {&frameCmdBufInfo, 1};

			IQueue::SSubmitInfo::SSemaphoreInfo frameSignalInfo = {
				.semaphore = timelineSemaphore.get(),
				.value = ++timelineValue,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			frameSubmitInfo.signalSemaphores = {&frameSignalInfo, 1};

			// Submit to GPU
			queue->submit({&frameSubmitInfo, 1});

			// Wait for old frames 
			if (frame >= framesInFlight)
			{
				ISemaphore::SWaitInfo frameWaitInfo = {
					.semaphore = timelineSemaphore.get(),
					.value = timelineValue - framesInFlight
				};
				m_device->blockForSemaphores({&frameWaitInfo, 1});
			}
		}

		// Wait for all remaining frames to complete
		ISemaphore::SWaitInfo finalWait = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({&finalWait, 1});

		auto endTime = std::chrono::high_resolution_clock::now();

		delete[] regions;
		delete[] commandPools;
		delete[] commandBuffers;

		// Calculate throughput
		double elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();
		uint64_t totalBytes = (uint64_t)totalFrames * tilesPerFrame * tileSizeBytes;
		double throughputGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsedSeconds;

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

		outMappedPtr = outAllocation.memory->map({0ull, outAllocation.memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ);
		if (!outMappedPtr)
			return logFail("Failed to map Device Memory!\n");

		return true;
	}
};

NBL_MAIN_FUNC(ImageUploadBenchmarkApp)
