#include "nbl/examples/examples.hpp"
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
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

		uint32_t hostVisibleOnlyBits = hostVisibleBits & ~deviceLocalBits;

		uint32_t hostVisibleDeviceLocalBits = hostVisibleBits & deviceLocalBits;

		m_logger->log("Memory type bits HostVisible: 0x%X, DeviceLocal: 0x%X, HostCached: 0x%X",
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

		m_queue = getQueue(IQueue::FAMILY_FLAGS::GRAPHICS_BIT);
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
			imgParams.usage = asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_STORAGE_BIT;
			imgParams.preinitialized = false;

			m_destinationImage = m_device->createImage(std::move(imgParams));
			if (!m_destinationImage)
				return logFail("Failed to create destination image!\n");

			m_destinationImage->setObjectDebugName("Destination Image");

			auto reqs = m_destinationImage->getMemoryReqs();
			reqs.memoryTypeBits &= deviceLocalBits;

			auto allocation = m_device->allocate(reqs, m_destinationImage.get(), IDeviceMemoryAllocation::EMAF_NONE);
			if (!allocation.isValid())
				return logFail("Failed to allocate DEVICE_LOCAL memory for destination image!\n");
		}

		//compute shader
		auto loadPrecompiledShader = [&]<core::StringLiteral ShaderKey>()->smart_refctd_ptr<IShader>
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = "app_resources";

			auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_physicalDevice->getLimits(), m_physicalDevice->getFeatures());
			m_logger->log("Loading shader with key: %s", ILogger::ELL_INFO, key.data());

			auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
			{
				m_logger->log("Asset bundle is empty for key: %s", ILogger::ELL_ERROR, key.data());
				return smart_refctd_ptr<IShader>(nullptr);
			}

			m_logger->log("Asset count: %u, asset type: %u", ILogger::ELL_INFO, assets.size(), (uint32_t)assets[0]->getAssetType());

			auto shader = IAsset::castDown<IShader>(assets[0]);
			return shader;
		};


		//Setup compute shader resources
		m_logger->log("\n=== Setting up Compute Shaders (Linear + Snake + Morton) ===", ILogger::ELL_PERFORMANCE);
		{
			auto shaderLib = loadPrecompiledShader.operator()<"snakeStore">();
			if (!shaderLib)
				return logFail("Failed to load shader library!\n");

			IGPUDescriptorSetLayout::SBinding dsBinding = {
				.binding = 0,
				.type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,
				.count = 1
			};
			auto dsLayout = m_device->createDescriptorSetLayout({&dsBinding, 1});
			if (!dsLayout)
				return logFail("Failed to create descriptor set layout!\n");

			asset::SPushConstantRange pcRange = {
				.stageFlags = hlsl::ShaderStage::ESS_COMPUTE,
				.offset = 0,
				.size = sizeof(SPushConstantData)
			};

			m_pipelineLayout = m_device->createPipelineLayout({&pcRange, 1}, smart_refctd_ptr(dsLayout));
			if (!m_pipelineLayout)
				return logFail("Failed to create pipeline layout!\n");

			IGPUComputePipeline::SCreationParams storeParams = {};
			storeParams.layout = m_pipelineLayout.get();
			storeParams.shader.shader = shaderLib.get();
			storeParams.shader.entryPoint = "linearStore";

			if (!m_device->createComputePipelines(nullptr, {&storeParams, 1}, &m_storePipeline))
				return logFail("Failed to create linearStore pipeline!\n");

			IGPUComputePipeline::SCreationParams loadParams = {};
			loadParams.layout = m_pipelineLayout.get();
			loadParams.shader.shader = shaderLib.get();
			loadParams.shader.entryPoint = "linearLoad";

			if (!m_device->createComputePipelines(nullptr, {&loadParams, 1}, &m_loadPipeline))
				return logFail("Failed to create linearLoad pipeline!\n");

			IGPUComputePipeline::SCreationParams snakeStoreParams = {};
			snakeStoreParams.layout = m_pipelineLayout.get();
			snakeStoreParams.shader.shader = shaderLib.get();
			snakeStoreParams.shader.entryPoint = "SnakeOrderStore";

			if (!m_device->createComputePipelines(nullptr, {&snakeStoreParams, 1}, &m_snakeStorePipeline))
				return logFail("Failed to create SnakeOrderStore pipeline!\n");

			IGPUComputePipeline::SCreationParams snakeLoadParams = {};
			snakeLoadParams.layout = m_pipelineLayout.get();
			snakeLoadParams.shader.shader = shaderLib.get();
			snakeLoadParams.shader.entryPoint = "SnakeOrderLoad";

			if (!m_device->createComputePipelines(nullptr, {&snakeLoadParams, 1}, &m_snakeLoadPipeline))
				return logFail("Failed to create SnakeOrderLoad pipeline!\n");

			IGPUComputePipeline::SCreationParams mortonStoreParams = {};
			mortonStoreParams.layout = m_pipelineLayout.get();
			mortonStoreParams.shader.shader = shaderLib.get();
			mortonStoreParams.shader.entryPoint = "MortonOrderStore";

			if (!m_device->createComputePipelines(nullptr, {&mortonStoreParams, 1}, &m_mortonStorePipeline))
				return logFail("Failed to create MortonOrderStore pipeline!\n");

			IGPUComputePipeline::SCreationParams mortonLoadParams = {};
			mortonLoadParams.layout = m_pipelineLayout.get();
			mortonLoadParams.shader.shader = shaderLib.get();
			mortonLoadParams.shader.entryPoint = "MortonOrderLoad";

			if (!m_device->createComputePipelines(nullptr, {&mortonLoadParams, 1}, &m_mortonLoadPipeline))
				return logFail("Failed to create MortonOrderLoad pipeline!\n");

			auto createBatchedPipeline = [&](const char* entryPoint, smart_refctd_ptr<IGPUComputePipeline>& outPipeline) -> bool
			{
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = m_pipelineLayout.get();
				params.shader.shader = shaderLib.get();
				params.shader.entryPoint = entryPoint;
				if (!m_device->createComputePipelines(nullptr, {&params, 1}, &outPipeline))
					return logFail("Failed to create %s pipeline!\n", entryPoint);
				return true;
			};

			if (!createBatchedPipeline("BatchedLinearStore", m_batchedLinearPipeline)) return false;
			if (!createBatchedPipeline("BatchedSnakeStore", m_batchedSnakePipeline)) return false;
			if (!createBatchedPipeline("BatchedMortonStore", m_batchedMortonPipeline)) return false;

			auto imageView = m_device->createImageView({
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = IGPUImage::EUF_STORAGE_BIT,
				.image = smart_refctd_ptr(m_destinationImage),
				.viewType = IGPUImageView::E_TYPE::ET_2D,
				.format = asset::E_FORMAT::EF_R8G8B8A8_UNORM
			});
			if (!imageView)
				return logFail("Failed to create image view!\n");

			uint32_t setCount = 1;
			auto dsPool = m_device->createDescriptorPoolForDSLayouts(
				IDescriptorPool::ECF_NONE, {&dsLayout.get(), 1}, &setCount);
			m_ds = dsPool->createDescriptorSet(smart_refctd_ptr(dsLayout));

			IGPUDescriptorSet::SDescriptorInfo imgInfo = {};
			imgInfo.desc = imageView;
			imgInfo.info.image.imageLayout = IGPUImage::LAYOUT::GENERAL;

			IGPUDescriptorSet::SWriteDescriptorSet dsWrite = {
				.dstSet = m_ds.get(),
				.binding = 0,
				.arrayElement = 0,
				.count = 1,
				.info = &imgInfo
			};
			m_device->updateDescriptorSets({&dsWrite, 1}, {});

			if (!createStagingBuffer(TILE_SIZE_BYTES, hostVisibleOnlyBits,
				"Verify Staging Buffer", m_stagingBuffer, m_stagingAlloc, m_stagingMappedPtr))
				return false;

			if (!createStagingBuffer(TILE_SIZE_BYTES, hostVisibleOnlyBits,
				"Verify Readback Buffer", m_readbackBuffer, m_readbackAlloc, m_readbackMappedPtr))
				return false;

			if (!createStagingBuffer(TILE_SIZE_BYTES, hostVisibleOnlyBits,
				"Snake Readback Buffer", m_snakeReadbackBuffer, m_snakeReadbackAlloc, m_snakeReadbackMappedPtr))
				return false;

			if (!createStagingBuffer(TILE_SIZE_BYTES, hostVisibleOnlyBits,
				"Morton Readback Buffer", m_mortonReadbackBuffer, m_mortonReadbackAlloc, m_mortonReadbackMappedPtr))
				return false;

			{
				uint32_t* pixels = static_cast<uint32_t*>(m_stagingMappedPtr);
				uint32_t totalPixels = TILE_SIZE * TILE_SIZE;
				for (uint32_t i = 0; i < totalPixels; i++)
				{
					uint8_t val = static_cast<uint8_t>(i & 0xFF);
					pixels[i] = val | (val << 8u) | (val << 16u) | (val << 24u);
				}

				if (!m_stagingAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				{
					ILogicalDevice::MappedMemoryRange range(m_stagingAlloc.memory.get(), 0, TILE_SIZE_BYTES);
					m_device->flushMappedMemoryRanges(1, &range);
				}
			}

			m_cmdPool = m_device->createCommandPool(
				m_queue->getFamilyIndex(),
				IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT
			);
			m_cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &m_cmdbuf);
			m_sem = m_device->createSemaphore(0);

			m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			{
				IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> initBarrier = {};
				initBarrier.oldLayout = IImage::LAYOUT::UNDEFINED;
				initBarrier.newLayout = IImage::LAYOUT::GENERAL;
				initBarrier.image = m_destinationImage.get();
				initBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				initBarrier.subresourceRange.baseMipLevel = 0;
				initBarrier.subresourceRange.levelCount = 1;
				initBarrier.subresourceRange.baseArrayLayer = 0;
				initBarrier.subresourceRange.layerCount = 1;
				initBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
				initBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
				initBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
				initBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
				m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&initBarrier, 1} });
			}
			m_cmdbuf->end();

			IQueue::SSubmitInfo submitInfo = {};
			IQueue::SSubmitInfo::SCommandBufferInfo cmdBufInfo = { .cmdbuf = m_cmdbuf.get() };
			submitInfo.commandBuffers = { &cmdBufInfo, 1 };

			IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
				.semaphore = m_sem.get(),
				.value = 1,
				.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};
			submitInfo.signalSemaphores = { &signalInfo, 1 };

			m_queue->submit({ &submitInfo, 1 });

			ISemaphore::SWaitInfo waitInfo = { .semaphore = m_sem.get(), .value = 1 };
			m_device->blockForSemaphores({ &waitInfo, 1 });
		}

		m_logger->log("Setup complete. Running verification loop (%u frames)", ILogger::ELL_PERFORMANCE, VERIFICATION_LOOP_COUNT);

		return true;
	}

	bool keepRunning() override { return m_frameIndex < VERIFICATION_LOOP_COUNT; }

	void workLoopBody() override
	{
		m_cmdPool->reset();

		//Clear readback buffers to zero
		memset(m_readbackMappedPtr, 0, TILE_SIZE_BYTES);
		if (!m_readbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_readbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->flushMappedMemoryRanges(1, &range);
		}
		memset(m_snakeReadbackMappedPtr, 0, TILE_SIZE_BYTES);
		if (!m_snakeReadbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_snakeReadbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->flushMappedMemoryRanges(1, &range);
		}
		memset(m_mortonReadbackMappedPtr, 0, TILE_SIZE_BYTES);
		if (!m_mortonReadbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_mortonReadbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->flushMappedMemoryRanges(1, &range);
		}

		m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> barrier = {};
			barrier.oldLayout = IImage::LAYOUT::GENERAL;
			barrier.newLayout = IImage::LAYOUT::GENERAL;
			barrier.image = m_destinationImage.get();
			barrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			barrier.subresourceRange.baseMipLevel = 0;
			barrier.subresourceRange.levelCount = 1;
			barrier.subresourceRange.baseArrayLayer = 0;
			barrier.subresourceRange.layerCount = 1;
			barrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			barrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			barrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			barrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_storePipeline.get());
		const IGPUDescriptorSet* sets[] = { m_ds.get() };
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);

		SPushConstantData storePc = {
			.deviceBufferAddress = m_stagingBuffer->getDeviceAddress(),
			.dstOffsetX = 0,
			.dstOffsetY = 0,
			.srcWidth = TILE_SIZE,
			.srcHeight = TILE_SIZE
		};
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &storePc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> midBarrier = {};
			midBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			midBarrier.newLayout = IImage::LAYOUT::GENERAL;
			midBarrier.image = m_destinationImage.get();
			midBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			midBarrier.subresourceRange.baseMipLevel = 0;
			midBarrier.subresourceRange.levelCount = 1;
			midBarrier.subresourceRange.baseArrayLayer = 0;
			midBarrier.subresourceRange.layerCount = 1;
			midBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			midBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			midBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			midBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&midBarrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_loadPipeline.get());
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);

		SPushConstantData loadPc = {
			.deviceBufferAddress = m_readbackBuffer->getDeviceAddress(),
			.dstOffsetX = 0,
			.dstOffsetY = 0,
			.srcWidth = TILE_SIZE,
			.srcHeight = TILE_SIZE
		};
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &loadPc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			asset::SMemoryBarrier memBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
				.dstAccessMask = ACCESS_FLAGS::HOST_READ_BIT
			};
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
		}

		//SNAKE VERIFICATION
		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> snakePreBarrier = {};
			snakePreBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			snakePreBarrier.newLayout = IImage::LAYOUT::GENERAL;
			snakePreBarrier.image = m_destinationImage.get();
			snakePreBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			snakePreBarrier.subresourceRange.baseMipLevel = 0;
			snakePreBarrier.subresourceRange.levelCount = 1;
			snakePreBarrier.subresourceRange.baseArrayLayer = 0;
			snakePreBarrier.subresourceRange.layerCount = 1;
			snakePreBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			snakePreBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			snakePreBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			snakePreBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&snakePreBarrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_snakeStorePipeline.get());
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &storePc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> snakeMidBarrier = {};
			snakeMidBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			snakeMidBarrier.newLayout = IImage::LAYOUT::GENERAL;
			snakeMidBarrier.image = m_destinationImage.get();
			snakeMidBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			snakeMidBarrier.subresourceRange.baseMipLevel = 0;
			snakeMidBarrier.subresourceRange.levelCount = 1;
			snakeMidBarrier.subresourceRange.baseArrayLayer = 0;
			snakeMidBarrier.subresourceRange.layerCount = 1;
			snakeMidBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			snakeMidBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			snakeMidBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			snakeMidBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&snakeMidBarrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_snakeLoadPipeline.get());
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);

		SPushConstantData snakeLoadPc = {
			.deviceBufferAddress = m_snakeReadbackBuffer->getDeviceAddress(),
			.dstOffsetX = 0,
			.dstOffsetY = 0,
			.srcWidth = TILE_SIZE,
			.srcHeight = TILE_SIZE
		};
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &snakeLoadPc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			asset::SMemoryBarrier memBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
				.dstAccessMask = ACCESS_FLAGS::HOST_READ_BIT
			};
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
		}

		//MORTON VERIFICATION

		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> mortonPreBarrier = {};
			mortonPreBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			mortonPreBarrier.newLayout = IImage::LAYOUT::GENERAL;
			mortonPreBarrier.image = m_destinationImage.get();
			mortonPreBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			mortonPreBarrier.subresourceRange.baseMipLevel = 0;
			mortonPreBarrier.subresourceRange.levelCount = 1;
			mortonPreBarrier.subresourceRange.baseArrayLayer = 0;
			mortonPreBarrier.subresourceRange.layerCount = 1;
			mortonPreBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			mortonPreBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			mortonPreBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			mortonPreBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&mortonPreBarrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_mortonStorePipeline.get());
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &storePc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> mortonMidBarrier = {};
			mortonMidBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			mortonMidBarrier.newLayout = IImage::LAYOUT::GENERAL;
			mortonMidBarrier.image = m_destinationImage.get();
			mortonMidBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			mortonMidBarrier.subresourceRange.baseMipLevel = 0;
			mortonMidBarrier.subresourceRange.levelCount = 1;
			mortonMidBarrier.subresourceRange.baseArrayLayer = 0;
			mortonMidBarrier.subresourceRange.layerCount = 1;
			mortonMidBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			mortonMidBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			mortonMidBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			mortonMidBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&mortonMidBarrier, 1} });
		}

		m_cmdbuf->bindComputePipeline(m_mortonLoadPipeline.get());
		m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, sets);

		SPushConstantData mortonLoadPc = {
			.deviceBufferAddress = m_mortonReadbackBuffer->getDeviceAddress(),
			.dstOffsetX = 0,
			.dstOffsetY = 0,
			.srcWidth = TILE_SIZE,
			.srcHeight = TILE_SIZE
		};
		m_cmdbuf->pushConstants(m_pipelineLayout.get(), hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &mortonLoadPc);
		m_cmdbuf->dispatch(TILE_SIZE * TILE_SIZE / 128u, 1u, 1u);

		{
			asset::SMemoryBarrier memBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
				.dstAccessMask = ACCESS_FLAGS::HOST_READ_BIT
			};
			m_cmdbuf->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .memBarriers = {&memBarrier, 1} });
		}

		m_cmdbuf->end();

		// Submit and wait
		uint64_t semValue = m_frameIndex + 2; // +2 because value 1 was used in init
		IQueue::SSubmitInfo submitInfo = {};
		IQueue::SSubmitInfo::SCommandBufferInfo cmdBufInfo = { .cmdbuf = m_cmdbuf.get() };
		submitInfo.commandBuffers = { &cmdBufInfo, 1 };

		IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
			.semaphore = m_sem.get(),
			.value = semValue,
			.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
		};
		submitInfo.signalSemaphores = { &signalInfo, 1 };

		//RenderDoc capture on first frame
		if (m_frameIndex == 0)
			m_api->startCapture();

		m_queue->submit({ &submitInfo, 1 });

		if (m_frameIndex == 0)
			m_api->endCapture();

		ISemaphore::SWaitInfo waitInfo = { .semaphore = m_sem.get(), .value = semValue };
		m_device->blockForSemaphores({ &waitInfo, 1 });

		if (!m_readbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_readbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->invalidateMappedMemoryRanges(1, &range);
		}
		if (!m_snakeReadbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_snakeReadbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->invalidateMappedMemoryRanges(1, &range);
		}
		if (!m_mortonReadbackAlloc.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
		{
			ILogicalDevice::MappedMemoryRange range(m_mortonReadbackAlloc.memory.get(), 0, TILE_SIZE_BYTES);
			m_device->invalidateMappedMemoryRanges(1, &range);
		}

		const uint32_t* srcPixels = static_cast<const uint32_t*>(m_stagingMappedPtr);
		const uint32_t* dstPixels = static_cast<const uint32_t*>(m_readbackMappedPtr);
		uint32_t totalPixels = TILE_SIZE * TILE_SIZE;
		uint32_t matchCount = 0;
		uint32_t firstMismatchIdx = ~0u;

		for (uint32_t i = 0; i < totalPixels; i++)
		{
			if (srcPixels[i] == dstPixels[i])
				matchCount++;
			else if (firstMismatchIdx == ~0u)
				firstMismatchIdx = i;
		}

		if (matchCount == totalPixels)
		{
			if (m_frameIndex == 0)
				m_logger->log("Frame %u: Linear PASS - All %u pixels match.", ILogger::ELL_PERFORMANCE, m_frameIndex, totalPixels);
		}
		else
		{
			m_logger->log("Frame %u: Linear FAIL %u / %u pixels matched. First mismatch at pixel %u: expected 0x%08X, got 0x%08X",
				ILogger::ELL_ERROR, m_frameIndex, matchCount, totalPixels, firstMismatchIdx, srcPixels[firstMismatchIdx], dstPixels[firstMismatchIdx]);
		}

		const uint32_t* snakeDstPixels = static_cast<const uint32_t*>(m_snakeReadbackMappedPtr);
		uint32_t snakeMatchCount = 0;
		uint32_t snakeFirstMismatchIdx = ~0u;

		for (uint32_t i = 0; i < totalPixels; i++)
		{
			if (srcPixels[i] == snakeDstPixels[i])
				snakeMatchCount++;
			else if (snakeFirstMismatchIdx == ~0u)
				snakeFirstMismatchIdx = i;
		}

		if (snakeMatchCount == totalPixels)
		{
			if (m_frameIndex == 0)
				m_logger->log("Frame %u: Snake PASS All %u pixels match.", ILogger::ELL_PERFORMANCE, m_frameIndex, totalPixels);
		}
		else
		{
			m_logger->log("Frame %u: Snake FAIL %u / %u pixels matched. First mismatch at pixel %u: expected 0x%08X, got 0x%08X",
				ILogger::ELL_ERROR, m_frameIndex, snakeMatchCount, totalPixels, snakeFirstMismatchIdx, srcPixels[snakeFirstMismatchIdx], snakeDstPixels[snakeFirstMismatchIdx]);
		}

		const uint32_t* mortonDstPixels = static_cast<const uint32_t*>(m_mortonReadbackMappedPtr);
		uint32_t mortonMatchCount = 0;
		uint32_t mortonFirstMismatchIdx = ~0u;

		for (uint32_t i = 0; i < totalPixels; i++)
		{
			if (srcPixels[i] == mortonDstPixels[i])
				mortonMatchCount++;
			else if (mortonFirstMismatchIdx == ~0u)
				mortonFirstMismatchIdx = i;
		}

		if (mortonMatchCount == totalPixels)
		{
			if (m_frameIndex == 0)
				m_logger->log("Frame %u: Morton PASS All %u pixels match.", ILogger::ELL_PERFORMANCE, m_frameIndex, totalPixels);
		}
		else
		{
			m_logger->log("Frame %u: Morton FAIL %u / %u pixels matched. First mismatch at pixel %u: expected 0x%08X, got 0x%08X",
				ILogger::ELL_ERROR, m_frameIndex, mortonMatchCount, totalPixels, mortonFirstMismatchIdx, srcPixels[mortonFirstMismatchIdx], mortonDstPixels[mortonFirstMismatchIdx]);
		}

		m_frameIndex++;
	}

	bool onAppTerminated() override
	{
		runAllBenchmarks();

		m_logger->log("\nResults above. Waiting 5 seconds before exit...", ILogger::ELL_PERFORMANCE);
		std::this_thread::sleep_for(std::chrono::seconds(5));

		if (m_stagingAlloc.memory)
			m_stagingAlloc.memory->unmap();
		if (m_readbackAlloc.memory)
			m_readbackAlloc.memory->unmap();
		if (m_snakeReadbackAlloc.memory)
			m_snakeReadbackAlloc.memory->unmap();
		if (m_mortonReadbackAlloc.memory)
			m_mortonReadbackAlloc.memory->unmap();
		return true;
	}

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
	static constexpr uint32_t TILE_SIZE = 128;
	static constexpr uint32_t TILE_BYTES_PER_PIXEL = 4;
	static constexpr uint32_t TILE_SIZE_BYTES = TILE_SIZE * TILE_SIZE * TILE_BYTES_PER_PIXEL;
	static constexpr uint32_t VERIFICATION_LOOP_COUNT = 300;

	struct SPushConstantData
	{
		uint64_t deviceBufferAddress;
		uint32_t dstOffsetX;
		uint32_t dstOffsetY;
		uint32_t srcWidth;
		uint32_t srcHeight;
		uint32_t tilesPerRow;
	};

	IQueue* m_queue = nullptr;
	smart_refctd_ptr<IGPUImage> m_destinationImage;
	smart_refctd_ptr<IGPUComputePipeline> m_storePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_loadPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_snakeStorePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_snakeLoadPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_mortonStorePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_mortonLoadPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_batchedLinearPipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_batchedSnakePipeline;
	smart_refctd_ptr<IGPUComputePipeline> m_batchedMortonPipeline;
	smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout;
	smart_refctd_ptr<IGPUDescriptorSet> m_ds;
	smart_refctd_ptr<IGPUBuffer> m_stagingBuffer;
	smart_refctd_ptr<IGPUBuffer> m_readbackBuffer;
	smart_refctd_ptr<IGPUBuffer> m_snakeReadbackBuffer;
	smart_refctd_ptr<IGPUBuffer> m_mortonReadbackBuffer;
	IDeviceMemoryAllocator::SAllocation m_stagingAlloc;
	IDeviceMemoryAllocator::SAllocation m_readbackAlloc;
	IDeviceMemoryAllocator::SAllocation m_snakeReadbackAlloc;
	IDeviceMemoryAllocator::SAllocation m_mortonReadbackAlloc;
	void* m_stagingMappedPtr = nullptr;
	void* m_readbackMappedPtr = nullptr;
	void* m_snakeReadbackMappedPtr = nullptr;
	void* m_mortonReadbackMappedPtr = nullptr;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	smart_refctd_ptr<IGPUCommandBuffer> m_cmdbuf;
	smart_refctd_ptr<ISemaphore> m_sem;
	uint32_t m_frameIndex = 0;

	void runAllBenchmarks()
	{
		constexpr uint32_t STAGING_BUFFER_SIZE = 64 * 1024 * 1024;
		constexpr uint32_t FRAMES_IN_FLIGHT = 4;
		constexpr uint32_t TILES_PER_FRAME = STAGING_BUFFER_SIZE / (TILE_SIZE_BYTES * FRAMES_IN_FLIGHT);
		constexpr uint32_t TOTAL_FRAMES = 1000;

		uint32_t hostVisibleBits = m_physicalDevice->getHostVisibleMemoryTypeBits();
		uint32_t deviceLocalBits = m_physicalDevice->getDeviceLocalMemoryTypeBits();
		uint32_t hostVisibleOnlyBits = hostVisibleBits & ~deviceLocalBits;
		uint32_t hostVisibleDeviceLocalBits = hostVisibleBits & deviceLocalBits;

		m_logger->log("\n=== RUNNING BENCHMARKS ===", ILogger::ELL_PERFORMANCE);

		struct BenchmarkResult
		{
			const char* name;
			double wallGBps;
			double gpuGBps;
			double memcpyGBps;
		};
		std::vector<BenchmarkResult> results;

		//SysRAM benchmarks
		{
			smart_refctd_ptr<IGPUBuffer> benchStagingBuffer;
			IDeviceMemoryAllocator::SAllocation benchStagingAlloc;
			void* benchMappedPtr = nullptr;
			uint32_t benchBufSize = STAGING_BUFFER_SIZE;

			if (createStagingBuffer(benchBufSize, hostVisibleOnlyBits,
				"Benchmark Staging (SysRAM)", benchStagingBuffer, benchStagingAlloc, benchMappedPtr))
			{
				m_logger->log("\n--- CopyBufferToImage (SysRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rCopy = runBenchmark("CopyBufferToImage (SysRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"CopyBufferToImage (SysRAM)", rCopy.wallGBps, rCopy.gpuGBps, rCopy.memcpyGBps});

				m_logger->log("\n--- Linear Compute (SysRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rLinear = runBenchmarkCompute("Linear Compute (SysRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedLinearPipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Linear Compute (SysRAM)", rLinear.wallGBps, rLinear.gpuGBps, rLinear.memcpyGBps});

				m_logger->log("\n--- Snake Compute (SysRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rSnake = runBenchmarkCompute("Snake Compute (SysRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedSnakePipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Snake Compute (SysRAM)", rSnake.wallGBps, rSnake.gpuGBps, rSnake.memcpyGBps});

				m_logger->log("\n--- Morton Compute (SysRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rMorton = runBenchmarkCompute("Morton Compute (SysRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedMortonPipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Morton Compute (SysRAM)", rMorton.wallGBps, rMorton.gpuGBps, rMorton.memcpyGBps});

				benchStagingAlloc.memory->unmap();
			}
		}

		//BAR/VRAM benchmarks (if available)
		if (hostVisibleDeviceLocalBits)
		{
			smart_refctd_ptr<IGPUBuffer> benchStagingBuffer;
			IDeviceMemoryAllocator::SAllocation benchStagingAlloc;
			void* benchMappedPtr = nullptr;
			uint32_t benchBufSize = STAGING_BUFFER_SIZE;

			if (createStagingBuffer(benchBufSize, hostVisibleDeviceLocalBits,
				"Benchmark Staging (BAR/VRAM)", benchStagingBuffer, benchStagingAlloc, benchMappedPtr))
			{
				m_logger->log("\n--- CopyBufferToImage (BAR/VRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rCopy = runBenchmark("CopyBufferToImage (BAR/VRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"CopyBufferToImage (BAR/VRAM)", rCopy.wallGBps, rCopy.gpuGBps, rCopy.memcpyGBps});

				m_logger->log("\n--- Linear Compute (BAR/VRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rLinear = runBenchmarkCompute("Linear Compute (BAR/VRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedLinearPipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Linear Compute (BAR/VRAM)", rLinear.wallGBps, rLinear.gpuGBps, rLinear.memcpyGBps});

				m_logger->log("\n--- Snake Compute (BAR/VRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rSnake = runBenchmarkCompute("Snake Compute (BAR/VRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedSnakePipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Snake Compute (BAR/VRAM)", rSnake.wallGBps, rSnake.gpuGBps, rSnake.memcpyGBps});

				m_logger->log("\n--- Morton Compute (BAR/VRAM) ---", ILogger::ELL_PERFORMANCE);
				auto rMorton = runBenchmarkCompute("Morton Compute (BAR/VRAM)",
					benchStagingBuffer.get(), benchStagingAlloc, benchMappedPtr,
					m_destinationImage.get(), m_batchedMortonPipeline.get(), m_pipelineLayout.get(), m_ds.get(),
					TILE_SIZE, TILE_SIZE_BYTES,
					TILES_PER_FRAME, FRAMES_IN_FLIGHT, TOTAL_FRAMES, m_queue);
				results.push_back({"Morton Compute (BAR/VRAM)", rMorton.wallGBps, rMorton.gpuGBps, rMorton.memcpyGBps});

				benchStagingAlloc.memory->unmap();
			}
		}

		//Summary table
		m_logger->log("\n=== BENCHMARK RESULTS ===", ILogger::ELL_PERFORMANCE);
		m_logger->log("%-36s | Wall GB/s | GPU GB/s | Memcpy GB/s", ILogger::ELL_PERFORMANCE, "Strategy");
		m_logger->log("-------------------------------------+-----------+----------+------------", ILogger::ELL_PERFORMANCE);
		for (const auto& r : results)
		{
			m_logger->log("%-36s | %9.2f | %8.2f | %10.2f", ILogger::ELL_PERFORMANCE, r.name, r.wallGBps, r.gpuGBps, r.memcpyGBps);
		}
		m_logger->log("=====================================+===========+==========+============", ILogger::ELL_PERFORMANCE);
	}

	struct BenchResult
	{
		double wallGBps;
		double gpuGBps;
		double memcpyGBps;
	};

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

	BenchResult runBenchmark(
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
			commandBuffers[0]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&initBarrier, 1} });
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
				m_device->blockForSemaphores({ &frameWaitInfo, 1 });
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
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&barrier, 1} });

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
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&afterBarrier, 1} });

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COPY_BIT, queryPool.get(), queryStartIndex + 1);

			commandBuffers[cmdBufIndex]->end();
			auto t4 = std::chrono::high_resolution_clock::now();

			IQueue::SSubmitInfo frameSubmitInfo = {};
			IQueue::SSubmitInfo::SCommandBufferInfo frameCmdBufInfo = { .cmdbuf = commandBuffers[cmdBufIndex].get() };
			frameSubmitInfo.commandBuffers = { &frameCmdBufInfo, 1 };

			IQueue::SSubmitInfo::SSemaphoreInfo frameSignalInfo = {
				.semaphore = timelineSemaphore.get(),
				.value = ++timelineValue,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			frameSubmitInfo.signalSemaphores = { &frameSignalInfo, 1 };

			queue->submit({ &frameSubmitInfo, 1 });
			auto t5 = std::chrono::high_resolution_clock::now();

			totalWaitTime += std::chrono::duration<double>(t2 - t1).count();
			totalMemcpyTime += std::chrono::duration<double>(t3 - t2).count();
			totalRecordTime += std::chrono::duration<double>(t4 - t3).count();
			totalSubmitTime += std::chrono::duration<double>(t5 - t4).count();
		}

		// End marker is after last submit, NOT after GPU finishes.
		auto endTime = std::chrono::high_resolution_clock::now();

		ISemaphore::SWaitInfo finalWait = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({ &finalWait, 1 });

		// Read timestamps from the last completed flight of command buffers
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

		// GPU timestamps only represent the last framesInFlight frames (earlier ones were overwritten)
		double avgGpuTimePerFrame = sampledGpuTimeSeconds / framesInFlight;
		double totalGpuTimeSeconds = avgGpuTimePerFrame * totalFrames;


		double elapsedSeconds = std::chrono::duration<double>(endTime - startTime).count();
		uint64_t totalBytes = (uint64_t)totalFrames * tilesPerFrame * tileSizeBytes;
		double totalGB = totalBytes / (1024.0 * 1024.0 * 1024.0);

		double wallThroughputGBps = totalGB / elapsedSeconds;
		double gpuThroughputGBps = totalGB / totalGpuTimeSeconds;

		m_logger->log("    GPU time (extrapolated): %.3f s", ILogger::ELL_PERFORMANCE, totalGpuTimeSeconds);
		m_logger->log("    CPU submit throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, wallThroughputGBps);
		m_logger->log("    GPU only throughput:   %.2f GB/s", ILogger::ELL_PERFORMANCE, gpuThroughputGBps);

		m_logger->log("  Timing breakdown for %s:", ILogger::ELL_PERFORMANCE, strategyName);
		m_logger->log("    Wait time:   %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalWaitTime, 100.0 * totalWaitTime / elapsedSeconds);
		m_logger->log("    Memcpy time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalMemcpyTime, 100.0 * totalMemcpyTime / elapsedSeconds);
		m_logger->log("    Record time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalRecordTime, 100.0 * totalRecordTime / elapsedSeconds);
		m_logger->log("    Submit time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalSubmitTime, 100.0 * totalSubmitTime / elapsedSeconds);
		double memcpyGBps = totalGB / totalMemcpyTime;
		m_logger->log("    Memcpy speed: %.2f GB/s", ILogger::ELL_PERFORMANCE, memcpyGBps);

		return { wallThroughputGBps, gpuThroughputGBps, memcpyGBps };
	}


	double runBenchmarkImageStaging(
		const char* strategyName,
		const std::vector<smart_refctd_ptr<IGPUImage>>& stagingImages,  
		const std::vector<size_t>& imageMemoryOffsets,                  
		IDeviceMemoryAllocation* stagingMemory,                         
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
			commandBuffers[0]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&initBarrier, 1} });
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
		std::vector<uint8_t> testPatternData(tileSizeBytes);
		for (uint32_t y = 0; y < tileSize; y++)
		{
			for (uint32_t x = 0; x < tileSize; x++)
			{
				uint32_t idx = (y * tileSize + x) * 4;
				testPatternData[idx + 0] = (x * 2) & 0xFF;  
				testPatternData[idx + 1] = (y * 2) & 0xFF;  
				testPatternData[idx + 2] = 128;              
				testPatternData[idx + 3] = 255;              
			}
		}

		uint32_t tilesPerRow = imageWidth / tileSize;

		double totalWaitTime = 0.0;
		double totalMemcpyTime = 0.0;
		double totalImageCreateTime = 0.0;  
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

			IGPUImage* stagingImage = stagingImages[cmdBufIndex].get();
			size_t memoryOffset = imageMemoryOffsets[cmdBufIndex];

			void* targetPtr = static_cast<uint8_t*>(mappedPtr) + memoryOffset;
			memcpy(targetPtr, testPatternData.data(), tileSizeBytes);

			// Flush if not HOST_COHERENT
			if (!stagingMemory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			{
				ILogicalDevice::MappedMemoryRange range(stagingMemory, memoryOffset, tileSizeBytes);
				m_device->flushMappedMemoryRanges(1, &range);
			}


			auto t3 = std::chrono::high_resolution_clock::now();



			auto t4 = std::chrono::high_resolution_clock::now();

			commandBuffers[cmdBufIndex]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			uint32_t queryStartIndex = cmdBufIndex * 2;
			commandBuffers[cmdBufIndex]->resetQueryPool(queryPool.get(), queryStartIndex, 2);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> stagingBarrier = {};
			stagingBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			stagingBarrier.newLayout = IImage::LAYOUT::GENERAL;
			stagingBarrier.image = stagingImage;
			stagingBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			stagingBarrier.subresourceRange.baseMipLevel = 0;
			stagingBarrier.subresourceRange.levelCount = 1;
			stagingBarrier.subresourceRange.baseArrayLayer = 0;
			stagingBarrier.subresourceRange.layerCount = 1;
			stagingBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::HOST_WRITE_BIT;
			stagingBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT;
			stagingBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT;
			stagingBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&stagingBarrier, 1} });

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> dstBarrier = {};
			dstBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			dstBarrier.newLayout = IImage::LAYOUT::GENERAL;
			dstBarrier.image = destinationImage;
			dstBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			dstBarrier.subresourceRange.baseMipLevel = 0;
			dstBarrier.subresourceRange.levelCount = 1;
			dstBarrier.subresourceRange.baseArrayLayer = 0;
			dstBarrier.subresourceRange.layerCount = 1;
			dstBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			dstBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT;
			dstBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			dstBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {.imgBarriers = {&dstBarrier, 1}});

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COPY_BIT, queryPool.get(), queryStartIndex + 0);

			uint32_t tileIndex = frame % tilesPerRow;  
			uint32_t tileX = (tileIndex % tilesPerRow) * tileSize;
			uint32_t tileY = (tileIndex / tilesPerRow) * tileSize;

			IImage::SImageCopy copyRegion = {};
			copyRegion.srcSubresource.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			copyRegion.srcSubresource.mipLevel = 0;
			copyRegion.srcSubresource.baseArrayLayer = 0;
			copyRegion.srcSubresource.layerCount = 1;
			copyRegion.srcOffset = { 0, 0, 0 };
			copyRegion.dstSubresource = copyRegion.srcSubresource;
			copyRegion.dstOffset = { tileX, tileY, 0 };
			copyRegion.extent = { tileSize, tileSize, 1 };

			commandBuffers[cmdBufIndex]->copyImage(
				stagingImage,
				IImage::LAYOUT::GENERAL,
				destinationImage,
				IImage::LAYOUT::GENERAL,
				1,
				&copyRegion
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
			auto t5 = std::chrono::high_resolution_clock::now();

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
			auto t6 = std::chrono::high_resolution_clock::now();



			totalWaitTime += std::chrono::duration<double>(t2 - t1).count();
			totalMemcpyTime += std::chrono::duration<double>(t3 - t2).count();
			totalImageCreateTime += std::chrono::duration<double>(t4 - t3).count();
			totalRecordTime += std::chrono::duration<double>(t5 - t4).count();
			totalSubmitTime += std::chrono::duration<double>(t6 - t5).count();
		}

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

		m_logger->log("    copyImage time: %.3f s", ILogger::ELL_PERFORMANCE, totalGpuTimeSeconds);
		m_logger->log("    Total throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, throughputGBps);

		m_logger->log("  Timing breakdown for %s:", ILogger::ELL_PERFORMANCE, strategyName);
		m_logger->log("    Wait time:         %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalWaitTime, 100.0 * totalWaitTime / elapsedSeconds);
		m_logger->log("    Memcpy time:       %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalMemcpyTime, 100.0 * totalMemcpyTime / elapsedSeconds);
		m_logger->log("    Image create time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalImageCreateTime, 100.0 * totalImageCreateTime / elapsedSeconds);
		m_logger->log("    Record time:       %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalRecordTime, 100.0 * totalRecordTime / elapsedSeconds);
		m_logger->log("    Submit time:       %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalSubmitTime, 100.0 * totalSubmitTime / elapsedSeconds);
		m_logger->log("    Memcpy speed:      %.2f GB/s", ILogger::ELL_PERFORMANCE, (totalBytes / (1024.0 * 1024.0 * 1024.0)) / totalMemcpyTime);

		return throughputGBps;
	}

	BenchResult runBenchmarkCompute(
		const char* strategyName,
		IGPUBuffer* stagingBuffer,
		IDeviceMemoryAllocator::SAllocation& stagingAlloc,
		void* mappedPtr,
		IGPUImage* destinationImage,
		IGPUComputePipeline* pipeline,
		IGPUPipelineLayout* pipelineLayout,
		IGPUDescriptorSet* ds,
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
			initBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			initBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
			initBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			commandBuffers[0]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&initBarrier, 1} });
		}
		commandBuffers[0]->end();

		IQueue::SSubmitInfo submitInfo = {};
		IQueue::SSubmitInfo::SCommandBufferInfo cmdBufInfo = { .cmdbuf = commandBuffers[0].get() };
		submitInfo.commandBuffers = { &cmdBufInfo, 1 };

		IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
			.semaphore = timelineSemaphore.get(),
			.value = ++timelineValue,
			.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
		};
		submitInfo.signalSemaphores = { &signalInfo, 1 };

		queue->submit({ &submitInfo, 1 });

		ISemaphore::SWaitInfo waitInfo = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({ &waitInfo, 1 });

		uint32_t imageWidth = destinationImage->getCreationParameters().extent.width;
		uint32_t tilesPerRow = imageWidth / tileSize;
		uint32_t partitionSize = tilesPerFrame * tileSizeBytes;

		std::vector<uint8_t> cpuSourceData(partitionSize);
		{
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::mt19937 g(seed);
			uint32_t* data = reinterpret_cast<uint32_t*>(cpuSourceData.data());
			for (uint32_t i = 0; i < partitionSize / sizeof(uint32_t); i++)
				data[i] = g();
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
				m_device->blockForSemaphores({ &frameWaitInfo, 1 });
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

			asset::SMemoryBarrier memBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
				.srcAccessMask = ACCESS_FLAGS::HOST_WRITE_BIT,
				.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
			};

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> dstBarrier = {};
			dstBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			dstBarrier.newLayout = IImage::LAYOUT::GENERAL;
			dstBarrier.image = destinationImage;
			dstBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			dstBarrier.subresourceRange.baseMipLevel = 0;
			dstBarrier.subresourceRange.levelCount = 1;
			dstBarrier.subresourceRange.baseArrayLayer = 0;
			dstBarrier.subresourceRange.layerCount = 1;
			dstBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			dstBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			dstBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			dstBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, {
				.memBarriers = {&memBarrier, 1},
				.imgBarriers = {&dstBarrier, 1}
			});

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, queryPool.get(), queryStartIndex + 0);

			commandBuffers[cmdBufIndex]->bindComputePipeline(pipeline);
			const IGPUDescriptorSet* sets[] = { ds };
			commandBuffers[cmdBufIndex]->bindDescriptorSets(asset::EPBP_COMPUTE, pipelineLayout, 0, 1, sets);

			// Single dispatch covering all tiles at once
			SPushConstantData pc = {
				.deviceBufferAddress = stagingBuffer->getDeviceAddress() + bufferOffset,
				.dstOffsetX = 0,
				.dstOffsetY = 0,
				.srcWidth = tileSize,
				.srcHeight = tileSize,
				.tilesPerRow = tilesPerRow
			};
			commandBuffers[cmdBufIndex]->pushConstants(pipelineLayout, hlsl::ShaderStage::ESS_COMPUTE, 0, sizeof(SPushConstantData), &pc);
			commandBuffers[cmdBufIndex]->dispatch(tilesPerFrame * tileSize * tileSize / 128u, 1u, 1u);

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> afterBarrier = {};
			afterBarrier.oldLayout = IImage::LAYOUT::GENERAL;
			afterBarrier.newLayout = IImage::LAYOUT::GENERAL;
			afterBarrier.image = destinationImage;
			afterBarrier.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			afterBarrier.subresourceRange.baseMipLevel = 0;
			afterBarrier.subresourceRange.levelCount = 1;
			afterBarrier.subresourceRange.baseArrayLayer = 0;
			afterBarrier.subresourceRange.layerCount = 1;
			afterBarrier.barrier.dep.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS;
			afterBarrier.barrier.dep.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS;
			afterBarrier.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			afterBarrier.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT;
			commandBuffers[cmdBufIndex]->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, { .imgBarriers = {&afterBarrier, 1} });

			commandBuffers[cmdBufIndex]->writeTimestamp(PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, queryPool.get(), queryStartIndex + 1);

			commandBuffers[cmdBufIndex]->end();
			auto t4 = std::chrono::high_resolution_clock::now();

			IQueue::SSubmitInfo frameSubmitInfo = {};
			IQueue::SSubmitInfo::SCommandBufferInfo frameCmdBufInfo = { .cmdbuf = commandBuffers[cmdBufIndex].get() };
			frameSubmitInfo.commandBuffers = { &frameCmdBufInfo, 1 };

			IQueue::SSubmitInfo::SSemaphoreInfo frameSignalInfo = {
				.semaphore = timelineSemaphore.get(),
				.value = ++timelineValue,
				.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
			};
			frameSubmitInfo.signalSemaphores = { &frameSignalInfo, 1 };

			queue->submit({ &frameSubmitInfo, 1 });
			auto t5 = std::chrono::high_resolution_clock::now();

			totalWaitTime += std::chrono::duration<double>(t2 - t1).count();
			totalMemcpyTime += std::chrono::duration<double>(t3 - t2).count();
			totalRecordTime += std::chrono::duration<double>(t4 - t3).count();
			totalSubmitTime += std::chrono::duration<double>(t5 - t4).count();
		}

		// End marker is after last submit, NOT after GPU finishes.
		auto endTime = std::chrono::high_resolution_clock::now();

		ISemaphore::SWaitInfo finalWait = {
			.semaphore = timelineSemaphore.get(),
			.value = timelineValue
		};
		m_device->blockForSemaphores({ &finalWait, 1 });

		// Read timestamps from the last completed flight of command buffers
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
		double totalGB = totalBytes / (1024.0 * 1024.0 * 1024.0);

		double wallThroughputGBps = totalGB / elapsedSeconds;
		double gpuThroughputGBps = totalGB / totalGpuTimeSeconds;

		m_logger->log("    GPU time (extrapolated): %.3f s", ILogger::ELL_PERFORMANCE, totalGpuTimeSeconds);
		m_logger->log("    CPU submit throughput: %.2f GB/s", ILogger::ELL_PERFORMANCE, wallThroughputGBps);
		m_logger->log("    GPU only throughput:   %.2f GB/s", ILogger::ELL_PERFORMANCE, gpuThroughputGBps);

		m_logger->log("  Timing breakdown for %s:", ILogger::ELL_PERFORMANCE, strategyName);
		m_logger->log("    Wait time:   %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalWaitTime, 100.0 * totalWaitTime / elapsedSeconds);
		m_logger->log("    Memcpy time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalMemcpyTime, 100.0 * totalMemcpyTime / elapsedSeconds);
		m_logger->log("    Record time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalRecordTime, 100.0 * totalRecordTime / elapsedSeconds);
		m_logger->log("    Submit time: %.3f s (%.1f%%)", ILogger::ELL_PERFORMANCE, totalSubmitTime, 100.0 * totalSubmitTime / elapsedSeconds);
		double memcpyGBps = totalGB / totalMemcpyTime;
		m_logger->log("    Memcpy speed: %.2f GB/s", ILogger::ELL_PERFORMANCE, memcpyGBps);

		return { wallThroughputGBps, gpuThroughputGBps, memcpyGBps };
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
		params.usage = IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
		outBuffer = m_device->createBuffer(std::move(params));
		if (!outBuffer)
			return logFail("Failed to create GPU buffer of size %d!\n", bufferSize);

		outBuffer->setObjectDebugName(debugName);

		auto reqs = outBuffer->getMemoryReqs();
		reqs.memoryTypeBits &= memoryTypeBits;

		outAllocation = m_device->allocate(reqs, outBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
		if (!outAllocation.isValid())
			return logFail("Failed to allocate Device Memory!\n");

		outMappedPtr = outAllocation.memory->map({ 0ull, outAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_WRITE);
		if (!outMappedPtr)
			return logFail("Failed to map Device Memory!\n");

		return true;
	}
};

NBL_MAIN_FUNC(ImageUploadBenchmarkApp)