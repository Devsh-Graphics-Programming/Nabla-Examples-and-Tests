// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.

#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"

// This time we let the new base class score and pick queue families, as well as initialize `nbl::video::IUtilities` for us
class StagingAndMultipleQueuesApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	// TODO: would be cool if we used `system::ISystem::listItemsInDirectory(sharedInputCWD/"GLI")` as our dataset
	static constexpr std::array imagesToLoad = {
		"../app_resources/test0.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png",
		"../app_resources/test0.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png",
		"../app_resources/test0.png"
	};
	static constexpr size_t IMAGE_CNT = imagesToLoad.size();

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	StagingAndMultipleQueuesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// This time we will load images and compute their histograms and output them as CSV
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(core::smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		constexpr size_t TIMELINE_SEMAPHORE_STARTING_VALUE = 0;
		m_imagesLoadedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);
		m_imagesProcessedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);
		m_histogramSavedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);

		// TODO: create/initialize array of atomic pointers to IGPUImage* and IGPUBuffer* to hold results

		// TODO: Change the capture start/end to become methods of IAPIConnection, because our current API is not how renderdoc works
		getComputeQueue()->startCapture();
		std::thread loadImagesThread(&StagingAndMultipleQueuesApp::loadImages, this);
		std::thread saveHistogramsThread(&StagingAndMultipleQueuesApp::saveHistograms, this);

		calculateHistograms();

		loadImagesThread.join();
		saveHistogramsThread.join();
		getComputeQueue()->endCapture();

		return true;
	}

	//
	void workLoopBody() override {}

	//
	bool keepRunning() override { return false; }

	//
	bool onAppTerminated() override
	{
		return device_base_t::onAppTerminated();
	}

protected:
	template<typename... Args>
	void logFailAndTerminate(const char* msg, Args&&... args)
	{
		m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
		std::exit(-1);
	}

private:
	smart_refctd_ptr<ISemaphore> m_imagesLoadedSemaphore, m_imagesProcessedSemaphore, m_histogramSavedSemaphore;
	std::atomic<uint32_t> imageHandlesCreated = 0u;
	std::atomic<uint32_t> transfersSubmitted = 0u;
	std::array<core::smart_refctd_ptr<IGPUImage>, IMAGE_CNT> images;

	static constexpr uint32_t FRAMES_IN_FLIGHT = 3u;
	smart_refctd_ptr<video::IGPUCommandPool> commandPools[FRAMES_IN_FLIGHT];

	smart_refctd_ptr<IGPUBuffer> histogramBuffer = nullptr;
	nbl::video::IDeviceMemoryAllocator::SAllocation m_histogramBufferAllocation = {};
	// TODO: make sure ranges are ok
	std::array<ILogicalDevice::MappedMemoryRange, FRAMES_IN_FLIGHT> m_histogramBufferMemoryRanges;
	uint32_t* m_histogramBufferMemPtrs[3];

	void loadImages()
	{
		const core::set<uint32_t> uniqueFamilyIndices = { getTransferUpQueue()->getFamilyIndex(), getComputeQueue()->getFamilyIndex() };
		const std::vector<uint32_t> familyIndices(uniqueFamilyIndices.begin(),uniqueFamilyIndices.end());
		const bool multipleQueueFamilies = familyIndices.size()>1;

		IAssetLoader::SAssetLoadParams lp;
		lp.logger = m_logger.get();

		auto transferUpQueue = getTransferUpQueue();
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, FRAMES_IN_FLIGHT> commandPools;
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>, FRAMES_IN_FLIGHT> commandBuffers;
		std::fill(commandPools.begin(), commandPools.end(), nullptr);

		core::smart_refctd_ptr<ICPUImage> cpuImages[IMAGE_CNT];
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			const core::bitflag<IGPUCommandPool::CREATE_FLAGS> commandPoolFlags = IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT;
			commandPools[i] = m_device->createCommandPool(transferUpQueue->getFamilyIndex(), commandPoolFlags);
			commandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, {commandBuffers.data() + i, 1}, core::smart_refctd_ptr(m_logger));
			commandBuffers[i]->setObjectDebugName(("Upload Command Buffer #"+std::to_string(i)).c_str());
		}

		core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0);
		imgFillSemaphore->setObjectDebugName("Image Fill Semaphore");
		SIntendedSubmitInfo intendedSubmit = {
			.queue = transferUpQueue,
			.waitSemaphores = {},
			.commandBuffers = {}, // fill later
			.scratchSemaphore = {
				.semaphore = imgFillSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			}
		};

		for (uint32_t imageIdx = 0; imageIdx < IMAGE_CNT; ++imageIdx)
		{
			cpuImages[imageIdx] = loadFistAssetInBundle<ICPUImage>(imagesToLoad[imageIdx]);

			const size_t resourceIdx = imageIdx % FRAMES_IN_FLIGHT;
			const auto& imageToLoad = imagesToLoad[imageIdx];
			auto& cmdBuff = commandBuffers[resourceIdx];

			auto isResourceReused = waitForResourceAvailability(m_imagesLoadedSemaphore.get(), imageIdx);
			if(isResourceReused)
				cmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);

			IGPUImage::SCreationParams imgParams;
			imgParams.type = IImage::E_TYPE::ET_2D;
			imgParams.extent = cpuImages[imageIdx]->getCreationParameters().extent;
			IPhysicalDevice::SImageFormatPromotionRequest formatPromotionRequest;
			IPhysicalDevice::SFormatImageUsages::SUsage usage;
			usage.sampledImage = 1;
			usage.transferDst = 1;
			formatPromotionRequest.usages = usage;
			formatPromotionRequest.originalFormat = cpuImages[imageIdx]->getCreationParameters().format;
			imgParams.format = m_physicalDevice->promoteImageFormat(formatPromotionRequest, IGPUImage::TILING::OPTIMAL);
			imgParams.mipLevels = 1u;
			imgParams.flags = IImage::ECF_NONE;
			imgParams.arrayLayers = 1u;
			imgParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			imgParams.usage = asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT; 
			if (multipleQueueFamilies)
			{
				imgParams.queueFamilyIndexCount = familyIndices.size();
				imgParams.queueFamilyIndices = familyIndices.data();
			}
			imgParams.preinitialized = false;

			images[imageIdx] = m_device->createImage(std::move(imgParams));
			images[imageIdx]->setObjectDebugName(("Image #"+std::to_string(imageIdx)).c_str());
			auto imageAllocation = m_device->allocate(images[imageIdx]->getMemoryReqs(), images[imageIdx].get(), IDeviceMemoryAllocation::EMAF_NONE);
			imageHandlesCreated++;
			imageHandlesCreated.notify_one();

			if (!imageAllocation.isValid())
				logFailAndTerminate("Failed to allocate Device Memory compatible with our image!\n");

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imageLayoutTransitionBarrier0;
			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imageLayoutTransitionBarrier1;
			{
				IImage::SSubresourceRange imgSubresourceRange{};
				imgSubresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgSubresourceRange.baseMipLevel = 0u;
				imgSubresourceRange.baseArrayLayer = 0u;
				imgSubresourceRange.levelCount = 1;
				imgSubresourceRange.layerCount = 1u;

				imageLayoutTransitionBarrier0.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
				imageLayoutTransitionBarrier0.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
				imageLayoutTransitionBarrier0.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::NONE;
				imageLayoutTransitionBarrier0.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				imageLayoutTransitionBarrier0.oldLayout = asset::IImage::LAYOUT::UNDEFINED;
				imageLayoutTransitionBarrier0.newLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
				imageLayoutTransitionBarrier0.image = images[imageIdx].get();
				imageLayoutTransitionBarrier0.subresourceRange = imgSubresourceRange;

				imageLayoutTransitionBarrier1.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS; 
				imageLayoutTransitionBarrier1.barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
				imageLayoutTransitionBarrier1.barrier.dep.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT;
				imageLayoutTransitionBarrier1.barrier.dep.dstStageMask = PIPELINE_STAGE_FLAGS::NONE; // NONE because the semaphore singnal comes right after
				imageLayoutTransitionBarrier1.oldLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
				imageLayoutTransitionBarrier1.newLayout = asset::IImage::LAYOUT::READ_ONLY_OPTIMAL;
				imageLayoutTransitionBarrier1.image = images[imageIdx].get();
				imageLayoutTransitionBarrier1.subresourceRange = imgSubresourceRange;
			}
			
			IQueue::SSubmitInfo::SCommandBufferInfo imgFillCmdBuffInfo = { cmdBuff.get() };
			intendedSubmit.commandBuffers = {&imgFillCmdBuffInfo,1};
			
			cmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pplnBarrierDepInfo0;
			pplnBarrierDepInfo0.imgBarriers = { &imageLayoutTransitionBarrier0, 1 };
			if (!cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, pplnBarrierDepInfo0))
				logFailAndTerminate("Failed to issue barrier!\n");

			const uint64_t oldCntr = intendedSubmit.scratchSemaphore.value;
			const bool uploadCommendRecorded = m_utils->updateImageViaStagingBuffer(
				intendedSubmit, cpuImages[imageIdx]->getBuffer(), cpuImages[imageIdx]->getCreationParameters().format,
				images[imageIdx].get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, cpuImages[imageIdx]->getRegions()
			);
			if (!uploadCommendRecorded)
				logFailAndTerminate("Couldn't update image data.\n");

			const auto newCntr = intendedSubmit.scratchSemaphore.value;
			if (newCntr!=oldCntr)
				m_logger->log("%d overflows when uploading image %d!\n", ILogger::ELL_PERFORMANCE, newCntr-oldCntr, imageIdx);

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pplnBarrierDepInfo1;
			pplnBarrierDepInfo1.imgBarriers = { &imageLayoutTransitionBarrier1, 1 };

			if(!cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, pplnBarrierDepInfo1))
				logFailAndTerminate("Failed to issue barrier!\n");

			cmdBuff->end();

			const IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphore = {
				.semaphore=m_imagesLoadedSemaphore.get(),
				.value=imageIdx+1u,
				// cannot signal from COPY stage because there's a layout transition we need to wait for right after and it doesn't have an explicit stage
				.stageMask=PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS
			};
			getTransferUpQueue()->submit(intendedSubmit.popSubmit({&signalSemaphore,1}));
			transfersSubmitted++;
			transfersSubmitted.notify_one();


			// TODO: this is for basic testing purposes, will be deleted ofc
			std::string msg = std::string("Image nr ") + std::to_string(imageIdx) + " loaded. Resource idx: " + std::to_string(resourceIdx);
			//std::this_thread::sleep_for(std::chrono::milliseconds(6969));
			m_logger->log(msg);
		}
	}

	void calculateHistograms()
	{
		// INITIALIZE COMMON DATA
		auto computeQueue = getComputeQueue();

		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
		core::smart_refctd_ptr<IGPUDescriptorSet> descSets[FRAMES_IN_FLIGHT];
		{
			nbl::video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
				{
					.binding = 0,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, // TODO: just an image descriptor type when separable samplers arrive
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1,
					.samplers = nullptr
				},
				{
					.binding = 1,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
					.stageFlags = IGPUShader::E_SHADER_STAGE::ESS_COMPUTE,
					.count = 1,
					.samplers = nullptr
				}
			};

			dsLayout = m_device->createDescriptorSetLayout(bindings);
			if (!dsLayout)
				logFailAndTerminate("Failed to create a Descriptor Layout!\n");
			auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 }, &FRAMES_IN_FLIGHT);
			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
			{
				descSets[i] = descPool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
				descSets[i]->setObjectDebugName(("Descriptor Set #" + std::to_string(i)).c_str());
			}
		}

		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, FRAMES_IN_FLIGHT> commandPools;
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>, FRAMES_IN_FLIGHT> commandBuffers;
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			const core::bitflag<IGPUCommandPool::CREATE_FLAGS> commandPoolFlags = IGPUCommandPool::CREATE_FLAGS::NONE;
			commandPools[i] = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), commandPoolFlags);
			commandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, {commandBuffers.data() + i, 1}, core::smart_refctd_ptr(m_logger));
			commandBuffers[i]->setObjectDebugName(("Histogram Command Buffer #" + std::to_string(i)).c_str());
		}

		// LOAD SHADER FROM FILE
		smart_refctd_ptr<ICPUShader> source;
		{
			source = loadFistAssetInBundle<ICPUShader>("../app_resources/comp_shader.hlsl");
			source->setShaderStage(IShader::ESS_COMPUTE);
		}

		if (!source)
			logFailAndTerminate("Could not create a CPU shader!");

		core::smart_refctd_ptr<IGPUShader> shader = m_device->createShader(source.get());
		if(!shader)
			logFailAndTerminate("Could not create a GPU shader!");

		// CREATE COMPUTE PIPELINE
		SPushConstantRange pc[1];
		pc[0].stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		pc[0].offset = 0;
		pc[0].size = sizeof(PushConstants);

		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		smart_refctd_ptr<IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout(pc,std::move(dsLayout));
		{
			// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
			if (!pplnLayout)
				logFailAndTerminate("Failed to create a Pipeline Layout!\n");

			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			// Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			// we'll cover the specialization constant API in another example
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
				logFailAndTerminate("Failed to create pipelines (compile & link shaders)!\n");
		}

		// CREATE AND MAP HISTOGRAM BUFFER
		{
			IGPUBuffer::SCreationParams gpuBufCreationParams;
			gpuBufCreationParams.size = COMBINED_HISTOGRAM_BUFFER_BYTE_SIZE;
			gpuBufCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
			histogramBuffer = m_device->createBuffer(std::move(gpuBufCreationParams));
			if (!histogramBuffer)
				logFailAndTerminate("Failed to create a GPU Buffer of size %d!\n", COMBINED_HISTOGRAM_BUFFER_BYTE_SIZE);

			histogramBuffer->setObjectDebugName("Histogram Buffer");

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = histogramBuffer->getMemoryReqs();
			// you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits() & m_physicalDevice->getDeviceLocalMemoryTypeBits();

			m_histogramBufferAllocation = m_device->allocate(reqs, histogramBuffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
			if (!m_histogramBufferAllocation.isValid())
				logFailAndTerminate("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			assert(histogramBuffer->getBoundMemory().memory == m_histogramBufferAllocation.memory.get());

			auto memoryRange = IDeviceMemoryAllocation::MemoryRange(0, m_histogramBufferAllocation.memory->getAllocationSize());

			m_histogramBufferMemoryRanges[0] = ILogicalDevice::MappedMemoryRange(histogramBuffer->getBoundMemory().memory, 0, HISTOGRAM_BYTE_SIZE);
			m_histogramBufferMemoryRanges[1] = ILogicalDevice::MappedMemoryRange(histogramBuffer->getBoundMemory().memory, HISTOGRAM_BYTE_SIZE, HISTOGRAM_BYTE_SIZE);
			m_histogramBufferMemoryRanges[2] = ILogicalDevice::MappedMemoryRange(histogramBuffer->getBoundMemory().memory, HISTOGRAM_BYTE_SIZE * 2, HISTOGRAM_BYTE_SIZE);

			m_histogramBufferMemPtrs[0] = static_cast<uint32_t*>(m_histogramBufferAllocation.memory->map(memoryRange, IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE));
			if (!m_histogramBufferMemPtrs[0])
				logFailAndTerminate("Failed to map the Device Memory!\n");
			m_histogramBufferMemPtrs[1] = m_histogramBufferMemPtrs[0] + HISTOGRAM_SIZE;
			m_histogramBufferMemPtrs[2] = m_histogramBufferMemPtrs[1] + HISTOGRAM_SIZE;
		}

		// TODO: will no longer be necessary after separable samplers and images
		IGPUSampler::SParams samplerParams;
		samplerParams.AnisotropicFilter = false;
		core::smart_refctd_ptr<IGPUSampler> sampler = m_device->createSampler(samplerParams);

		IGPUDescriptorSet::SDescriptorInfo bufInfo;
		bufInfo.desc = smart_refctd_ptr(histogramBuffer);
		bufInfo.info.buffer = { .offset = 0u, .size = histogramBuffer->getSize() };

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			IGPUDescriptorSet::SWriteDescriptorSet write[1] = {
				{.dstSet = descSets[i].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &bufInfo }
			};
			m_device->updateDescriptorSets(1, write, 0u, nullptr);
		}

		// PROCESS IMAGES
		for (uint32_t imageToProcessId = 0; imageToProcessId < IMAGE_CNT; imageToProcessId++)
		{
			const auto resourceIdx = imageToProcessId % FRAMES_IN_FLIGHT;
			auto& cmdBuff = commandBuffers[resourceIdx];
			auto& commandPool = commandPools[resourceIdx];
			
			auto isResourceReused = waitForResourceAvailability(m_imagesProcessedSemaphore.get(), imageToProcessId);
			if (isResourceReused)
				commandPool->reset();

			// UPDATE DESCRIPTOR SET WRITES
			IGPUDescriptorSet::SDescriptorInfo imgInfo;
			IGPUImageView::SCreationParams params{};
			params.viewType = IImageView<IGPUImage>::ET_2D;

			for (auto old = imageHandlesCreated.load(); old <= imageToProcessId; old = imageHandlesCreated.load())
				imageHandlesCreated.wait(old);

			params.image = images[imageToProcessId];
			params.format = images[imageToProcessId]->getCreationParameters().format;
			params.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			params.subresourceRange.layerCount = images[imageToProcessId]->getCreationParameters().arrayLayers;

			auto view = m_device->createImageView(std::move(params));
			if (!view)
				logFailAndTerminate("Couldn't create descriptor.");
			view->setObjectDebugName(("Image View #"+std::to_string(imageToProcessId)).c_str());
			imgInfo.desc = std::move(view);
			imgInfo.info.image = { .sampler = sampler, .imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL };

			IGPUDescriptorSet::SWriteDescriptorSet write[1] = {
				{.dstSet = descSets[resourceIdx].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &imgInfo }
			};
			m_device->updateDescriptorSets(1, write, 0u, nullptr);

			cmdBuff->begin(IGPUCommandBuffer::USAGE::NONE);
			cmdBuff->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdBuff->bindComputePipeline(pipeline.get());

			cmdBuff->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &descSets[resourceIdx].get());

			const auto imageExtent = images[imageToProcessId]->getCreationParameters().extent;
			const uint32_t wgCntX = (imageExtent.width + WorkgroupSizeX - 1) / WorkgroupSizeX;
			const uint32_t wgCntY = (imageExtent.height + WorkgroupSizeY - 1) / WorkgroupSizeY;

			PushConstants constants;
			constants.histogramBufferOffset = HISTOGRAM_SIZE * resourceIdx;

			cmdBuff->pushConstants(pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(PushConstants), &constants);
			cmdBuff->dispatch(wgCntX, wgCntY, 1);

			cmdBuff->endDebugMarker();
			cmdBuff->end();

			IQueue::SSubmitInfo submitInfo[1];
			IQueue::SSubmitInfo::SCommandBufferInfo cmdBuffSubmitInfo[] = { {cmdBuff.get()} };
			IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphoreSubmitInfo[] = { {.semaphore = m_imagesProcessedSemaphore.get(), .value = imageToProcessId+1, .stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT } };
			const uint64_t histogramSaveWaitSemaphoreValue = imageToProcessId + 1 - FRAMES_IN_FLIGHT;
			IQueue::SSubmitInfo::SSemaphoreInfo waitSemaphoreSubmitInfo[] = { 
				{.semaphore = m_imagesLoadedSemaphore.get(), .value = imageToProcessId + 1, .stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT},
				{.semaphore = m_histogramSavedSemaphore.get(), .value = histogramSaveWaitSemaphoreValue, .stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}
			};
			submitInfo[0].commandBuffers = cmdBuffSubmitInfo;
			submitInfo[0].signalSemaphores = signalSemaphoreSubmitInfo;
			submitInfo[0].waitSemaphores = {waitSemaphoreSubmitInfo, imageToProcessId < FRAMES_IN_FLIGHT ? 1u : 2u};
			// Some Devices like all of the Intel GPUs do not have enough queues for us to allocate different queues to compute and transfers,
			// so our `BasicMultiQueueApplication` will "alias" a single queue to both usages. Normally you don't need to care, but here we're
			// attempting to do "out-of-order" "submit-before-signal" so we need to "hold back" submissions if the queues are aliased!
			if (getTransferUpQueue()==computeQueue)
			for (auto old = transfersSubmitted.load(); old <= imageToProcessId; old = transfersSubmitted.load())
				transfersSubmitted.wait(old);
			computeQueue->submit(submitInfo);
			std::string msg = std::string("Image nr ") + std::to_string(imageToProcessId) + " processed. Resource idx: " + std::to_string(resourceIdx);
			m_logger->log(msg);
		}
	}

	void saveHistograms()
	{
		std::array<std::ofstream, IMAGE_CNT> files;
		for (auto& file : files)
		{
			static uint32_t i = 0u;
			file.open("histogram_" + std::to_string(i++) + ".csv", std::ios::out | std::ios::trunc);
		}


		for (uint32_t imageHistogramIdx = 0; imageHistogramIdx < IMAGE_CNT; ++imageHistogramIdx)
		{
			waitForPreviousStep(m_imagesProcessedSemaphore.get(), imageHistogramIdx + 1);
			images[imageHistogramIdx] = nullptr;

			const uint32_t resourceIdx = imageHistogramIdx % FRAMES_IN_FLIGHT;
			uint32_t* histogramBuff = m_histogramBufferMemPtrs[resourceIdx];

			if(!m_device->invalidateMappedMemoryRanges(1, &m_histogramBufferMemoryRanges[resourceIdx]))
				logFailAndTerminate("Failed to invalidate the Device Memory!\n");

			size_t offset = 0;
			auto& file = files[imageHistogramIdx];
			for (uint32_t i = 0u; i < CHANEL_CNT; ++i)
			{
				constexpr const char* channelNames[] = {"RED","GREEN","BLUE"};
				file << channelNames[i] << ',';
				for (uint32_t j = 0u; j < VAL_PER_CHANEL_CNT; ++j)
				{
					file << histogramBuff[offset] << ',';
					histogramBuff[offset] = 0;
					offset++;
				}

				file << '\n';
			}

			if(!m_device->flushMappedMemoryRanges(1, &m_histogramBufferMemoryRanges[resourceIdx]))
				logFailAndTerminate("Failed to flush the Device Memory!\n");

			m_histogramSavedSemaphore->signal(imageHistogramIdx + 1);
			std::string msg = std::string("Image nr ") + std::to_string(imageHistogramIdx) + " saved. Resource idx: " + std::to_string(resourceIdx);
			m_logger->log(msg);
		}

		m_histogramBufferAllocation.memory->unmap();
		for (auto& file : files)
			file.close();
	}

	inline void waitForPreviousStep(ISemaphore* semaphore, uint32_t waitVal)
	{
		const ISemaphore::SWaitInfo imagesReadyToBeDownloaded[] = {
			{
				.semaphore = semaphore,
				.value = waitVal
			}
		};

		if (m_device->blockForSemaphores(imagesReadyToBeDownloaded) != ISemaphore::WAIT_RESULT::SUCCESS)
			logFailAndTerminate("Couldn't block for the `m_imagesProcessedSemaphore`.");
	}
	
	//! return value: inditaces if resource will be reused
	bool waitForResourceAvailability(ISemaphore* semaphore, uint32_t imageIdx)
	{
		if (imageIdx >= FRAMES_IN_FLIGHT)
		{
			const ISemaphore::SWaitInfo cmdBufDonePending[] = {
				{
					.semaphore = semaphore,
					.value = imageIdx + 1 - FRAMES_IN_FLIGHT
				}
			};

			if (m_device->blockForSemaphores(cmdBufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
				logFailAndTerminate("Couldn't block for the `m_imagesProcessedSemaphore`.");

			return true;
		}

		return false;
	}

	template<typename AssetType>
	core::smart_refctd_ptr<AssetType> loadFistAssetInBundle(const std::string& path)
	{
		IAssetLoader::SAssetLoadParams lp;
		SAssetBundle bundle = m_assetMgr->getAsset(path, lp);
		if (bundle.getContents().empty())
			logFailAndTerminate("Couldn't load an asset.", ILogger::ELL_ERROR);

		auto asset = IAsset::castDown<AssetType>(bundle.getContents()[0]);
		if (!asset)
			logFailAndTerminate("Incorrect asset loaded.", ILogger::ELL_ERROR);

		return asset;
	}
};

NBL_MAIN_FUNC(StagingAndMultipleQueuesApp)
