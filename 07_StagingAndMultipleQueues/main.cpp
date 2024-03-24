// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: improve validation of writes (IDescriptor::E_TYPE IGPUDescriptorSet::validateWrite)

// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

//std::this_thread::sleep_for(std::chrono::seconds(1)); // TODO: remove

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"

#if 0
// This time we let the new base class score and pick queue families, as well as initialize `nbl::video::IUtilities` for us
class StagingAndMultipleQueuesApp final : public examples::BasicMultiQueueApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::BasicMultiQueueApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

	static constexpr uint32_t IMAGE_CNT = 3u;
	static constexpr std::array<std::string_view, IMAGE_CNT> imagesToLoad = {
		"../app_resources/test0.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png"
	};

public:
	// Yay thanks to multiple inheritance we cannot forward ctors anymore
	StagingAndMultipleQueuesApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	// This time we will load images and compute their histograms and output them as CSV
	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Remember to call the base class initialization!
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// TODO: create all semaphores before going into threads
		// TODO: create/initialize array of atomic pointers to IGPUImage* and IGPUBuffer* to hold results

		std::thread loadImagesThread(&StagingAndMultipleQueuesApp::loadImages, this);
		std::thread saveHistogramsThread(&StagingAndMultipleQueuesApp::saveHistograms, this);

		calculateHistograms();

		loadImagesThread.join();
		saveHistogramsThread.join();

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
	// Override will become irrelevant in the vulkan_1_3 branch
	SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		retval.shaderStorageImageWriteWithoutFormat = true;
		retval.vulkanMemoryModelDeviceScope = true; // Needed for atomic operations.
		return retval;
	}

	// Ideally don't want to have to 
	SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval = device_base_t::getPreferredDeviceFeatures();
		retval.shaderStorageImageReadWithoutFormat = true;
		return retval;
	}

	template<typename... Args>
	void logFailAndTerminate(const char* msg, Args&&... args)
	{
		m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
		std::exit(-1);
	}

private:
	smart_refctd_ptr<ISemaphore> m_uploadSemaphore, m_processSemaphore, m_saveSeamphore;
	std::array<core::smart_refctd_ptr<IGPUImage>, IMAGE_CNT> images;

	smart_refctd_ptr<IGPUBuffer> histogramBuffer = nullptr;
	nbl::video::IDeviceMemoryAllocator::SMemoryOffset histogramBufferAllocation = {};

	std::mutex imagesReadyMutex;
	std::condition_variable imagesReady;
	bool imagesReadyFlag = false;

	std::mutex histogramBufferReadyMutex;
	std::condition_variable histogramBufferReady;
	bool histogramBufferReadyFlag = false;

	void loadImages()
	{
		IAssetLoader::SAssetLoadParams lp;
		lp.logger = m_logger.get();

		// TODO: In each thread make FRAMES_IN_FLIGHT commandPools[] with ECF_NONE ( 1 pool : 1 command buffer)
		core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> commandPoolFlags = static_cast<IGPUCommandPool::E_CREATE_FLAGS>(IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
		smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), commandPoolFlags);

		// TODO: loop over images to load
			// TODO: If imageIndex>=FRAMES_IN_FLIGHT need to block for Semaphore at value imageIndex+1-FRAMES_IN_FLIGHT
			// TODO: Create IGPUImage with matching parameters, EXCEPT format, format we promote using IPHysicalDevice
			// TODO: just use Transfer UP queue with IUtilities to fill image
			// TODO: layout transition from TRANSFER_DST to SHADER_READ_ONLY/SAMPLED
			// TODO: make sure to use the non-blocking variant of `IUtilities` to control the `IQueue::SSubmitInfo::signalSemaphores`

		core::smart_refctd_ptr<IGPUCommandBuffer> transferCmdBuffer;
		core::smart_refctd_ptr<IGPUCommandBuffer> computeCmdBuffer;
		m_device->createCommandBuffers(commandPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmdBuffer);
		m_device->createCommandBuffers(commandPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &computeCmdBuffer);

		IGPUObjectFromAssetConverter cpu2gpu;
		IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		cpu2gpuParams.assetManager = m_assetMgr.get();
		cpu2gpuParams.device = m_device.get();
		cpu2gpuParams.finalQueueFamIx = getGraphicsQueue()->getFamilyIndex();
		cpu2gpuParams.utilities = m_utils.get();
		cpu2gpuParams.pipelineCache = nullptr;

		auto trasferSempahore = m_device->createSemaphore();
		auto computeSempahore = m_device->createSemaphore();

		// TODO: make sure it make sense to use graphics queue twice
		// both compute and transfer queues here require graphics capability
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].queue = getGraphicsQueue();
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].queue = getGraphicsQueue();
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].cmdbuf = transferCmdBuffer;
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].cmdbuf = computeCmdBuffer;
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_TRANSFER].semaphore = &trasferSempahore;
		cpu2gpuParams.perQueue[IGPUObjectFromAssetConverter::EQU_COMPUTE].semaphore = &computeSempahore;

		std::array<core::smart_refctd_ptr<ICPUImage>, IMAGE_CNT> cpuImages;

		for (uint32_t i = 0u; i < IMAGE_CNT; i++)
		{
			const path pth = imagesToLoad[i];

			auto assetBundle = m_assetMgr->getAsset(pth.string(), lp);
			assert(assetBundle.getAssetType() == IAsset::E_TYPE::ET_IMAGE);

			cpuImages[i] = IAsset::castDown<ICPUImage>(*assetBundle.getContents().begin());
			assert(cpuImages[i].get() != nullptr);

			cpuImages[i]->getCreationParameters();
			cpuImages[i]->setImageUsageFlags(IImage::E_USAGE_FLAGS::EUF_STORAGE_BIT);
		}

		getGraphicsQueue()->startCapture();
		cpu2gpuParams.beginCommandBuffers();

		auto gpu_image_array = cpu2gpu.getGPUObjectsFromAssets(cpuImages.data(), cpuImages.data() + cpuImages.size(), cpu2gpuParams);
		if (!gpu_image_array || gpu_image_array->size() != IMAGE_CNT)
			logFailAndTerminate("Failed to convert from ICPUImage to IGPUImage.");

		// Do the submits ourselves, dont wait
		cpu2gpuParams.waitForCreationToComplete();

		// do the debug names BEFORE the submit
		core::smart_refctd_ptr<IGPUImage> image;
		for (uint32_t i = 0u; i < IMAGE_CNT; i++)
		{
			if (!(*gpu_image_array)[i])
				logFailAndTerminate("Failed to convert from ICPUImage to IGPUImage.");

			image = (*gpu_image_array)[i];
			const std::string imageDbgName = std::string("Image ") + std::to_string(i);
			image->setObjectDebugName(imageDbgName.c_str());
			images[i] = std::move(image);
		}

		getGraphicsQueue()->endCapture();

		imagesReadyFlag = true;
		imagesReady.notify_one();
	}

	void calculateHistograms()
	{
		const size_t histogramBufferByteSize = IMAGE_CNT * 256u * 3u * sizeof(uint32_t);

		// TODO: create FRAMES_IN_FLIGHT commandpools, buffers and descriptor sets

		// Create histogram buffer.
		// TODO: create FRAMES_IN_FLIGHT count buffers and reuse
		{
			IGPUBuffer::SCreationParams gpuBufCreationParams;
			gpuBufCreationParams.size = histogramBufferByteSize;
			gpuBufCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
			histogramBuffer = m_device->createBuffer(std::move(gpuBufCreationParams));
			if (!histogramBuffer)
				logFailAndTerminate("Failed to create a GPU Buffer of size %d!\n", gpuBufCreationParams.size);

			histogramBuffer->setObjectDebugName("Histogram Buffer");

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = histogramBuffer->getMemoryReqs();
			// you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

			histogramBufferAllocation = m_device->allocate(reqs, histogramBuffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
			if (!histogramBufferAllocation.isValid())
				logFailAndTerminate("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

			assert(histogramBuffer->getBoundMemory() == histogramBufferAllocation.memory.get());
		}

		// TODO: create the pipeline BEFORE anything else
		smart_refctd_ptr<nbl::video::IGPUSpecializedShader> specShader;
		{
			// Normally we'd use the ISystem and the IAssetManager to load shaders flexibly from (virtual) files for ease of development (syntax highlighting and Intellisense),
			// but I want to show the full process of assembling a shader from raw source code at least once.
			smart_refctd_ptr<nbl::asset::IShaderCompiler> compiler = make_smart_refctd_ptr<nbl::asset::CHLSLCompiler>(smart_refctd_ptr(m_system));

			// TODO: move to `app_resources` and load from file
			constexpr const char* source = R"===(
					#pragma wave shader_stage(compute)

					#include "../app_resources/common.hlsl"

					static const uint32_t RED_OFFSET = 0u;
					static const uint32_t GREEN_OFFSET = 256u;
					static const uint32_t BLUE_OFFSET = 256u * 2u;

					[[vk::binding(0,0)]] RWTexture2D<float4> texture;
					[[vk::binding(1,0)]] RWStructuredBuffer<uint> histogram;

					[[vk::push_constant]]
					PushConstants constants;

					[numthreads(WorkgroupSizeX,WorkgroupSizeY,1)]
					void main(uint32_t3 ID : SV_DispatchThreadID)
					{
						uint width;
						uint height;
						texture.GetDimensions(width, height);
						if(ID.x >= width || ID.y >= height)
							return;

						float4 texel = texture.Load(ID.xy);

						const uint32_t redVal = uint32_t(texel.r * 255u);
						const uint32_t greenVal = uint32_t(texel.g * 255u);
						const uint32_t blueVal = uint32_t(texel.b * 255u);

						InterlockedAdd(histogram[constants.histogramBufferOffset + RED_OFFSET + redVal], 1);
						InterlockedAdd(histogram[constants.histogramBufferOffset + GREEN_OFFSET + greenVal], 1);
						InterlockedAdd(histogram[constants.histogramBufferOffset + BLUE_OFFSET + blueVal], 1);
					}
				)===";

			const string imageCntAsStr = std::to_string(IMAGE_CNT);
			const string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);

			auto includeFinder = core::make_smart_refctd_ptr<IShaderCompiler::CIncludeFinder>(core::smart_refctd_ptr(m_system));

			CHLSLCompiler::SOptions options = {};
			// really we should set it to `ESS_COMPUTE` since we know, but we'll test the `#pragma` handling fur teh lulz
			options.stage = asset::IShader::E_SHADER_STAGE::ESS_UNKNOWN;
			// want as much debug as possible
			options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
			// this lets you source-level debug/step shaders in renderdoc
			if (m_device->getPhysicalDevice()->getLimits().shaderNonSemanticInfo)
				options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT;
			// if you don't set the logger and source identifier you'll have no meaningful errors
			options.preprocessorOptions.sourceIdentifier = "embedded.comp.hlsl";
			options.preprocessorOptions.logger = m_logger.get();
			options.preprocessorOptions.includeFinder = includeFinder.get();

			smart_refctd_ptr<nbl::asset::ICPUShader> cpuShader;
			if (!(cpuShader = compiler->compileToSPIRV(source, options)))
				logFailAndTerminate("Failed to compile following HLSL Shader:\n%s\n", source);

			// Note how each ILogicalDevice method takes a smart-pointer r-value, so that the GPU objects refcount their dependencies
			smart_refctd_ptr<nbl::video::IGPUShader> shader = m_device->createShader(std::move(cpuShader));
			if (!shader)
				logFailAndTerminate("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

			// we'll cover the specialization constant API in another example
			const nbl::asset::ISpecializedShader::SInfo info(nullptr, nullptr, "main");
			// theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one
			specShader = m_device->createSpecializedShader(shader.get(), info);
		}

		// TODO: loop
			// TODO: block on `m_processSemaphore` until it reached value `imageIndex+1-FRAMES_IN_FLIGHT`
			// const auto resourceIx = imageIndex%FRAMES_IN_FLIGHT
			// TODO: update descriptor set [resourceIx]
			// TODO: re-record/re-use command descriptor set[resourceIx]
			// TODO: no barriers needed because Timelien Semaphore signal/wait takes care of everything (and no layout transition on this queue)
			// TODO: when submitting, wait on `m_uploadSemaphore` and value `imageIndex+1` AND (if `imageIndex>=FRAMES_IN_FLIGHT`) `m_histogramSemaphore` and value `imageIndex+1-FRAMES_IN_FLIGHT`
		constexpr uint32_t BINDING_CNT = 2u;
		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[BINDING_CNT] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IGPUShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr
			},
			{
				.binding = 1,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IGPUShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr
			}
		};

		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings, bindings + BINDING_CNT);
		if (!dsLayout)
			logFailAndTerminate("Failed to create a Descriptor Layout!\n");

		SPushConstantRange pc;
		pc.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		pc.offset = 0u;
		pc.size = sizeof(PushConstants);

		// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout(&pc, &pc + 1, smart_refctd_ptr(dsLayout));
		if (!pplnLayout)
			logFailAndTerminate("Failed to create a Pipeline Layout!\n");

		// we use strong typing on the pipelines, since there's no reason to polymorphically switch between different pipelines
		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline = m_device->createComputePipeline(nullptr, smart_refctd_ptr(pplnLayout), std::move(specShader));

		IGPUCommandBuffer::SImageMemoryBarrier commonImageTransitionBarrier;
		{
			IImage::SSubresourceRange imgSubresourceRange{};
			imgSubresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			imgSubresourceRange.baseMipLevel = 0u;
			imgSubresourceRange.baseArrayLayer = 0u;

			commonImageTransitionBarrier.barrier.srcAccessMask = E_ACCESS_FLAGS::EAF_NONE;
			commonImageTransitionBarrier.barrier.dstAccessMask = E_ACCESS_FLAGS::EAF_MEMORY_READ_BIT | E_ACCESS_FLAGS::EAF_MEMORY_WRITE_BIT;
			commonImageTransitionBarrier.oldLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL; // TODO: do the image layout transition WHEN UPLOADING!
			commonImageTransitionBarrier.newLayout = asset::IImage::EL_GENERAL; // TODO: check if there is more suitable layout
			commonImageTransitionBarrier.srcQueueFamilyIndex = 0u;
			commonImageTransitionBarrier.dstQueueFamilyIndex = 0u;
			commonImageTransitionBarrier.image = nullptr;
			commonImageTransitionBarrier.subresourceRange = imgSubresourceRange;
		}

		std::vector<IGPUCommandBuffer::SImageMemoryBarrier> imgLayoutTransitionBarriers(IMAGE_CNT, commonImageTransitionBarrier);
		for (uint32_t i = 0u; i < IMAGE_CNT; i++)
		{
			auto& barrier = imgLayoutTransitionBarriers[i];
			barrier.subresourceRange.levelCount = images[i]->getCreationParameters().mipLevels;
			barrier.subresourceRange.layerCount = images[i]->getCreationParameters().arrayLayers;
			barrier.image = images[i];
		}

		auto computeQueue = getComputeQueue();
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::ECF_TRANSIENT_BIT);
			if (!m_device->createCommandBuffers(cmdpool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf))
				logFailAndTerminate("Failed to create Command Buffers!\n");
		}

		// Wait for other the thread to load images.
		std::unique_lock lk(imagesReadyMutex);
		imagesReady.wait(lk, [this]() { return imagesReadyFlag; });

		std::array<smart_refctd_ptr<nbl::video::IGPUDescriptorSet>, IMAGE_CNT>(descriptorSets);
		{
			for (uint32_t i = 0u; i < IMAGE_CNT; i++)
			{
				smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1);
				descriptorSets[i] = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));

				IGPUDescriptorSet::SDescriptorInfo imgInfo;
				IGPUImageView::SCreationParams params{};
				params.viewType = IImageView<IGPUImage>::ET_2D;
				params.image = images[i];
				params.format = images[i]->getCreationParameters().format;
				params.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				params.subresourceRange.layerCount = images[i]->getCreationParameters().arrayLayers;

				imgInfo.desc = m_device->createImageView(std::move(params));
				imgInfo.info.image = { .sampler = nullptr, .imageLayout = IImage::E_LAYOUT::EL_GENERAL };

				IGPUDescriptorSet::SDescriptorInfo bufInfo;
				bufInfo.desc = smart_refctd_ptr(histogramBuffer);
				bufInfo.info.buffer = { .offset = 0u, .size = histogramBuffer->getSize() };

				IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
					{.dstSet = descriptorSets[i].get(), .binding = 0, .arrayElement = 0, .count = 1, .descriptorType = IDescriptor::E_TYPE::ET_STORAGE_IMAGE, .info = &imgInfo },
					{.dstSet = descriptorSets[i].get(), .binding = 1, .arrayElement = 0, .count = 1, .descriptorType = IDescriptor::E_TYPE::ET_STORAGE_BUFFER, .info = &bufInfo }
				};

				m_device->updateDescriptorSets(2, writes, 0u, nullptr);
			}
		}

		cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

		for (auto& barrier : imgLayoutTransitionBarriers)
		{
			cmdbuf->pipelineBarrier(
				E_PIPELINE_STAGE_FLAGS::EPSF_HOST_BIT | E_PIPELINE_STAGE_FLAGS::EPSF_ALL_COMMANDS_BIT,
				E_PIPELINE_STAGE_FLAGS::EPSF_ALL_COMMANDS_BIT,
				E_DEPENDENCY_FLAGS::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u,
				&barrier
			);
		}

		// If you enable the `debugUtils` API Connection feature on a supported backend as we've done, you'll get these pretty debug sections in RenderDoc
		cmdbuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		// you want to bind the pipeline first to avoid accidental unbind of descriptor sets due to compatibility matching
		cmdbuf->bindComputePipeline(pipeline.get());

		for (uint32_t i = 0u; i < IMAGE_CNT; i++)
		{
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &descriptorSets[i].get());

			const auto imageExtent = images[i]->getCreationParameters().extent;
			const uint32_t wgCntX = imageExtent.width / WorkgroupSizeX;
			const uint32_t wgCntY = imageExtent.height / WorkgroupSizeY;

			PushConstants constants;
			constants.histogramBufferOffset = 256 * 3 * i;

			cmdbuf->pushConstants(pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(PushConstants), &constants);
			cmdbuf->dispatch(wgCntX, wgCntY, 1);
		}

		cmdbuf->endDebugMarker();

		// Normally you'd want to perform a memory barrier when using the output of a compute shader or renderpass,
		// however waiting on a timeline semaphore (or fence) on the Host makes all Device writes visible.
		cmdbuf->end();

		smart_refctd_ptr<IGPUFence> done = m_device->createFence(IGPUFence::ECF_UNSIGNALED);
		{
			// Default, we have no semaphores to wait on before we can start our workload
			IGPUQueue::SSubmitInfo submitInfo = {};
			// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
			// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
			submitInfo.commandBufferCount = 1;
			submitInfo.commandBuffers = &cmdbuf.get();

			// We have a cool integration with RenderDoc that allows you to start and end captures programmatically.
			// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
			computeQueue->startCapture();
			computeQueue->submit(1u, &submitInfo, done.get());
			computeQueue->endCapture();
		}
		// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
		m_device->blockForFences(1, &done.get());

		histogramBufferReadyFlag = true;
		histogramBufferReady.notify_one();
	}

	void saveHistograms()
	{
		std::array<std::ofstream, IMAGE_CNT> files;
		for (auto& file : files)
		{
			static uint32_t i = 0u;
			file.open("histogram_" + std::to_string(i++) + ".csv", std::ios::out | std::ios::trunc);
		}

		// TODO: loop over image indices
			// TODO: block on `m_processSemaphore` for value `imageIndex+1`
			// invalidate mapped memory
			// TODO: read and write results to file
			// TODO: `m_histogramSemaphore->signalFromHost(imageIndex+1)`
		// Wait for other the thread to load images.
		std::unique_lock lk(histogramBufferReadyMutex);
		histogramBufferReady.wait(lk, [this]() { return histogramBufferReadyFlag; });

		// TODO: map the buffers when creating them (persistent mapping)
		const IDeviceMemoryAllocation::MappedMemoryRange memoryRange(histogramBufferAllocation.memory.get(), 0ull, histogramBufferAllocation.memory->getAllocationSize());
		uint32_t* imageBufferMemPtr = static_cast<uint32_t*>(m_device->mapMemory(memoryRange, IDeviceMemoryAllocation::EMCAF_READ));
		if (!imageBufferMemPtr)
			logFailAndTerminate("Failed to map the Device Memory!\n");

		if (!histogramBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memoryRange);

		size_t offset = 0u;
		for (auto& file : files)
		{
			for (uint32_t i = 0u; i < 3u; i++)
			{
				for (uint32_t i = 0u; i < 256u; i++)
				{
					file << imageBufferMemPtr[offset] << ',';
					offset++;
				}

				file << '\n';
			}

			file.close();
		}

		// not needed at all
		m_device->unmapMemory(histogramBufferAllocation.memory.get());
	}


};
#endif

// This time we let the new base class score and pick queue families, as well as initialize `nbl::video::IUtilities` for us
class StagingAndMultipleQueuesApp final : public examples::BasicMultiQueueApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::BasicMultiQueueApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

	static constexpr std::array imagesToLoad = {
		"../app_resources/test0.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png",
		"../app_resources/test0.png",
		"../app_resources/test1.png",
		"../app_resources/test2.png"
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
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		// TODO: create all semaphores before going into threads

		constexpr size_t TIMELINE_SEMAPHORE_STARTING_VALUE = 0;
		m_imagesLoadedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);
		m_imagesProcessedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);
		m_imagesDownloadedSemaphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);
		m_imagesSavedSeamphore = m_device->createSemaphore(TIMELINE_SEMAPHORE_STARTING_VALUE);

		// TODO: create/initialize array of atomic pointers to IGPUImage* and IGPUBuffer* to hold results

		std::thread loadImagesThread(&StagingAndMultipleQueuesApp::loadImages, this);
		//std::thread saveHistogramsThread(&StagingAndMultipleQueuesApp::saveHistograms, this);

		calculateHistograms();

		loadImagesThread.join();
		//saveHistogramsThread.join();

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
	// Override will become irrelevant in the vulkan_1_3 branch
	SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retval = device_base_t::getRequiredDeviceFeatures();
		//retval.shaderStorageImageWriteWithoutFormat = true;
		//retval.vulkanMemoryModelDeviceScope = true; // Needed for atomic operations.
		return retval;
	}

	// Ideally don't want to have to 
	SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
	{
		auto retval = device_base_t::getPreferredDeviceFeatures();
		//retval.shaderStorageImageReadWithoutFormat = true;
		return retval;
	}

	template<typename... Args>
	void logFailAndTerminate(const char* msg, Args&&... args)
	{
		m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
		std::exit(-1);
	}

private:
	smart_refctd_ptr<ISemaphore> m_imagesLoadedSemaphore, m_imagesProcessedSemaphore, m_imagesDownloadedSemaphore, m_imagesSavedSeamphore;
	std::atomic<uint32_t> m_imagesLoadedCnt, m_imagesProcessedCnt, m_imagesDownloadedCnt, m_imagesSavedCnt;
	std::array<core::smart_refctd_ptr<IGPUImage>, IMAGE_CNT> images;

	static constexpr uint32_t FRAMES_IN_FLIGHT = 3u;
	smart_refctd_ptr<video::IGPUCommandPool> commandPools[FRAMES_IN_FLIGHT];

	smart_refctd_ptr<IGPUBuffer> histogramBuffer = nullptr;
	nbl::video::IDeviceMemoryAllocator::SAllocation histogramBufferAllocation = {};

	std::mutex assetManagerMutex; // TODO: make function for loading assets

	void loadImages()
	{
		IAssetLoader::SAssetLoadParams lp;
		lp.logger = m_logger.get();

		// LOAD CPU IMAGES
		core::smart_refctd_ptr<ICPUImage> cpuImages[IMAGE_CNT];
		{
			std::lock_guard<std::mutex> assetManagerLock(assetManagerMutex);

			for(uint32_t i = 0; i < IMAGE_CNT; ++i)
			{
				SAssetBundle bundle = m_assetMgr->getAsset(imagesToLoad[i], lp);

				if (bundle.getContents().empty())
					logFailAndTerminate("Couldn't load an image.", ILogger::ELL_ERROR);

				cpuImages[i] = IAsset::castDown<ICPUImage>(bundle.getContents()[0]);
				if(!cpuImages[i])
					logFailAndTerminate("Asset loaded is not an image.", ILogger::ELL_ERROR);
			}
		}

		// TODO: In each thread make FRAMES_IN_FLIGHT commandPools[] with ECF_NONE ( 1 pool : 1 command buffer)
		auto transferUpQueue = getTransferUpQueue();
		// TODO: i want to use IGPUCommandPool::CREATE_FLAGS::NONE but `updateImageViaStagingBuffer` requires command buffers to be resetable
		const core::bitflag<IGPUCommandPool::CREATE_FLAGS> commandPoolFlags = static_cast<IGPUCommandPool::CREATE_FLAGS>(IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, FRAMES_IN_FLIGHT> commandPools;
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>, FRAMES_IN_FLIGHT> commandBuffers;
		std::fill(commandPools.begin(), commandPools.end(), nullptr);
		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			commandPools[i] = m_device->createCommandPool(transferUpQueue->getFamilyIndex(), commandPoolFlags);
			commandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, std::span(commandBuffers.data() + i, 1), core::smart_refctd_ptr(m_logger));
		}

		// TODO: loop over images to load
			// DONE: If imageIndex>=FRAMES_IN_FLIGHT need to block for Semaphore at value imageIndex+1-FRAMES_IN_FLIGHT
			// DONE: Create IGPUImage with matching parameters, EXCEPT format, format we promote using IPHysicalDevice
			// TODO: just use Transfer UP queue with IUtilities to fill image
			// TODO: layout transition from TRANSFER_DST to SHADER_READ_ONLY/SAMPLED
			// TODO: make sure to use the non-blocking variant of `IUtilities` to control the `IQueue::SSubmitInfo::signalSemaphores`

		size_t imageIdx = 0;
		for (uint32_t i = 0; i < IMAGE_CNT; ++i)
		{
			const size_t resourceIdx = imageIdx % FRAMES_IN_FLIGHT;
			const auto& imageToLoad = imagesToLoad[i];
			auto& cmdBuff = commandBuffers[resourceIdx]; 
			// block if  imageIdx >= FRAMES_IN_FLIGHT
			if (imageIdx >= FRAMES_IN_FLIGHT)
			{
				const ISemaphore::SWaitInfo cmdBufDonePending[] = {
					{
						.semaphore = m_imagesLoadedSemaphore.get(),
						.value = imageIdx + 1 - FRAMES_IN_FLIGHT
					}
				};

				if (m_device->blockForSemaphores(cmdBufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					logFailAndTerminate("Couldn't block for the `m_imagesLoadedSemaphore`.");

				cmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
			}


			IGPUImage::SCreationParams imgParams;
			imgParams.type = IImage::E_TYPE::ET_2D;
			imgParams.extent.height = 100;
			imgParams.extent.width = 200;
			imgParams.extent.depth = 1u;
			IPhysicalDevice::SImageFormatPromotionRequest formatPromotionRequest;
			IPhysicalDevice::SFormatImageUsages::SUsage usage;
			usage.sampledImage = 1;
			usage.transferDst = 1;
			formatPromotionRequest.usages = usage;
			imgParams.format = m_physicalDevice->promoteImageFormat(formatPromotionRequest, IGPUImage::TILING::OPTIMAL);
			imgParams.mipLevels = 1u;
			imgParams.flags = IImage::ECF_NONE;
			imgParams.arrayLayers = 1u;
			imgParams.samples = IImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT;
			imgParams.usage = asset::IImage::EUF_TRANSFER_DST_BIT | asset::IImage::EUF_SAMPLED_BIT; 
			constexpr uint32_t FAMILY_INDICES_CNT = 2; // TODO: test on intel integrated GPU (which allows only one queue family)
			uint32_t familyIndices[FAMILY_INDICES_CNT] = { getTransferUpQueue()->getFamilyIndex(), getComputeQueue()->getFamilyIndex() };
			imgParams.queueFamilyIndexCount = FAMILY_INDICES_CNT;
			imgParams.queueFamilyIndices = familyIndices;
			imgParams.preinitialized = false;

			// TODO: load actual images
			images[i] = m_device->createImage(std::move(imgParams));
			auto imageAllocation = m_device->allocate(images[i]->getMemoryReqs(), images[i].get(), IDeviceMemoryAllocation::EMAF_NONE);

			if (!imageAllocation.isValid())
				logFailAndTerminate("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imageLayoutTransitionBarrier0; // TODO: better names maybe?
			{
				IImage::SSubresourceRange imgSubresourceRange{};
				imgSubresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgSubresourceRange.baseMipLevel = 0u;
				imgSubresourceRange.baseArrayLayer = 0u;
				imgSubresourceRange.levelCount = 1;
				imgSubresourceRange.layerCount = 1u;

				imageLayoutTransitionBarrier0.barrier.dep.srcAccessMask = ACCESS_FLAGS::NONE;
				imageLayoutTransitionBarrier0.barrier.dep.dstAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS;
				imageLayoutTransitionBarrier0.oldLayout = asset::IImage::LAYOUT::UNDEFINED;
				imageLayoutTransitionBarrier0.newLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL; // TODO: use more suitable layout
				imageLayoutTransitionBarrier0.image = images[i].get();
				imageLayoutTransitionBarrier0.subresourceRange = imgSubresourceRange;
			}

			IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imageLayoutTransitionBarrier1;
			{
				IImage::SSubresourceRange imgSubresourceRange{};
				imgSubresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
				imgSubresourceRange.baseMipLevel = 0u;
				imgSubresourceRange.baseArrayLayer = 0u;
				imgSubresourceRange.levelCount = 1u;
				imgSubresourceRange.layerCount = 1u;

				imageLayoutTransitionBarrier1.barrier.dep.srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS; 
				imageLayoutTransitionBarrier1.barrier.dep.dstAccessMask = ACCESS_FLAGS::NONE;
				imageLayoutTransitionBarrier1.oldLayout = asset::IImage::LAYOUT::TRANSFER_DST_OPTIMAL;
				imageLayoutTransitionBarrier1.newLayout = asset::IImage::LAYOUT::GENERAL; // TODO: use more suitable layout
				imageLayoutTransitionBarrier1.image = images[i].get();
				imageLayoutTransitionBarrier1.subresourceRange = imgSubresourceRange;
			}

			core::smart_refctd_ptr<ISemaphore> imgFillSemaphore = m_device->createSemaphore(0); // TODO: don't create semaphore every iteration
			IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = { {.cmdbuf = cmdBuff.get()} };

			IQueue::SSubmitInfo::SCommandBufferInfo imgFillCmdBuffInfo = { cmdBuff.get() };
			IQueue::SSubmitInfo::SSemaphoreInfo imgFillSemaphoreWaitInfo = {
				.semaphore = imgFillSemaphore.get(),
				.value = 1,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};

			IQueue::SSubmitInfo imgFillSubmitInfo = {
				.waitSemaphores = {&imgFillSemaphoreWaitInfo, 1},
				.commandBuffers = {&imgFillCmdBuffInfo, 1}
			};

			transferUpQueue->submit({ &imgFillSubmitInfo, 1 });

			IQueue::SSubmitInfo::SSemaphoreInfo imgFillSemaphoreInfo =
			{
				.semaphore = imgFillSemaphore.get(),
				.value = 0,
				.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
			};
			SIntendedSubmitInfo intendedSubmit = {
				.frontHalf = {.queue = transferUpQueue, .waitSemaphores = {}, .commandBuffers = cmdBufs}, .signalSemaphores = {&imgFillSemaphoreInfo, 1}
			};
			
			cmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);


			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pplnBarrierDepInfo0;
			pplnBarrierDepInfo0.imgBarriers = std::span(&imageLayoutTransitionBarrier0, &imageLayoutTransitionBarrier0 + 1);
			if (!cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, pplnBarrierDepInfo0))
				logFailAndTerminate("Failed to issue barrier!\n");

			transferUpQueue->startCapture();
			bool uploaded = m_utils->updateImageViaStagingBuffer(
				intendedSubmit, cpuImages[i]->getBuffer(), cpuImages[i]->getCreationParameters().format,
				images[i].get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, cpuImages[i]->getRegions()
			);
			if (!uploaded)
				logFailAndTerminate("Couldn't update image data.\n");

			IGPUCommandBuffer::SPipelineBarrierDependencyInfo pplnBarrierDepInfo1;
			pplnBarrierDepInfo1.imgBarriers = std::span(&imageLayoutTransitionBarrier1, &imageLayoutTransitionBarrier1 + 1);

			if(!cmdBuff->pipelineBarrier(E_DEPENDENCY_FLAGS::EDF_NONE, pplnBarrierDepInfo1))
				logFailAndTerminate("Failed to issue barrier!\n");

			cmdBuff->end();

			core::smart_refctd_ptr<ISemaphore> allDoneSemaphore = m_device->createSemaphore(0u);
			IQueue::SSubmitInfo submitInfo[1];
			IQueue::SSubmitInfo::SCommandBufferInfo cmdBuffSubmitInfo[] = { {cmdBuff.get()} };
			IQueue::SSubmitInfo::SSemaphoreInfo signalSemaphoreSubmitInfo[] = { { .semaphore = m_imagesLoadedSemaphore.get(), .value = imageIdx+1, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS } };
			submitInfo[0].commandBuffers = cmdBuffSubmitInfo;
			submitInfo[0].signalSemaphores = signalSemaphoreSubmitInfo;
			getTransferUpQueue()->submit(submitInfo);
			transferUpQueue->endCapture();

			// this is for basic testing purposes, will be deleted ofc
			std::cout << "Image nr " << imageIdx << " loaded. Resource idx: " << resourceIdx << '\n';
			imageIdx++;
		}
	}

	void calculateHistograms()
	{
		// INITIALIZE COMMON DATA
		auto computeQueue = getComputeQueue();
		const core::bitflag<IGPUCommandPool::CREATE_FLAGS> commandPoolFlags = static_cast<IGPUCommandPool::CREATE_FLAGS>(IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, FRAMES_IN_FLIGHT> commandPools;
		std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>, FRAMES_IN_FLIGHT> commandBuffers;
		core::smart_refctd_ptr<IGPUDescriptorSet> descSets[FRAMES_IN_FLIGHT];
		std::fill(commandPools.begin(), commandPools.end(), nullptr);
		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER,
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
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout[1] = { m_device->createDescriptorSetLayout(bindings) };
		if (!dsLayout[0])
			logFailAndTerminate("Failed to create a Descriptor Layout!\n");
		smart_refctd_ptr<nbl::video::IDescriptorPool> descPools[FRAMES_IN_FLIGHT] = {
			m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(&dsLayout[0].get(), 1)),
			m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(&dsLayout[0].get(), 1)),
			m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, std::span(&dsLayout[0].get(), 1))
		};

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			commandPools[i] = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), commandPoolFlags);
			commandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, std::span(commandBuffers.data() + i, 1), core::smart_refctd_ptr(m_logger));
			
			descSets[i] = descPools[i]->createDescriptorSet(core::smart_refctd_ptr(dsLayout[0]));
		}

		// LOAD SHADER FROM FILE
		smart_refctd_ptr<ICPUShader> source;
		{
			std::lock_guard<std::mutex> assetManagerLock(assetManagerMutex);

			IAssetLoader::SAssetLoadParams lp;
			lp.logger = m_logger.get();
			auto assetBundle = m_assetMgr->getAsset("../app_resources/comp_shader.hlsl", lp);
			assert(assetBundle.getAssetType() == IAsset::E_TYPE::ET_SHADER);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				logFailAndTerminate("Could not load shader!");

			// It would be super weird if loading a shader from a file produced more than 1 asset
			assert(assets.size() == 1);
			source = IAsset::castDown<ICPUShader>(assets[0]);
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
		pc[0].offset = 0u;
		pc[0].size = sizeof(PushConstants);

		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		{
			// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
			smart_refctd_ptr<IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout(pc, smart_refctd_ptr(dsLayout[0]));
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

		// CREATE HISTOGRAM BUFFER
		{
			const size_t histogramBufferByteSize = IMAGE_CNT * 256u * 3u * sizeof(uint32_t);

			IGPUBuffer::SCreationParams gpuBufCreationParams;
			gpuBufCreationParams.size = histogramBufferByteSize;
			gpuBufCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT;
			histogramBuffer = m_device->createBuffer(std::move(gpuBufCreationParams));
			if (!histogramBuffer)
				logFailAndTerminate("Failed to create a GPU Buffer of size %d!\n", gpuBufCreationParams.size);

			histogramBuffer->setObjectDebugName("Histogram Buffer");

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = histogramBuffer->getMemoryReqs();
			// you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
			reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

			histogramBufferAllocation = m_device->allocate(reqs, histogramBuffer.get(), nbl::video::IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE);
			if (!histogramBufferAllocation.isValid())
				logFailAndTerminate("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

			//assert(histogramBuffer->getBoundMemory() == histogramBufferAllocation.memory.get());
		}

		// PROCESS IMAGES
		size_t imageIdx = 0;
		for (uint32_t imageToProcessId = 0; imageToProcessId < IMAGE_CNT; imageToProcessId++)
		{
			const ISemaphore::SWaitInfo imagesReadyToBeProcessed[] = {
					{
						.semaphore = m_imagesLoadedSemaphore.get(),
						.value = imageIdx + 1
					}
			};
			if (m_device->blockForSemaphores(imagesReadyToBeProcessed) != ISemaphore::WAIT_RESULT::SUCCESS)
				logFailAndTerminate("Couldn't block for the `m_imagesLoadedSemaphore`.");

			if (imageIdx >= FRAMES_IN_FLIGHT)
			{
				const ISemaphore::SWaitInfo cmdBufDonePending[] = {
					{
						.semaphore = m_imagesProcessedSemaphore.get(),
						.value = imageIdx + 1 - FRAMES_IN_FLIGHT
					}
				};

				if (m_device->blockForSemaphores(cmdBufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS)
					logFailAndTerminate("Couldn't block for the `m_imagesProcessedSemaphore`.");
			}

			const auto resourceIdx = imageIdx % FRAMES_IN_FLIGHT;
			auto& cmdBuf = commandBuffers[resourceIdx];

			// UPDATE DESCRIPTOR SET WRITES
			IGPUDescriptorSet::SDescriptorInfo imgInfo;
			IGPUImageView::SCreationParams params{};
			params.viewType = IImageView<IGPUImage>::ET_2D;
			params.image = images[imageIdx];
			params.format = images[imageIdx]->getCreationParameters().format;
			params.subresourceRange.aspectMask = IImage::E_ASPECT_FLAGS::EAF_COLOR_BIT;
			params.subresourceRange.layerCount = images[resourceIdx]->getCreationParameters().arrayLayers;

			IGPUSampler::SParams samplerParams;
			samplerParams.AnisotropicFilter = false;
			core::smart_refctd_ptr<IGPUSampler> sampler = m_device->createSampler(samplerParams);

			imgInfo.desc = m_device->createImageView(std::move(params));
			if (!imgInfo.desc)
				logFailAndTerminate("Couldn't create descriptor.");
			imgInfo.info.image = { .sampler = sampler, .imageLayout = IImage::LAYOUT::GENERAL };

			IGPUDescriptorSet::SDescriptorInfo bufInfo;
			bufInfo.desc = smart_refctd_ptr(histogramBuffer);
			bufInfo.info.buffer = { .offset = 0u, .size = histogramBuffer->getSize() };

			IGPUDescriptorSet::SWriteDescriptorSet writes[2] = {
				{.dstSet = descSets[resourceIdx].get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &imgInfo },
				{.dstSet = descSets[resourceIdx].get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &bufInfo }
			};

			m_device->updateDescriptorSets(2, writes, 0u, nullptr);

			cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			//cmdBuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1)); // TODO: figure it out

			cmdBuf->bindComputePipeline(pipeline.get());

			cmdBuf->end();

			IQueue::SSubmitInfo::SCommandBufferInfo cmdBufInfo = { cmdBuf.get() };
			IQueue::SSubmitInfo submitInfo = {
				.commandBuffers = { &cmdBufInfo, 1 },
			};
			computeQueue->submit({ &submitInfo, 1 });
			cmdBuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);

			std::cout << "Image nr " << imageIdx << " processed.\n";
			m_imagesProcessedSemaphore->signal(++imageIdx);
		}
	}

	void saveHistograms()
	{
		//// TODO: In each thread make FRAMES_IN_FLIGHT commandPools[] with ECF_NONE ( 1 pool : 1 command buffer)
		//const core::bitflag<IGPUCommandPool::CREATE_FLAGS> commandPoolFlags = static_cast<IGPUCommandPool::CREATE_FLAGS>(IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
		//std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, FRAMES_IN_FLIGHT> commandPools;
		//std::array<core::smart_refctd_ptr<nbl::video::IGPUCommandBuffer>, FRAMES_IN_FLIGHT> commandBuffers;
		//std::fill(commandPools.begin(), commandPools.end(), nullptr);
		//for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		//{
		//	commandPools[i] = m_device->createCommandPool(getTransferUpQueue()->getFamilyIndex(), commandPoolFlags);
		//	commandPools[i]->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, std::span(commandBuffers.data() + i, 1), core::smart_refctd_ptr(m_logger));
		//}

		//size_t imageIdx = 0;
		//for (uint32_t imageToProcessId = 0; imageToProcessId < IMAGE_CNT; imageToProcessId++)
		//{
		//	const ISemaphore::SWaitInfo imagesReadyToBeProcessed[] = {
		//			{
		//				.semaphore = m_imagesLoadedSemaphore.get(),
		//				.value = imageIdx + 1 - FRAMES_IN_FLIGHT
		//			}
		//	};

		//	if (m_device->blockForSemaphores(imagesReadyToBeProcessed) == ISemaphore::WAIT_RESULT::SUCCESS)
		//		logFailAndTerminate("Couldn't block for the `m_imagesProcessedSemaphore`.");

		//	if (imageIdx >= FRAMES_IN_FLIGHT)
		//	{
		//		const ISemaphore::SWaitInfo cmdBufDonePending[] = {
		//			{
		//				.semaphore = m_imagesProcessedSemaphore.get(),
		//				.value = imageIdx + 1 - FRAMES_IN_FLIGHT
		//			}
		//		};

		//		if (m_device->blockForSemaphores(cmdBufDonePending) == ISemaphore::WAIT_RESULT::SUCCESS)
		//			logFailAndTerminate("Couldn't block for the `m_imagesProcessedSemaphore`.");
		//	}

		//	const auto resourceIdx = imageIdx % FRAMES_IN_FLIGHT;
		//	auto& cmdBuff = commandBuffers[resourceIdx];
		//}
	}
};

NBL_MAIN_FUNC(StagingAndMultipleQueuesApp)