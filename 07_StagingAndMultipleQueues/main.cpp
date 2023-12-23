// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"


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
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// This time we will load images and compute their histograms and output them as CSV
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(std::move(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

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
		bool keepRunning() override {return false;}

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

			core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> commandPoolFlags = static_cast<IGPUCommandPool::E_CREATE_FLAGS>(IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			smart_refctd_ptr<nbl::video::IGPUCommandPool> commandPool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), commandPoolFlags);

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

			cpu2gpuParams.waitForCreationToComplete();

			core::smart_refctd_ptr<IGPUImage> image;
			for (uint32_t i = 0u; i < IMAGE_CNT; i++)
			{
				if(!(*gpu_image_array)[i])
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

			// Create histogram buffer.
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

			smart_refctd_ptr<nbl::video::IGPUSpecializedShader> specShader;
			{
				// Normally we'd use the ISystem and the IAssetManager to load shaders flexibly from (virtual) files for ease of development (syntax highlighting and Intellisense),
				// but I want to show the full process of assembling a shader from raw source code at least once.
				smart_refctd_ptr<nbl::asset::IShaderCompiler> compiler = make_smart_refctd_ptr<nbl::asset::CHLSLCompiler>(smart_refctd_ptr(m_system));

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
				commonImageTransitionBarrier.oldLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
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

			// TODO: should i use fence instead?
			// Wait for other the thread to load images.
			std::unique_lock lk(histogramBufferReadyMutex);
			histogramBufferReady.wait(lk, [this]() { return histogramBufferReadyFlag; });

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

			m_device->unmapMemory(histogramBufferAllocation.memory.get());
		}


};


NBL_MAIN_FUNC(StagingAndMultipleQueuesApp)
