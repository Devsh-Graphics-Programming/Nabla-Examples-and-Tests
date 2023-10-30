// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <CommonAPI.h>
#include "nbl/ext/ScreenShot/ScreenShot.h"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include <iostream>
#include <cstdio>

using namespace nbl;

#define SWITCH_IMAGES_PER_X_MILISECONDS 750
constexpr std::string_view testingImagePathsFile = "../imagesTestList.txt";

struct NBL_CAPTION_DATA_TO_DISPLAY
{
	std::string viewType;
	std::string name;
	std::string extension;
};

class ColorSpaceTestSampleApp : public ApplicationBase
{
	constexpr static uint32_t WIN_W = 512u;
	constexpr static uint32_t WIN_H = 512u;
	constexpr static uint32_t SC_IMG_COUNT = 3u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;
	static_assert(FRAMES_IN_FLIGHT > SC_IMG_COUNT);

	core::smart_refctd_ptr<ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<video::ISurface> surface;
	core::smart_refctd_ptr<video::IUtilities> utilities;
	core::smart_refctd_ptr<video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<video::ISwapchain> swapchain;
	core::smart_refctd_ptr<video::IGPURenderpass> renderpass;
	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUFramebuffer>> fbos;
	std::array<std::array<core::smart_refctd_ptr<video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<system::ISystem> system;
	core::smart_refctd_ptr<asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	video::ISwapchain::SCreationParams m_swapchainCreationParams;

	uint32_t lastWidth = WIN_W;
	uint32_t lastHeight = WIN_H;

public:
	void setWindow(core::smart_refctd_ptr<ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<system::ISystem>&& system) override
	{
		system = std::move(system);
	}

	APP_CONSTRUCTOR(ColorSpaceTestSampleApp);

	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_TRANSFER_SRC_BIT);

		CommonAPI::InitParams initParams;
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { _NBL_APP_NAME_ };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = WIN_W;
		initParams.windowHeight = WIN_H;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		system = std::move(initOutput.system);
		window = std::move(initParams.window);
		windowManager = std::move(initOutput.windowManager);
		windowCb = std::move(initParams.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		renderpass = std::move(initOutput.renderToSwapchainRenderpass);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, WIN_W, WIN_H, swapchain);
		assert(swapchain);
		fbos = CommonAPI::createFBOWithSwapchainImages(
			swapchain->getImageCount(), WIN_W, WIN_H,
			logicalDevice, swapchain, renderpass,
			asset::EF_UNKNOWN
		);

		video::IGPUObjectFromAssetConverter cpu2gpu;

		auto createDescriptorPool = [&](const uint32_t textureCount)
		{
			constexpr uint32_t maxItemCount = 256u;
			{
				video::IDescriptorPool::SCreateInfo createInfo;
				createInfo.maxSets = maxItemCount;
				createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER)] = textureCount;
				return logicalDevice->createDescriptorPool(std::move(createInfo));
			}
		};

		asset::ISampler::SParams samplerParams = { asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETC_CLAMP_TO_EDGE, asset::ISampler::ETBC_FLOAT_OPAQUE_BLACK, asset::ISampler::ETF_LINEAR, asset::ISampler::ETF_LINEAR, asset::ISampler::ESMM_LINEAR, 0u, false, asset::ECO_ALWAYS };
		auto immutableSampler = logicalDevice->createSampler(samplerParams);

		video::IGPUDescriptorSetLayout::SBinding binding{ 0u, asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, video::IGPUShader::ESS_FRAGMENT, 1u, &immutableSampler };
		auto gpuDescriptorSetLayout3 = logicalDevice->createDescriptorSetLayout(&binding, &binding + 1u);
		auto gpuDescriptorPool = createDescriptorPool(1u); // per single texture
		auto fstProtoPipeline = ext::FullScreenTriangle::createProtoPipeline(cpu2gpuParams, 0u);

		auto createGPUPipeline = [&](asset::IImageView<asset::ICPUImage>::E_TYPE typeOfImage) -> core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>
		{
			auto getPathToFragmentShader = [&]() -> std::string
			{
				switch (typeOfImage)
				{
				case asset::IImageView<asset::ICPUImage>::ET_2D:
					return "../present2D.frag";
				case asset::IImageView<asset::ICPUImage>::ET_2D_ARRAY:
					return "../present2DArray.frag";
				case asset::IImageView<asset::ICPUImage>::ET_CUBE_MAP:
					return "../presentCubemap.frag";
				default:
					assert(false);
					return "";
				}
			};

			auto fs_bundle = assetManager->getAsset(getPathToFragmentShader(), {});
			auto fs_contents = fs_bundle.getContents();
			if (fs_contents.empty())
				assert(false);

			asset::ICPUSpecializedShader* cpuFragmentShader = static_cast<asset::ICPUSpecializedShader*>(fs_contents.begin()->get());

			core::smart_refctd_ptr<video::IGPUSpecializedShader> gpuFragmentShader;
			{
				cpu2gpuParams.beginCommandBuffers();
				auto gpu_array = cpu2gpu.getGPUObjectsFromAssets(&cpuFragmentShader, &cpuFragmentShader + 1, cpu2gpuParams);
				cpu2gpuParams.waitForCreationToComplete(false);

				if (!gpu_array.get() || gpu_array->size() < 1u || !(*gpu_array)[0])
					assert(false);

				gpuFragmentShader = (*gpu_array)[0];
			}

			auto constants = std::get<asset::SPushConstantRange>(fstProtoPipeline);
			auto gpuPipelineLayout = logicalDevice->createPipelineLayout(&constants, &constants + 1, nullptr, nullptr, nullptr, core::smart_refctd_ptr(gpuDescriptorSetLayout3));
			return ext::FullScreenTriangle::createRenderpassIndependentPipeline(logicalDevice.get(), fstProtoPipeline, std::move(gpuFragmentShader), std::move(gpuPipelineLayout));
		};

		auto gpuPipelineFor2D = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_2D);
		auto gpuPipelineFor2DArrays = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_2D_ARRAY);
		auto gpuPipelineForCubemaps = createGPUPipeline(asset::IImageView<asset::ICPUImage>::E_TYPE::ET_CUBE_MAP);

		core::vector<core::smart_refctd_ptr<asset::ICPUImageView>> cpuImageViews;
		core::vector<NBL_CAPTION_DATA_TO_DISPLAY> captionTexturesData;
		{
			std::ifstream list(testingImagePathsFile.data());
			if (list.is_open())
			{
				std::string line;
				for (; std::getline(list, line); )
				{
					if (line != "" && line[0] != ';')
					{
						auto& pathToTexture = line;
						auto& newCpuImageViewTexture = cpuImageViews.emplace_back();

						constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
						asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
						auto cpuTextureBundle = assetManager->getAsset(pathToTexture, loadParams);
						auto cpuTextureContents = cpuTextureBundle.getContents();
						{
							bool status = !cpuTextureContents.empty();
							assert(status);
						}

						if (cpuTextureContents.begin() == cpuTextureContents.end())
							assert(false); // cannot perform test in this scenario
						
						// Since this is ColorSpaceTest
						const asset::IImage::E_ASPECT_FLAGS aspectMask = asset::IImage::EAF_COLOR_BIT;
						auto asset = *cpuTextureContents.begin();
						switch (asset->getAssetType())
						{
						case asset::IAsset::ET_IMAGE:
						{
							asset::ICPUImageView::SCreationParams viewParams = {};
							viewParams.flags = static_cast<decltype(viewParams.flags)>(0u);
							viewParams.image = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(asset);
							viewParams.format = viewParams.image->getCreationParameters().format;
							viewParams.viewType = decltype(viewParams.viewType)::ET_2D;
							viewParams.subresourceRange.aspectMask = aspectMask;
							viewParams.subresourceRange.baseArrayLayer = 0u;
							viewParams.subresourceRange.layerCount = 1u;
							viewParams.subresourceRange.baseMipLevel = 0u;
							viewParams.subresourceRange.levelCount = 1u;

							newCpuImageViewTexture = asset::ICPUImageView::create(std::move(viewParams));
						} break;

						case asset::IAsset::ET_IMAGE_VIEW:
						{
							newCpuImageViewTexture = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(asset);
						} break;

						default:
						{
							assert(false); // in that case provided asset is wrong
						}
						}
						
						newCpuImageViewTexture->getCreationParameters().image->addImageUsageFlags(asset::IImage::EUF_SAMPLED_BIT);

						std::filesystem::path filename, extension;
						core::splitFilename(pathToTexture.c_str(), nullptr, &filename, &extension);

						auto& captionData = captionTexturesData.emplace_back();
						captionData.name = filename.string();
						captionData.extension = extension.string();
						captionData.viewType = [&]()
						{
							const auto& viewType = newCpuImageViewTexture->getCreationParameters().viewType;

							if (viewType == asset::IImageView<video::IGPUImage>::ET_2D)
								return std::string("ET_2D");
							else if (viewType == asset::IImageView<video::IGPUImage>::ET_2D_ARRAY)
								return std::string("ET_2D_ARRAY");
							else if (viewType == asset::IImageView<video::IGPUImage>::ET_CUBE_MAP)
								return std::string("ET_CUBE_MAP");
							else
								assert(false);
						}();

						const std::string finalFileNameWithExtension = captionData.name + captionData.extension;
						std::cout << finalFileNameWithExtension << "\n";

						auto tryToWrite = [&](asset::IAsset* asset)
						{
							asset::IAssetWriter::SAssetWriteParams wparams(asset);
							std::string assetPath = "imageAsset_" + finalFileNameWithExtension;
							return assetManager->writeAsset(assetPath, wparams);
						};

						if (!tryToWrite(newCpuImageViewTexture->getCreationParameters().image.get()))
							if (!tryToWrite(newCpuImageViewTexture.get()))
								assert(false); // could not write an asset
					}
				}
			}
		}
		
		// we clone because we need these cpuimages later for directly using upload utilitiy
		core::vector<core::smart_refctd_ptr<asset::ICPUImageView>> clonedCpuImageViews(cpuImageViews.size());
		for(uint32_t i = 0; i < cpuImageViews.size(); ++i)
			clonedCpuImageViews[i] = core::smart_refctd_ptr_static_cast<asset::ICPUImageView>(cpuImageViews[i]->clone());
		
		// Allocate and Leave 8MB for image uploads, to test image copy with small memory remaining 
		{
			uint32_t localOffset = video::StreamingTransientDataBufferMT<>::invalid_value;
			uint32_t maxFreeBlock = utilities->getDefaultUpStreamingBuffer()->max_size();
			const uint32_t allocationAlignment = 64u;
			const uint32_t allocationSize = maxFreeBlock - (0x00F0000u * 4u);
			utilities->getDefaultUpStreamingBuffer()->multi_allocate(std::chrono::steady_clock::now() + std::chrono::microseconds(500u), 1u, &localOffset, &allocationSize, &allocationAlignment);
		}

		cpu2gpuParams.beginCommandBuffers();
		auto gpuImageViews = cpu2gpu.getGPUObjectsFromAssets(clonedCpuImageViews.data(), clonedCpuImageViews.data() + clonedCpuImageViews.size(), cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		
		if (!gpuImageViews || gpuImageViews->size() < cpuImageViews.size())
			assert(false);

		// Creates GPUImageViews from Loaded CPUImageViews but this time use IUtilities::updateImageViaStagingBuffer directly and only copy sub-regions for testing purposes.
		core::vector<core::smart_refctd_ptr<video::IGPUImageView>> weirdGPUImages;
		{
			// Create GPU Images based on cpuImageViews
			for(uint32_t i = 0; i < cpuImageViews.size(); ++i)
			{
				auto& cpuImageView = cpuImageViews[i];
				auto imageviewCreateParams = cpuImageView->getCreationParameters();
				auto& cpuImage = imageviewCreateParams.image;
				auto imageCreateParams = cpuImage->getCreationParameters();
				
				auto regions = cpuImage->getRegions();
				std::vector<asset::IImage::SBufferCopy> newRegions;
				// Make new regions weird
				for(uint32_t r = 0; r < regions.size(); ++r)
				{
					auto & region = regions[0];
					
					const auto quarterWidth = core::max(region.imageExtent.width / 4, 1u);
					const auto quarterHeight = core::max(region.imageExtent.height / 4, 1u);
					const auto texelBlockInfo = asset::TexelBlockInfo(imageCreateParams.format);
					const auto imageExtentsInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(region.imageExtent.width, region.imageExtent.height, region.imageExtent.depth));

					// Pattern we're trying to achieve (Copy only the regions marked by X)
					// +----+----+
					// |  xx|    |
					// +----+----+
					// |    |xx  |
					// +----+----+
					{
						asset::IImage::SBufferCopy newRegion = region;
						newRegion.imageExtent.width = quarterWidth;
						newRegion.imageExtent.height = quarterHeight;
						newRegion.imageExtent.depth = region.imageExtent.depth;
						newRegion.imageOffset.x = quarterWidth;
						newRegion.imageOffset.y = quarterHeight;
						newRegion.imageOffset.z = 0u;
						auto offsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(newRegion.imageOffset.x, newRegion.imageOffset.y, newRegion.imageOffset.z));
						newRegion.bufferOffset =  (offsetInBlocks.y * imageExtentsInBlocks.x + offsetInBlocks.x) * texelBlockInfo.getBlockByteSize();
						newRegion.bufferRowLength = region.imageExtent.width;
						newRegion.bufferImageHeight = region.imageExtent.height;
						newRegions.push_back(newRegion);
					}
					{
						asset::IImage::SBufferCopy newRegion = region;
						newRegion.imageExtent.width = quarterWidth;
						newRegion.imageExtent.height = quarterHeight;
						newRegion.imageExtent.depth = 1u;
						newRegion.imageOffset.x = quarterWidth * 2;
						newRegion.imageOffset.y = quarterHeight * 2;
						newRegion.imageOffset.z = 0u;
						auto offsetInBlocks = texelBlockInfo.convertTexelsToBlocks(core::vector3du32_SIMD(newRegion.imageOffset.x, newRegion.imageOffset.y, newRegion.imageOffset.z));
						newRegion.bufferOffset =  (offsetInBlocks.y * imageExtentsInBlocks.x + offsetInBlocks.x) * texelBlockInfo.getBlockByteSize();
						newRegion.bufferRowLength = region.imageExtent.width;
						newRegion.bufferImageHeight = region.imageExtent.height;
						newRegions.push_back(newRegion);
					}
				}
				
				video::IPhysicalDevice::SImageFormatPromotionRequest promotionRequest = {};
				promotionRequest.originalFormat = imageCreateParams.format;
				promotionRequest.usages = imageCreateParams.usage | asset::IImage::EUF_TRANSFER_DST_BIT;
				auto newFormat = physicalDevice->promoteImageFormat(promotionRequest, video::IGPUImage::ET_OPTIMAL);

				video::IGPUImage::SCreationParams gpuImageCreateInfo = {};
				gpuImageCreateInfo.flags = imageCreateParams.flags;
				gpuImageCreateInfo.type = imageCreateParams.type;
				gpuImageCreateInfo.format = newFormat;
				gpuImageCreateInfo.extent = imageCreateParams.extent;
				gpuImageCreateInfo.mipLevels = imageCreateParams.mipLevels;
				gpuImageCreateInfo.arrayLayers = imageCreateParams.arrayLayers;
				gpuImageCreateInfo.samples = imageCreateParams.samples;
				gpuImageCreateInfo.tiling = video::IGPUImage::ET_OPTIMAL;
				gpuImageCreateInfo.usage = imageCreateParams.usage | asset::IImage::EUF_TRANSFER_DST_BIT;
				gpuImageCreateInfo.queueFamilyIndexCount = 0u;
				gpuImageCreateInfo.queueFamilyIndices = nullptr;
				gpuImageCreateInfo.initialLayout = asset::IImage::EL_UNDEFINED;
				auto gpuImage = logicalDevice->createImage(std::move(gpuImageCreateInfo));
				
				auto gpuImageMemReqs = gpuImage->getMemoryReqs();
				gpuImageMemReqs.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
				logicalDevice->allocate(gpuImageMemReqs, gpuImage.get(), video::IDeviceMemoryAllocation::EMAF_NONE);

				video::IGPUImageView::SCreationParams gpuImageViewCreateInfo = {};
				gpuImageViewCreateInfo.flags = static_cast<video::IGPUImageView::E_CREATE_FLAGS>(imageviewCreateParams.flags);
				gpuImageViewCreateInfo.image = gpuImage;
				gpuImageViewCreateInfo.viewType = static_cast<video::IGPUImageView::E_TYPE>(imageviewCreateParams.viewType);
				gpuImageViewCreateInfo.format = gpuImageCreateInfo.format;
				memcpy(&gpuImageViewCreateInfo.components, &imageviewCreateParams.components, sizeof(imageviewCreateParams.components));
				gpuImageViewCreateInfo.subresourceRange = imageviewCreateParams.subresourceRange;
				gpuImageViewCreateInfo.subresourceRange.levelCount = imageCreateParams.mipLevels - imageviewCreateParams.subresourceRange.baseMipLevel;
				
				auto gpuImageView = logicalDevice->createImageView(std::move(gpuImageViewCreateInfo));

				weirdGPUImages.push_back(gpuImageView);

				auto& transferCommandPools = commandPools[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				auto transferQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_UP];
				core::smart_refctd_ptr<video::IGPUCommandBuffer> transferCmd;
				logicalDevice->createCommandBuffers(transferCommandPools[0u].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1u, &transferCmd);
				
				auto transferFence = logicalDevice->createFence(video::IGPUFence::ECF_UNSIGNALED);

				transferCmd->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
				
				video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransition = {};
				layoutTransition.barrier.srcAccessMask = asset::EAF_NONE;
				layoutTransition.barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
				layoutTransition.oldLayout = asset::IImage::EL_UNDEFINED;
				layoutTransition.newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
				layoutTransition.srcQueueFamilyIndex = ~0u;
				layoutTransition.dstQueueFamilyIndex = ~0u;
				layoutTransition.image = gpuImageView->getCreationParameters().image;
				layoutTransition.subresourceRange = gpuImageView->getCreationParameters().subresourceRange;
				transferCmd->pipelineBarrier(asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EPSF_TRANSFER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &layoutTransition);
				
				video::IGPUQueue::SSubmitInfo submit = {};
				submit.commandBufferCount = 1u;
				submit.commandBuffers = &transferCmd.get();
				submit.waitSemaphoreCount = 0u;
				submit.pWaitSemaphores = nullptr;
				submit.pWaitDstStageMask = nullptr;
				core::SRange<const asset::IImage::SBufferCopy> copyRegions(newRegions.data(), newRegions.data() + newRegions.size());
				
				utilities->updateImageViaStagingBufferAutoSubmit( cpuImage->getBuffer(), imageCreateParams.format, gpuImage.get(), asset::IImage::EL_TRANSFER_DST_OPTIMAL, copyRegions, transferQueue, transferFence.get(), submit);
				// transferCmd->end();

				logicalDevice->blockForFences(1u, &transferFence.get());
			}
		}

		auto getCurrentGPURenderpassIndependentPipeline = [&](video::IGPUImageView* gpuImageView)
		{
			switch (gpuImageView->getCreationParameters().viewType)
			{
			case asset::IImageView<video::IGPUImage>::ET_2D:
			{
				return gpuPipelineFor2D;
			}

			case asset::IImageView<video::IGPUImage>::ET_2D_ARRAY:
			{
				return gpuPipelineFor2DArrays;
			}

			case asset::IImageView<video::IGPUImage>::ET_CUBE_MAP:
			{
				return gpuPipelineForCubemaps;
			}

			default:
				assert(false);
			}
		};

		auto ds = gpuDescriptorPool->createDescriptorSet(core::smart_refctd_ptr(gpuDescriptorSetLayout3));

		auto presentImageOnTheScreen = [&](core::smart_refctd_ptr<video::IGPUImageView> gpuImageView, const NBL_CAPTION_DATA_TO_DISPLAY& captionData)
		{
			auto windowExtent = gpuImageView->getCreationParameters().image->getCreationParameters().extent;

			bool didResize = false;
			if (windowExtent.width != lastWidth || windowExtent.height != lastHeight)
			{
				didResize = windowManager->setWindowSize(window.get(), windowExtent.width, windowExtent.height);
				assert(didResize);
			}
			// can't just use windowExtent as the actual window size may have been capped by windows
			VkExtent3D imgExtents = { window->getWidth(), window->getHeight(), 1 };

			if (didResize)
			{
				CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, imgExtents.width, imgExtents.height, swapchain);
				assert(swapchain);
				fbos = CommonAPI::createFBOWithSwapchainImages(
					swapchain->getImageCount(), imgExtents.width, imgExtents.height,
					logicalDevice, swapchain, renderpass,
					asset::EF_UNKNOWN
				);

				lastWidth = imgExtents.width;
				lastHeight = imgExtents.height;
			}

			video::IGPUDescriptorSet::SDescriptorInfo info;
			{
				info.desc = gpuImageView;
				info.info.image.sampler = nullptr;
				info.info.image.imageLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
			}

			video::IGPUDescriptorSet::SWriteDescriptorSet write;
			write.dstSet = ds.get();
			write.binding = 0u;
			write.arrayElement = 0u;
			write.count = 1u;
			write.descriptorType = asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER;
			write.info = &info;

			logicalDevice->updateDescriptorSets(1u, &write, 0u, nullptr);

			auto currentGpuRenderpassIndependentPipeline = getCurrentGPURenderpassIndependentPipeline(gpuImageView.get());
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> gpuGraphicsPipeline;
			{
				video::IGPUGraphicsPipeline::SCreationParams graphicsPipelineParams = {};
				graphicsPipelineParams.renderpassIndependent = core::smart_refctd_ptr<video::IGPURenderpassIndependentPipeline>(const_cast<video::IGPURenderpassIndependentPipeline*>(currentGpuRenderpassIndependentPipeline.get()));
				graphicsPipelineParams.renderpass = core::smart_refctd_ptr(renderpass);

				gpuGraphicsPipeline = logicalDevice->createGraphicsPipeline(nullptr, std::move(graphicsPipelineParams));
			}

			const std::string windowCaption = "[Nabla Engine] Color Space Test Demo - CURRENT IMAGE: " + captionData.name + " - VIEW TYPE: " + captionData.viewType + " - EXTENSION: " + captionData.extension;
			window->setCaption(windowCaption);

			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FRAMES_IN_FLIGHT];

			core::smart_refctd_ptr<video::IGPUFence> frameComplete[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
			core::smart_refctd_ptr<video::IGPUSemaphore> renderFinished[FRAMES_IN_FLIGHT] = { nullptr };

			const auto& graphicsCommandPools = commandPools[CommonAPI::InitOutput::EQT_GRAPHICS];
			for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
			{
				logicalDevice->createCommandBuffers(graphicsCommandPools[i].get(), video::IGPUCommandBuffer::EL_PRIMARY, 1, commandBuffers+i);
				imageAcquire[i] = logicalDevice->createSemaphore();
				renderFinished[i] = logicalDevice->createSemaphore();
			}

			auto startPoint = std::chrono::high_resolution_clock::now();

			uint32_t imgnum = 0u;
			int32_t resourceIx = -1;
			for (;;)
			{
				resourceIx++;
				if (resourceIx >= FRAMES_IN_FLIGHT)
					resourceIx = 0;

				auto& cb = commandBuffers[resourceIx];
				auto& fence = frameComplete[resourceIx];
				if (fence)
				{
					while (logicalDevice->waitForFences(1u, &fence.get(), false, MAX_TIMEOUT) == video::IGPUFence::ES_TIMEOUT)
					{
					}
					logicalDevice->resetFences(1u, &fence.get());
				}
				else
					fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

				auto aPoint = std::chrono::high_resolution_clock::now();
				if (std::chrono::duration_cast<std::chrono::milliseconds>(aPoint - startPoint).count() > SWITCH_IMAGES_PER_X_MILISECONDS)
					break;

				// acquire image 
				swapchain->acquireNextImage(MAX_TIMEOUT, imageAcquire[resourceIx].get(), nullptr, &imgnum);

				cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);  // TODO: Reset Frame's CommandPool


				video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransition = {};
				layoutTransition.barrier.srcAccessMask = asset::EAF_NONE;
				layoutTransition.barrier.dstAccessMask = asset::EAF_SHADER_READ_BIT;
				layoutTransition.oldLayout = asset::IImage::EL_UNDEFINED;
				layoutTransition.newLayout = asset::IImage::EL_SHADER_READ_ONLY_OPTIMAL;
				layoutTransition.srcQueueFamilyIndex = ~0u;
				layoutTransition.dstQueueFamilyIndex = ~0u;
				layoutTransition.image = gpuImageView->getCreationParameters().image;
				layoutTransition.subresourceRange = gpuImageView->getCreationParameters().subresourceRange;

				cb->pipelineBarrier(asset::EPSF_BOTTOM_OF_PIPE_BIT, asset::EPSF_COMPUTE_SHADER_BIT, asset::EDF_NONE, 0u, nullptr, 0u, nullptr, 1u, &layoutTransition);

				asset::SViewport viewport;
				viewport.minDepth = 1.f;
				viewport.maxDepth = 0.f;
				viewport.x = 0u;
				viewport.y = 0u;
				viewport.width = imgExtents.width;
				viewport.height = imgExtents.height;
				cb->setViewport(0u, 1u, &viewport);

				VkRect2D scissor;
				scissor.offset = { 0, 0 };
				scissor.extent = { imgExtents.width, imgExtents.height };

				cb->setScissor(0u, 1u, &scissor);

				video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo;
				{
					VkRect2D area;
					area.offset = { 0,0 };
					area.extent = { imgExtents.width, imgExtents.height };
					asset::SClearValue clear;
					clear.color.float32[0] = 1.f;
					clear.color.float32[1] = 1.f;
					clear.color.float32[2] = 1.f;
					clear.color.float32[3] = 1.f;
					beginInfo.clearValueCount = 1u;
					beginInfo.framebuffer = fbos->begin()[imgnum];
					beginInfo.renderpass = renderpass;
					beginInfo.renderArea = area;
					beginInfo.clearValues = &clear;
				}

				cb->beginRenderPass(&beginInfo, asset::ESC_INLINE);
				cb->bindGraphicsPipeline(gpuGraphicsPipeline.get());
				cb->bindDescriptorSets(asset::EPBP_GRAPHICS, gpuGraphicsPipeline->getRenderpassIndependentPipeline()->getLayout(), 3, 1, &ds.get());
				ext::FullScreenTriangle::recordDrawCalls(gpuGraphicsPipeline, 0u, swapchain->getPreTransform(), cb.get());
				cb->endRenderPass();
				cb->end();

				CommonAPI::Submit(
					logicalDevice.get(),
					cb.get(),
					queues[CommonAPI::InitOutput::EQT_GRAPHICS],
					imageAcquire[resourceIx].get(),
					renderFinished[resourceIx].get(),
					fence.get());

				CommonAPI::Present(
					logicalDevice.get(),
					swapchain.get(),
					queues[CommonAPI::InitOutput::EQT_GRAPHICS],
					renderFinished[resourceIx].get(),
					imgnum);
			}

			logicalDevice->waitIdle();

			const auto& fboCreationParams = fbos->begin()[imgnum]->getCreationParameters();
			auto gpuSourceImageView = fboCreationParams.attachments[0];

			const std::string writePath = "screenShot_" + captionData.name + ".png";

			return ext::ScreenShot::createScreenShot(
				logicalDevice.get(),
				queues[decltype(initOutput)::EQT_TRANSFER_UP],
				nullptr,
				gpuSourceImageView.get(),
				assetManager.get(),
				writePath,
				asset::IImage::EL_PRESENT_SRC,
				asset::EAF_NONE);
		};

		for (size_t i = 0; i < gpuImageViews->size(); ++i)
		{
			auto gpuImageView = (*gpuImageViews)[i];
			if (gpuImageView)
			{
				auto& captionData = captionTexturesData[i];
		
				bool status = presentImageOnTheScreen(core::smart_refctd_ptr(gpuImageView), captionData);
				assert(status);
			}
		}

		// Now present weird images (sub-region copies)
		for (size_t i = 0; i < weirdGPUImages.size(); ++i)
		{
			auto gpuImageView = weirdGPUImages[i];
			if (gpuImageView)
			{
				NBL_CAPTION_DATA_TO_DISPLAY captionData = {};
				captionData.name = "Weird Region";
				bool status = presentImageOnTheScreen(core::smart_refctd_ptr(gpuImageView), captionData);
				assert(status);
			}
		}
	}

	void onAppTerminated_impl() override
	{

	}

	void workLoopBody() override
	{
		
	}

	bool keepRunning() override
	{
		return false;
	}
	
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return logicalDevice.get();
	}
	video::IGPURenderpass* getRenderpass() override
	{
		return renderpass.get();
	}
	void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override
	{
		surface = std::move(s);
	}
	void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override
	{
		for (int i = 0; i < f.size(); i++)
		{
			fbos->begin()[i] = core::smart_refctd_ptr(f[i]);
		}
	}
	void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override
	{
		swapchain = std::move(s);
	}
	uint32_t getSwapchainImageCount() override
	{
		return swapchain->getImageCount();
	}
	virtual asset::E_FORMAT getDepthFormat() override
	{
		return asset::EF_D32_SFLOAT;
	}
};

NBL_COMMON_API_MAIN(ColorSpaceTestSampleApp)

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }