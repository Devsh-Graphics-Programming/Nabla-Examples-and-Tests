#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;

class ComputeShaderSampleApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;


	core::smart_refctd_ptr<nbl::ui::IWindowManager> windowManager;
	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
	video::IPhysicalDevice* physicalDevice;
	std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>, CommonAPI::InitOutput::MaxSwapChainImageCount> fbo;
	std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
	video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;
	core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;
	video::IGPUObjectFromAssetConverter cpu2gpu;
	core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUImage>> m_swapchainImages;

	int32_t m_resourceIx = -1;

	std::array<core::smart_refctd_ptr<video::IGPUDescriptorSet>, 2> m_outputTargetDescriptorSet;
	std::array<core::smart_refctd_ptr<video::IGPUImageView>, 2> m_outputTargetImageView;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline = nullptr;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_descriptorSetLayout;

	core::smart_refctd_ptr<video::IGPUImageView> m_inImageView;

	bool hasPresentedWithBlit = false;
	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	struct CSwapchainResources : public CommonAPI::IRetiredSwapchainResources
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> oldImage = nullptr;

		~CSwapchainResources() override {}
	};

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override
	{
		system = std::move(system);
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
			fbo[i] = core::smart_refctd_ptr(f[i]);
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
	virtual nbl::asset::E_FORMAT getDepthFormat() override
	{
		return nbl::asset::EF_D32_SFLOAT;
	}

	APP_CONSTRUCTOR(ComputeShaderSampleApp);

	void createSwapchainImage(uint32_t i)
	{
		auto& img = m_swapchainImages->begin()[i];
		img = swapchain->createImage(i);
		assert(img);
	}

	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT | asset::IImage::EUF_TRANSFER_DST_BIT);
		std::array<asset::E_FORMAT, 1> acceptableSurfaceFormats = { asset::EF_B8G8R8A8_UNORM };

		CommonAPI::InitParams initParams;
		initParams.windowCb = core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(this);
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { _NBL_APP_NAME_ };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = 768u;
		initParams.windowHeight = 512u;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		system = std::move(initOutput.system);
		window = std::move(initParams.window);
		windowCb = std::move(initParams.windowCb);
		apiConnection = std::move(initOutput.apiConnection);
		surface = std::move(initOutput.surface);
		physicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		utilities = std::move(initOutput.utilities);
		queues = std::move(initOutput.queues);
		assetManager = std::move(initOutput.assetManager);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		windowManager = std::move(initOutput.windowManager);
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, initParams.windowWidth, initParams.windowHeight, swapchain);
		assert(swapchain);

		commandPools = std::move(initOutput.commandPools);
		const auto& computeCommandPools = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		m_swapchainImages = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUImage>>>(swapchainImageCount);

		video::IGPUObjectFromAssetConverter CPU2GPU;

		const char* pathToShader = "../compute.hlsl";
		core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*assetManager->getAsset(pathToShader, params).getContents().begin());
			specializedShader = CPU2GPU.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, cpu2gpuParams)->front();
		}
		assert(specializedShader);


		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			logicalDevice->createCommandBuffers(
				computeCommandPools[i].get(),
				video::IGPUCommandBuffer::EL_PRIMARY,
				1,
				m_cmdbuf + i);
		}


		const uint32_t bindingCount = 2u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount];
		{
			bindings[0].binding = 0u;
			bindings[0].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;

			bindings[1].binding = 1u;
			bindings[1].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;
		}
		m_descriptorSetLayout = logicalDevice->createDescriptorSetLayout(bindings, bindings + bindingCount);

		const char* pathToImage = "../../media/GLI/kueken7_rgba8_unorm.ktx";

		constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES | asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);
		asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
		auto cpuImageBundle = assetManager->getAsset(pathToImage, loadParams);
		auto cpuImageContents = cpuImageBundle.getContents();
		if (cpuImageContents.empty())
		{
			logger->log("Failed to read image at path %s", nbl::system::ILogger::ELL_ERROR, pathToImage);
			exit(-1);
		}
		auto cpuImage = asset::IAsset::castDown<asset::ICPUImageView>(*cpuImageContents.begin());
		// fix up usage flags to not get validation errors (TODO: Remove when Asset Converter 2.0 comes)
		cpuImage->getCreationParameters().image->addImageUsageFlags(asset::IImage::EUF_STORAGE_BIT);

		cpu2gpuParams.beginCommandBuffers();
		auto inImage = CPU2GPU.getGPUObjectsFromAssets(&cpuImage, &cpuImage+1, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		assert(inImage);
		m_inImageView = inImage->begin()[0];

		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			createSwapchainImage(i);
		}

		asset::SPushConstantRange pcRange = {};
		pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
		pcRange.offset = 0u;
		pcRange.size = 3 * sizeof(uint32_t);
		core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
			logicalDevice->createPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(m_descriptorSetLayout));

		m_pipeline = logicalDevice->createComputePipeline(nullptr,
			core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(specializedShader));

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; i++)
		{
			m_imageAcquire[i] = logicalDevice->createSemaphore();
			m_renderFinished[i] = logicalDevice->createSemaphore();
		}
	}

	std::unique_ptr<CommonAPI::IRetiredSwapchainResources> onCreateResourcesWithSwapchain(const uint32_t imgnum)
	{
		logger->log("onCreateResourcesWithSwapchain(%i)", system::ILogger::ELL_INFO, imgnum);
		CSwapchainResources* retiredResources(new CSwapchainResources{});
		retiredResources->oldImage = m_swapchainImages->begin()[imgnum];
		retiredResources->retiredFrameId = m_frameIx;
		createSwapchainImage(imgnum);

		return std::unique_ptr<CommonAPI::IRetiredSwapchainResources>(retiredResources);
	}

	void onCreateResourcesWithTripleBufferTarget(nbl::core::smart_refctd_ptr<nbl::video::IGPUImage>& image, uint32_t bufferIx)
	{
		// TODO: figure out better way of handling triple buffer target resources
		logger->log("onCreateResourcesWithTripleBufferTarget(%i) || %ix%i", system::ILogger::ELL_INFO, bufferIx, image->getCreationParameters().extent.width, image->getCreationParameters().extent.height);
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = image->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = image;

			m_outputTargetImageView[bufferIx] = logicalDevice->createImageView(std::move(viewParams));
			assert(m_outputTargetImageView[bufferIx]);
		}

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = nullptr;
		{
			video::IDescriptorPool::SCreateInfo createInfo = {};
			createInfo.maxSets = 1;
			createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = 2;

			descriptorPool = logicalDevice->createDescriptorPool(std::move(createInfo));
		}

		m_outputTargetDescriptorSet[bufferIx] = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(m_descriptorSetLayout));

		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		// image2D -- my input
		{
			descriptorInfos[0].info.image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[0].info.image.sampler = nullptr;
			descriptorInfos[0].desc = m_inImageView;

			writeDescriptorSets[0].dstSet = m_outputTargetDescriptorSet[bufferIx].get();
			writeDescriptorSets[0].binding = 1u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		// image2D -- swapchain image
		{
			descriptorInfos[1].info.image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[1].info.image.sampler = nullptr;
			descriptorInfos[1].desc = m_outputTargetImageView[bufferIx]; // shouldn't IGPUDescriptorSet hold a reference to the resources in its descriptors?

			writeDescriptorSets[1].dstSet = m_outputTargetDescriptorSet[bufferIx].get();
			writeDescriptorSets[1].binding = 0u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
			writeDescriptorSets[1].descriptorType = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			writeDescriptorSets[1].info = &descriptorInfos[1];
		}

		logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
	}

	bool onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		auto guard = recreateSwapchain(w, h, m_swapchainCreationParams, swapchain);
		PresentedFrameInfo frame = getLastPresentedFrame();
		logger->log("acquired guard onWindowResized_impl(%i, %i) -- last presented frame: %i", system::ILogger::ELL_INFO, w, h, frame.frameIx);
		waitForFrame(FRAMES_IN_FLIGHT, m_frameComplete[frame.resourceIx]);
		immediateImagePresent(
			queues[CommonAPI::InitOutput::EQT_COMPUTE], 
			swapchain.get(),
			m_swapchainImages->begin(), 
			frame.frameIx,
			frame.width, frame.height);
		hasPresentedWithBlit = true;
		logger->log("Done immediateImagePresent", system::ILogger::ELL_INFO);

		return true;
	}

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		waitForFrame(FRAMES_IN_FLIGHT, fence);
		logicalDevice->resetFences(1u, &fence.get());

		uint32_t imgnum = 0u;
		core::smart_refctd_ptr<video::ISwapchain> sw = acquire(swapchain, m_imageAcquire[m_resourceIx].get(), &imgnum);
		// these are safe to access now that we went through the acquire
		core::smart_refctd_ptr<video::IGPUImage> swapchainImg = *(m_swapchainImages->begin() + imgnum);
		uint32_t windowWidth = sw->getCreationParameters().width;
		uint32_t windowHeight = sw->getCreationParameters().height;
		video::IGPUImage* outputImage = getTripleBufferTarget(m_frameIx, windowWidth, windowHeight, sw->getCreationParameters().surfaceFormat.format, sw->getCreationParameters().imageUsage);

		// safe to proceed
		cb->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT); // TODO: Begin doesn't release the resources in the command pool, meaning the old swapchains never get dropped
		cb->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT); // TODO: Reset Frame's CommandPool

		{
			asset::SViewport vp;
			vp.minDepth = 1.f;
			vp.maxDepth = 0.f;
			vp.x = 0u;
			vp.y = 0u;
			vp.width = windowWidth;
			vp.height = windowHeight;
			cb->setViewport(0u, 1u, &vp);

			VkRect2D scissor;
			scissor.extent = { windowWidth, windowHeight };
			scissor.offset = { 0, 0 };
			cb->setScissor(0u, 1u, &scissor);
		}

		const uint32_t numBarriers = 2;
		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier[numBarriers] = {};
		for (uint32_t i = 0; i < numBarriers; i++) {
			layoutTransBarrier[i].srcQueueFamilyIndex = ~0u;
			layoutTransBarrier[i].dstQueueFamilyIndex = ~0u;
			layoutTransBarrier[i].subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			layoutTransBarrier[i].subresourceRange.baseMipLevel = 0u;
			layoutTransBarrier[i].subresourceRange.levelCount = 1u;
			layoutTransBarrier[i].subresourceRange.baseArrayLayer = 0u;
			layoutTransBarrier[i].subresourceRange.layerCount = 1u;
		}

		layoutTransBarrier[1].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier[1].barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier[1].oldLayout = asset::IImage::EL_UNDEFINED;
		layoutTransBarrier[1].newLayout = asset::IImage::EL_GENERAL;
		layoutTransBarrier[1].image = core::smart_refctd_ptr<video::IGPUImage>(outputImage);

		cb->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier[1]);

		const uint32_t pushConstants[3] = { windowWidth, windowHeight, sw->getPreTransform() };

		const video::IGPUDescriptorSet* tmp[] = { m_outputTargetDescriptorSet[m_frameIx % 2].get() };
		cb->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, tmp);

		cb->bindComputePipeline(m_pipeline.get());

		const asset::SPushConstantRange& pcRange = m_pipeline->getLayout()->getPushConstantRanges().begin()[0];
		cb->pushConstants(m_pipeline->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, pushConstants);

		cb->dispatch((windowWidth + 15u) / 16u, (windowHeight + 15u) / 16u, 1u);

		layoutTransBarrier[0].barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier[0].barrier.dstAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		layoutTransBarrier[0].oldLayout = asset::IImage::EL_UNDEFINED;
		layoutTransBarrier[0].newLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		layoutTransBarrier[0].image = swapchainImg;

		layoutTransBarrier[1].barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier[1].barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
		layoutTransBarrier[1].oldLayout = asset::IImage::EL_GENERAL;
		layoutTransBarrier[1].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;

		cb->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_TRANSFER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			numBarriers, &layoutTransBarrier[0]);

		nbl::asset::SImageBlit blit;
		blit.srcSubresource.aspectMask = nbl::video::IGPUImage::EAF_COLOR_BIT;
		blit.srcSubresource.layerCount = 1;
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { windowWidth, windowHeight, 1 };
		blit.dstSubresource.aspectMask = nbl::video::IGPUImage::EAF_COLOR_BIT;
		blit.dstSubresource.layerCount = 1;
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { windowWidth, windowHeight, 1 };

		// TODO this causes performance warnings, make image source use TRANSFER_SRC and swapchain image use TRANSFER_DST
		cb->blitImage(
			outputImage, nbl::asset::IImage::EL_TRANSFER_SRC_OPTIMAL,
			swapchainImg.get(), nbl::asset::IImage::EL_TRANSFER_DST_OPTIMAL,
			1, &blit, nbl::asset::ISampler::ETF_NEAREST
		);

		layoutTransBarrier[0].barrier.srcAccessMask = asset::EAF_TRANSFER_WRITE_BIT;
		layoutTransBarrier[0].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier[0].oldLayout = asset::IImage::EL_TRANSFER_DST_OPTIMAL;
		layoutTransBarrier[0].newLayout = asset::IImage::EL_PRESENT_SRC;

		layoutTransBarrier[1].barrier.srcAccessMask = asset::EAF_TRANSFER_READ_BIT;
		layoutTransBarrier[1].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier[1].oldLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
		layoutTransBarrier[1].newLayout = asset::IImage::EL_GENERAL;

		cb->pipelineBarrier(
			asset::EPSF_TRANSFER_BIT,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			numBarriers, &layoutTransBarrier[0]);

		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			cb.get(),
			queues[CommonAPI::InitOutput::EQT_COMPUTE],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		// Resize-blit will only happen with frames that have submitted with
		// this new resolution, which is what we want
		PresentedFrameInfo frame;
		frame.resourceIx = m_resourceIx;
		frame.frameIx = m_frameIx;
		frame.width = windowWidth;
		frame.height = windowHeight;
		setLastPresentedFrame(frame);

		{
			// Hold the lock here even though this is potentially the old swapchain, as presenting to
			// an old and new swapchain at the same time, or presenting to old swapchain after new one
			// causes a crash (despite what the spec says)
			std::unique_lock guard(m_swapchainPtrMutex);
			if (!hasPresentedWithBlit)
			{
				CommonAPI::Present(
					logicalDevice.get(),
					sw.get(),
					queues[CommonAPI::InitOutput::EQT_COMPUTE],
					m_renderFinished[m_resourceIx].get(),
					imgnum);
			}
			// TODO path where we don't call present has m_renderFinished be unused, causing validation error as well
			hasPresentedWithBlit = false;
		}
		m_frameIx++;
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

//NBL_COMMON_API_MAIN(ComputeShaderSampleApp)
int main(int argc, char** argv) {
	CommonAPI::main<ComputeShaderSampleApp>(argc, argv);
}

extern "C" {  _declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001; }