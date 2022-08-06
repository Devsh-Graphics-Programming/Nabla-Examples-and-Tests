#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;

class ComputeShaderSampleApp : public ApplicationBase
{
	uint32_t windowWidth = 768u;
	uint32_t windowHeight = 512u;
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
	int32_t m_frameIx = 0;

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline = nullptr;
	core::vector<core::smart_refctd_ptr<video::IGPUDescriptorSet>> m_descriptorSets;
	core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_descriptorSetLayout;

	// These can be removed after descriptor lifetime tracking
	core::vector<core::smart_refctd_ptr<video::IGPUImageView>> m_swapchainImageViews;
	core::smart_refctd_ptr<video::IGPUImageView> m_inImageView;

	core::deque<CommonAPI::IRetiredSwapchainResources*> m_qRetiredSwapchainResources;
	std::mutex m_resizeLock;
	std::condition_variable m_resizeWaitForFrame;

	uint32_t m_swapchainIteration = 0;
	std::array<uint32_t, CommonAPI::InitOutput::MaxSwapChainImageCount> m_imageSwapchainIterations;
	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	struct CSwapchainResources : public CommonAPI::IRetiredSwapchainResources
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> oldImageView = nullptr;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> oldImage = nullptr;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> descriptorSet = nullptr;

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
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = img->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = img;

			m_swapchainImageViews[i] = logicalDevice->createImageView(std::move(viewParams));
			assert(m_swapchainImageViews[i]);
		}

		const uint32_t descriptorPoolSizeCount = 1u;
		video::IDescriptorPool::SDescriptorPoolSize poolSizes[descriptorPoolSizeCount];
		poolSizes[0].type = asset::EDT_STORAGE_IMAGE;
		poolSizes[0].count = 2u;

		video::IDescriptorPool::E_CREATE_FLAGS descriptorPoolFlags =
			static_cast<video::IDescriptorPool::E_CREATE_FLAGS>(0);

		core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool
			= logicalDevice->createDescriptorPool(descriptorPoolFlags, 1,
				descriptorPoolSizeCount, poolSizes);

		m_descriptorSets[i] = logicalDevice->createDescriptorSet(descriptorPool.get(),
			core::smart_refctd_ptr(m_descriptorSetLayout));

		const uint32_t writeDescriptorCount = 2u;

		video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[writeDescriptorCount];
		video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[writeDescriptorCount] = {};

		// image2D -- my input
		{
			descriptorInfos[0].image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[0].image.sampler = nullptr;
			descriptorInfos[0].desc = m_inImageView;

			writeDescriptorSets[0].dstSet = m_descriptorSets[i].get();
			writeDescriptorSets[0].binding = 1u;
			writeDescriptorSets[0].arrayElement = 0u;
			writeDescriptorSets[0].count = 1u;
			writeDescriptorSets[0].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[0].info = &descriptorInfos[0];
		}

		// image2D -- swapchain image
		{
			descriptorInfos[1].image.imageLayout = asset::IImage::EL_GENERAL;
			descriptorInfos[1].image.sampler = nullptr;
			descriptorInfos[1].desc = m_swapchainImageViews[i]; // shouldn't IGPUDescriptorSet hold a reference to the resources in its descriptors?

			writeDescriptorSets[1].dstSet = m_descriptorSets[i].get();
			writeDescriptorSets[1].binding = 0u;
			writeDescriptorSets[1].arrayElement = 0u;
			writeDescriptorSets[1].count = 1u;
			writeDescriptorSets[1].descriptorType = asset::EDT_STORAGE_IMAGE;
			writeDescriptorSets[1].info = &descriptorInfos[1];
		}

		logicalDevice->updateDescriptorSets(writeDescriptorCount, writeDescriptorSets, 0u, nullptr);
		m_imageSwapchainIterations[i] = m_swapchainIteration;
	}

	void onResize(uint32_t w, uint32_t h) override
	{
		std::unique_lock guard(m_resizeLock);
		windowWidth = w;
		windowHeight = h;
		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, w, h, swapchain);
		assert(swapchain);
		m_swapchainIteration++;
		m_resizeWaitForFrame.wait(guard);
	}

	void onAppInitialized_impl() override
	{
		const auto swapchainImageUsage = static_cast<asset::IImage::E_USAGE_FLAGS>(asset::IImage::EUF_COLOR_ATTACHMENT_BIT | asset::IImage::EUF_STORAGE_BIT);
		std::array<asset::E_FORMAT, 1> acceptableSurfaceFormats = { asset::EF_B8G8R8A8_UNORM };

		CommonAPI::InitParams initParams;
		initParams.window = core::smart_refctd_ptr(window);
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { "02.ComputeShader" };
		initParams.framesInFlight = FRAMES_IN_FLIGHT;
		initParams.windowWidth = windowWidth;
		initParams.windowHeight = windowHeight;
		initParams.swapchainImageCount = 3u;
		initParams.swapchainImageUsage = swapchainImageUsage;
		initParams.acceptableSurfaceFormats = acceptableSurfaceFormats.data();
		initParams.acceptableSurfaceFormatCount = acceptableSurfaceFormats.size();
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));
		initParams.windowCb->setApplication(this);

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

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, windowWidth, windowHeight, swapchain);
		assert(swapchain);

		commandPools = std::move(initOutput.commandPools);
		const auto& computeCommandPools = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

#if 0
		// Todo(achal): Pending bug investigation, when both API connections are created at
		// the same time
		core::smart_refctd_ptr<video::COpenGLConnection> api =
			video::COpenGLConnection::create(core::smart_refctd_ptr(system), 0, "02.ComputeShader", video::COpenGLDebugCallback(core::smart_refctd_ptr(logger)));

		core::smart_refctd_ptr<video::CSurfaceGLWin32> surface =
			video::CSurfaceGLWin32::create(core::smart_refctd_ptr(api),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
#endif

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		m_swapchainImages = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUImage>>>(swapchainImageCount);
		m_swapchainImageViews.resize(swapchainImageCount);

		video::IGPUObjectFromAssetConverter CPU2GPU;

		const char* pathToShader = "../compute.comp";
		core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
		{
			asset::IAssetLoader::SAssetLoadParams params = {};
			params.logger = logger.get();
			auto spec = (assetManager->getAsset(pathToShader, params).getContents());
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
			bindings[0].type = asset::EDT_STORAGE_IMAGE;
			bindings[0].count = 1u;
			bindings[0].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[0].samplers = nullptr;

			bindings[1].binding = 1u;
			bindings[1].type = asset::EDT_STORAGE_IMAGE;
			bindings[1].count = 1u;
			bindings[1].stageFlags = asset::IShader::ESS_COMPUTE;
			bindings[1].samplers = nullptr;
		}
		m_descriptorSetLayout =
			logicalDevice->createDescriptorSetLayout(bindings, bindings + bindingCount);

		// Todo(achal): Uncomment once the KTX loader works
#if 0
		constexpr auto cachingFlags = static_cast<asset::IAssetLoader::E_CACHING_FLAGS>(
			asset::IAssetLoader::ECF_DONT_CACHE_REFERENCES & asset::IAssetLoader::ECF_DONT_CACHE_TOP_LEVEL);

		const char* pathToImage = "../../media/GLI/kueken7_rgba8_unorm.ktx";

		asset::IAssetLoader::SAssetLoadParams loadParams(0ull, nullptr, cachingFlags);
		auto cpuImageBundle = assetManager->getAsset(pathToImage, loadParams);
		auto cpuImageContents = cpuImageBundle.getContents();
		if (cpuImageContents.empty())
		{
			logger->log("Failed to read image at path %s", nbl::system::ILogger::ELL_ERROR, pathToImage);
			exit(-1);
		}
		auto cpuImage = core::smart_refctd_ptr_static_cast<asset::ICPUImage>(*cpuImageContents.begin());
		__debugbreak();
#endif

		const uint32_t imageWidth = windowWidth;
		const uint32_t imageHeight = windowHeight;
		const uint32_t imageChannelCount = 4u;
		const uint32_t mipLevels = 1u; // WILL NOT WORK FOR MORE THAN 1 MIPS, but doesn't matter since it is temporary until KTX loading works
		const size_t imageSize = imageWidth * imageHeight * imageChannelCount * sizeof(uint8_t);
		auto imagePixels = core::make_smart_refctd_ptr<asset::ICPUBuffer>(imageSize);

		uint32_t* dstPixel = (uint32_t*)imagePixels->getPointer();
		for (uint32_t y = 0u; y < imageHeight; ++y)
		{
			for (uint32_t x = 0u; x < imageWidth; ++x)
			{
				// Should be red in R8G8B8A8_UNORM
				*dstPixel++ = 0x000000FF;
			}
		}

		core::smart_refctd_ptr<asset::ICPUImage> inImage_CPU = nullptr;
		{
			video::IGPUImage::SCreationParams creationParams = {};
			creationParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			creationParams.type = asset::IImage::ET_2D;
			creationParams.format = asset::EF_R8G8B8A8_UNORM;
			creationParams.extent = { imageWidth, imageHeight, 1u };
			creationParams.mipLevels = mipLevels;
			creationParams.arrayLayers = 1u;
			creationParams.samples = asset::IImage::ESCF_1_BIT;
			creationParams.tiling = video::IGPUImage::ET_OPTIMAL;
			if (apiConnection->getAPIType() == video::EAT_VULKAN ||
				apiConnection->getAPIType() == video::EAT_OPENGL_ES)
			{
				const auto& formatUsages = physicalDevice->getImageFormatUsagesOptimal(creationParams.format);
				assert(formatUsages.storageImage);
				assert(formatUsages.sampledImage);
				assert(asset::isFloatingPointFormat(creationParams.format) || asset::isNormalizedFormat(creationParams.format));
			}
			creationParams.usage = core::bitflag(asset::IImage::EUF_STORAGE_BIT) | asset::IImage::EUF_TRANSFER_DST_BIT;
			creationParams.queueFamilyIndexCount = 0u;
			creationParams.queueFamilyIndices = nullptr;
			creationParams.initialLayout = asset::IImage::EL_UNDEFINED;

			auto imageRegions = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<asset::ICPUImage::SBufferCopy>>(1ull);
			imageRegions->begin()->bufferOffset = 0ull;
			imageRegions->begin()->bufferRowLength = creationParams.extent.width;
			imageRegions->begin()->bufferImageHeight = 0u;
			imageRegions->begin()->imageSubresource = {};
			imageRegions->begin()->imageSubresource.aspectMask = asset::IImage::EAF_COLOR_BIT;
			imageRegions->begin()->imageSubresource.layerCount = 1u;
			imageRegions->begin()->imageOffset = { 0, 0, 0 };
			imageRegions->begin()->imageExtent = { creationParams.extent.width, creationParams.extent.height, 1u };

			inImage_CPU = asset::ICPUImage::create(std::move(creationParams));
			inImage_CPU->setBufferAndRegions(core::smart_refctd_ptr<asset::ICPUBuffer>(imagePixels), imageRegions);
		}

		cpu2gpuParams.beginCommandBuffers();
		auto inImage = CPU2GPU.getGPUObjectsFromAssets(&inImage_CPU, &inImage_CPU + 1, cpu2gpuParams);
		cpu2gpuParams.waitForCreationToComplete(false);
		assert(inImage);

		// Create an image view for input image
		{
			video::IGPUImageView::SCreationParams viewParams;
			viewParams.format = inImage_CPU->getCreationParameters().format;
			viewParams.viewType = asset::IImageView<video::IGPUImage>::ET_2D;
			viewParams.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			viewParams.subresourceRange.baseMipLevel = 0u;
			viewParams.subresourceRange.levelCount = 1u;
			viewParams.subresourceRange.baseArrayLayer = 0u;
			viewParams.subresourceRange.layerCount = 1u;
			viewParams.image = inImage->begin()[0];

			m_inImageView = logicalDevice->createImageView(std::move(viewParams));
		}
		assert(m_inImageView);

		m_descriptorSets.resize(swapchainImageCount);
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

	void onAppTerminated_impl() override
	{
		logicalDevice->waitIdle();
	}

	std::unique_lock<std::mutex> acquireImage(uint32_t* imgnum)
	{
		while (true)
		{
			std::unique_lock guard(m_resizeLock);
			if (swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, imgnum) == nbl::video::ISwapchain::EAIR_SUCCESS) {
				return guard;
			}
		}
	}

	void workLoopBody() override
	{
		m_resourceIx++;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& cb = m_cmdbuf[m_resourceIx];
		auto& commandPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE][m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];
		if (fence)
		{
			logicalDevice->blockForFences(1u, &fence.get());
			logicalDevice->resetFences(1u, &fence.get());
			if (m_frameIx >= FRAMES_IN_FLIGHT) CommonAPI::dropRetiredSwapchainResources(m_qRetiredSwapchainResources, m_frameIx - FRAMES_IN_FLIGHT);
		}
		else
			fence = logicalDevice->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		// acquire image 
		uint32_t imgnum = 0u;
		std::unique_lock guard = acquireImage(&imgnum);

		if (m_swapchainIteration > m_imageSwapchainIterations[imgnum])
		{
			CSwapchainResources* retiredResources(new CSwapchainResources{});
			retiredResources->oldImageView = m_swapchainImageViews[imgnum];
			retiredResources->oldImage = m_swapchainImages->begin()[imgnum];
			retiredResources->descriptorSet = m_descriptorSets[imgnum];
			retiredResources->retiredFrameId = m_frameIx;

			CommonAPI::retireSwapchainResources(m_qRetiredSwapchainResources, retiredResources);
			createSwapchainImage(imgnum);
		}

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

		video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarrier = {};
		layoutTransBarrier.srcQueueFamilyIndex = ~0u;
		layoutTransBarrier.dstQueueFamilyIndex = ~0u;
		layoutTransBarrier.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
		layoutTransBarrier.subresourceRange.baseMipLevel = 0u;
		layoutTransBarrier.subresourceRange.levelCount = 1u;
		layoutTransBarrier.subresourceRange.baseArrayLayer = 0u;
		layoutTransBarrier.subresourceRange.layerCount = 1u;

		const uint32_t pushConstants[3] = { windowWidth, windowHeight, swapchain->getPreTransform() };

		layoutTransBarrier.barrier.srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.barrier.dstAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier.oldLayout = asset::IImage::EL_UNDEFINED;
		layoutTransBarrier.newLayout = asset::IImage::EL_GENERAL;
		layoutTransBarrier.image = *(m_swapchainImages->begin() + imgnum);

		cb->pipelineBarrier(
			asset::EPSF_TOP_OF_PIPE_BIT,
			asset::EPSF_COMPUTE_SHADER_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier);

		const video::IGPUDescriptorSet* tmp[] = { m_descriptorSets[imgnum].get() };
		cb->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, tmp);

		cb->bindComputePipeline(m_pipeline.get());

		const asset::SPushConstantRange& pcRange = m_pipeline->getLayout()->getPushConstantRanges().begin()[0];
		cb->pushConstants(m_pipeline->getLayout(), pcRange.stageFlags, pcRange.offset, pcRange.size, pushConstants);

		cb->dispatch((windowWidth + 15u) / 16u, (windowHeight + 15u) / 16u, 1u);

		layoutTransBarrier.barrier.srcAccessMask = asset::EAF_SHADER_WRITE_BIT;
		layoutTransBarrier.barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0);
		layoutTransBarrier.oldLayout = asset::IImage::EL_GENERAL;
		layoutTransBarrier.newLayout = asset::IImage::EL_PRESENT_SRC;

		cb->pipelineBarrier(
			asset::EPSF_COMPUTE_SHADER_BIT,
			asset::EPSF_BOTTOM_OF_PIPE_BIT,
			static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
			0u, nullptr,
			0u, nullptr,
			1u, &layoutTransBarrier);

		cb->end();

		CommonAPI::Submit(
			logicalDevice.get(),
			cb.get(),
			queues[CommonAPI::InitOutput::EQT_COMPUTE],
			m_imageAcquire[m_resourceIx].get(),
			m_renderFinished[m_resourceIx].get(),
			fence.get());

		CommonAPI::Present(
			logicalDevice.get(),
			swapchain.get(),
			queues[CommonAPI::InitOutput::EQT_COMPUTE],
			m_renderFinished[m_resourceIx].get(),
			imgnum);
		m_frameIx++;
		m_resizeWaitForFrame.notify_all();
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