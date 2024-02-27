#define _NBL_STATIC_LIB_
#include <nabla.h>

#include "../common/CommonAPI.h"

using namespace nbl;

class ComputeShaderSampleApp : public ApplicationBase
{
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;


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


	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };


	bool hasPresentedWithBlit = false;
	nbl::video::ISwapchain::SCreationParams m_swapchainCreationParams;

	struct CSwapchainResources : public CommonAPI::IRetiredSwapchainResources
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImage> oldImage = nullptr;

		~CSwapchainResources() override {}
	};

public:
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

	void createSwapchainImage(uint32_t i)
	{
		auto& img = m_swapchainImages->begin()[i];
		img = swapchain->createImage(i);
		assert(img);
	}

	void onAppInitialized_impl() override
	{
		m_swapchainCreationParams = std::move(initOutput.swapchainCreationParams);

		CommonAPI::createSwapchain(std::move(logicalDevice), m_swapchainCreationParams, initParams.windowWidth, initParams.windowHeight, swapchain);
		assert(swapchain);

		commandPools = std::move(initOutput.commandPools);
		const auto& computeCommandPools = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

		const uint32_t swapchainImageCount = swapchain->getImageCount();
		m_swapchainImages = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<core::smart_refctd_ptr<video::IGPUImage>>>(swapchainImageCount);


		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			createSwapchainImage(i);
		}


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

// CB got recorded here

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
};