// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

/**
This example just shows a screen which clears to red,
nothing fancy, just to show that Irrlicht links fine
**/
#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <nabla.h>

#if defined(_NBL_PLATFORM_WINDOWS_)
#	include <nbl/system/CColoredStdoutLoggerWin32.h>
#endif // TODO more platforms
// TODO: make these include themselves via `nabla.h`

#include "nbl/system/IApplicationFramework.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"

using namespace nbl;

#define LOG(...) printf(__VA_ARGS__); printf("\n");
class DemoEventCallback : public ui::IWindow::IEventCallback
{
public:
	bool isWindowOpen() const { return !m_gotWindowClosedMsg; }

private:
	bool onWindowShown_impl() override
	{
		LOG("Window Shown");
		return true;
	}
	bool onWindowHidden_impl() override
	{
		LOG("Window hidden");
		return true;
	}
	bool onWindowMoved_impl(int32_t x, int32_t y) override
	{
		LOG("Window window moved to { %d, %d }", x, y);
		return true;
	}
	bool onWindowResized_impl(uint32_t w, uint32_t h) override
	{
		LOG("Window resized to { %u, %u }", w, h);
		return true;
	}
	bool onWindowMinimized_impl() override
	{
		LOG("Window minimized");
		return true;
	}
	bool onWindowMaximized_impl() override
	{
		LOG("Window maximized");
		return true;
	}
	void onGainedMouseFocus_impl() override
	{
		LOG("Window gained mouse focus");
	}
	void onLostMouseFocus_impl() override
	{
		LOG("Window lost mouse focus");
	}
	void onGainedKeyboardFocus_impl() override
	{
		LOG("Window gained keyboard focus");
	}
	void onLostKeyboardFocus_impl() override
	{
		LOG("Window lost keyboard focus");
	}
	bool onWindowClosed_impl() override
	{
		LOG("Window closed");
		m_gotWindowClosedMsg = true;
		return true;
	}
	void onMouseConnected_impl(core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
	{
		LOG("A mouse has been connected");
	}
	void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
	{
		LOG("A mouse has been disconnected");
	}
	void onKeyboardConnected_impl(core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
	{
		LOG("A keyboard has been connected");
	}
	void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* mch) override
	{
		LOG("A keyboard has been disconnected");
	}

public:
	bool m_gotWindowClosedMsg = false;
};

static core::smart_refctd_ptr<system::ISystem> createSystem()
{
#ifdef _NBL_PLATFORM_WINDOWS_
	return nbl::core::make_smart_refctd_ptr<nbl::system::CSystemWin32>();
#endif
	return nullptr;
}

class HelloWorldSampleApp : public system::IApplicationFramework, public ui::IGraphicalApplicationFramework, public core::IReferenceCounted
{
	constexpr static uint32_t WIN_W = 800u;
	constexpr static uint32_t WIN_H = 600u;
	constexpr static uint32_t FRAMES_IN_FLIGHT = 5u;
	static constexpr uint64_t MAX_TIMEOUT = 99999999999999ull;

	core::smart_refctd_ptr<nbl::ui::IWindow> window;
	core::smart_refctd_ptr<DemoEventCallback> windowCb;
	core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
	core::smart_refctd_ptr<nbl::video::ISurface> surface;
	core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
	core::smart_refctd_ptr<nbl::video::ILogicalDevice> device;
	video::IPhysicalDevice* gpu;
	video::IGPUQueue* graphicsQueue = nullptr;
	video::IGPUQueue* presentQueue = nullptr;
	core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain;
	core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
	core::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>> fbos;
	core::smart_refctd_ptr<nbl::system::ISystem> system;
	core::smart_refctd_ptr<nbl::system::ILogger> logger;

	int32_t m_resourceIx = -1;
	uint32_t m_acquiredNextFBO = {};

	core::smart_refctd_ptr<video::IGPUSemaphore> m_imageAcquire[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUSemaphore> m_renderFinished[FRAMES_IN_FLIGHT] = { nullptr };
	core::smart_refctd_ptr<video::IGPUFence> m_frameComplete[FRAMES_IN_FLIGHT] = { nullptr };

	core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf[FRAMES_IN_FLIGHT] = { nullptr };

public:
	void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override
	{
		window = std::move(wnd);
	}
	void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}
	nbl::ui::IWindow* getWindow() override
	{
		return window.get();
	}
	video::IAPIConnection* getAPIConnection() override
	{
		return apiConnection.get();
	}
	video::ILogicalDevice* getLogicalDevice()  override
	{
		return device.get();
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
			fbos[i] = core::smart_refctd_ptr(f[i]);
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

	HelloWorldSampleApp(
		const std::filesystem::path& _localInputCWD,
		const std::filesystem::path& _localOutputCWD,
		const std::filesystem::path& _sharedInputCWD,
		const std::filesystem::path& _sharedOutputCWD) : system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	void onAppInitialized_impl() override
	{
		const char* APP_NAME = "01.HelloWorld";

		// create basic system objects
		system = createSystem();
		auto logLevelMask = core::bitflag(system::ILogger::ELL_DEBUG) | system::ILogger::ELL_PERFORMANCE | system::ILogger::ELL_WARNING | system::ILogger::ELL_ERROR;
		logger = core::make_smart_refctd_ptr<system::CColoredStdoutLoggerWin32>(logLevelMask);

		// set windo event callback
#ifndef _NBL_PLATFORM_ANDROID_
		auto windowManager = core::make_smart_refctd_ptr<nbl::ui::CWindowManagerWin32>();
		windowCb = core::make_smart_refctd_ptr<DemoEventCallback>();

		ui::IWindow::SCreationParams params;
		params.width = WIN_W;
		params.height = WIN_H;
		params.x = 64;
		params.y = 64;
		params.system = core::smart_refctd_ptr(system);
		params.flags = ui::IWindow::ECF_NONE;
		params.windowCaption = APP_NAME;
		params.callback = windowCb;
		// TODO (deprilula): Win32 window seems to be resizable despite the lack of resizability flag in the creation parameters!
		window = windowManager->createWindow(std::move(params));
#else
		assert(window);
		window->setEventCallback(core::smart_refctd_ptr(windowCb));
#endif

		// create API connection
		const auto swapChainMode = nbl::video::E_SWAPCHAIN_MODE::ESM_SURFACE;
		{
			nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable;
			apiFeaturesToEnable.swapchainMode = swapChainMode;
			apiFeaturesToEnable.validations = true;
			apiFeaturesToEnable.debugUtils = true;

			apiConnection = video::CVulkanConnection::create(
				core::smart_refctd_ptr(system),
				0, APP_NAME,
				core::smart_refctd_ptr(logger), apiFeaturesToEnable);
			assert(apiConnection);
		}

		{
			surface = video::CSurfaceVulkanWin32::create(
				core::smart_refctd_ptr<video::CVulkanConnection>(static_cast<video::CVulkanConnection*>(apiConnection.get())),
				core::smart_refctd_ptr<ui::IWindowWin32>(static_cast<ui::IWindowWin32*>(window.get())));
			assert(surface);
		}

		// Find a suitable gpu
		auto gpus = apiConnection->getPhysicalDevices();
		assert(!gpus.empty());
		uint32_t graphicsFamilyIndex(~0u);
		uint32_t presentFamilyIndex(~0u);

		// Todo(achal): Probably want to put these into some struct
		uint32_t minSwapchainImageCount(~0u);
		video::ISurface::SFormat surfaceFormat;
		video::ISurface::E_PRESENT_MODE presentMode;
		VkExtent2D swapchainExtent;

		// Todo(achal): Look at this:
		// https://github.com/Devsh-Graphics-Programming/Nabla/blob/6bd5061abe0a2020142efda827269ea6c07f0f2f/examples_tests/common/CommonAPI.h
		for (size_t i = 0ull; i < gpus.size(); ++i)
		{
			gpu = gpus.begin()[i];

			bool isGPUSuitable = false;

			// Todo(achal): Abstract out
			// Find required queue family indices
			{
				const auto& queueFamilyProperties = gpu->getQueueFamilyProperties();

				for (uint32_t familyIndex = 0u; familyIndex < queueFamilyProperties.size(); ++familyIndex)
				{
					const auto& familyProperty = queueFamilyProperties.begin() + familyIndex;

					if ((familyProperty->queueFlags & video::IPhysicalDevice::E_QUEUE_FLAGS::EQF_GRAPHICS_BIT).value)
						graphicsFamilyIndex = familyIndex;

					if (surface->isSupportedForPhysicalDevice(gpu, familyIndex))
						presentFamilyIndex = familyIndex;

					if ((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u))
					{
						isGPUSuitable = true;
						break;
					}
				}
			}

			// Since our workload is not headless compute, a swapchain is mandatory
			if (!gpu->getFeatures().swapchainMode.hasFlags(swapChainMode))
				isGPUSuitable = false;

			// Todo(achal): Abstract it out
			// Check if the surface is adequate
			{
				uint32_t surfaceFormatCount;
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, nullptr);
				std::vector<video::ISurface::SFormat> surfaceFormats(surfaceFormatCount);
				surface->getAvailableFormatsForPhysicalDevice(gpu, surfaceFormatCount, surfaceFormats.data());

				video::ISurface::E_PRESENT_MODE availablePresentModes =
					surface->getAvailablePresentModesForPhysicalDevice(gpu);

				video::ISurface::SCapabilities surfaceCapabilities = {};
				if (!surface->getSurfaceCapabilitiesForPhysicalDevice(gpu, surfaceCapabilities))
					isGPUSuitable = false;

				printf("Min swapchain image count: %d\n", surfaceCapabilities.minImageCount);
				printf("Max swapchain image count: %d\n", surfaceCapabilities.maxImageCount);

				if ((surfaceFormats.empty()) || (availablePresentModes == video::ISurface::EPM_UNKNOWN))
					isGPUSuitable = false;

				// Todo(achal): Probably a more sophisticated way to choose these
				minSwapchainImageCount = surfaceCapabilities.minImageCount + 1u;
				if (minSwapchainImageCount > surfaceCapabilities.maxImageCount)
					minSwapchainImageCount = surfaceCapabilities.maxImageCount;

				surfaceFormat = surfaceFormats[0];
				presentMode = static_cast<video::ISurface::E_PRESENT_MODE>(availablePresentModes & (1 << 0));
				swapchainExtent = surfaceCapabilities.currentExtent;
			}

			if (isGPUSuitable) // find the first suitable GPU
				break;
		}
		assert((graphicsFamilyIndex != ~0u) && (presentFamilyIndex != ~0u));

		video::ILogicalDevice::SCreationParams deviceCreationParams = {};
		if (graphicsFamilyIndex == presentFamilyIndex)
		{
			deviceCreationParams.queueParamsCount = 1u;
		}
		else
		{
			deviceCreationParams.queueParamsCount = 2u;
		}

		std::vector<uint32_t> queueFamilyIndices(deviceCreationParams.queueParamsCount);
		{
			const uint32_t temp[] = { graphicsFamilyIndex, presentFamilyIndex };
			for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
				queueFamilyIndices[i] = temp[i];
		}

		const float priority = video::IGPUQueue::DEFAULT_QUEUE_PRIORITY;
		std::vector<video::ILogicalDevice::SQueueCreationParams> queueCreationParams(deviceCreationParams.queueParamsCount);
		for (uint32_t i = 0u; i < deviceCreationParams.queueParamsCount; ++i)
		{
			queueCreationParams[i].familyIndex = queueFamilyIndices[i];
			queueCreationParams[i].count = 1u;
			queueCreationParams[i].flags = static_cast<video::IGPUQueue::E_CREATE_FLAGS>(0);
			queueCreationParams[i].priorities = &priority;
		}
		deviceCreationParams.queueParams = queueCreationParams.data();
		video::IPhysicalDevice::SFeatures requiredFeatures = {};
		requiredFeatures.swapchainMode = swapChainMode;
		deviceCreationParams.featuresToEnable = requiredFeatures;

		device = gpu->createLogicalDevice(std::move(deviceCreationParams));
		// no point concurrent sharing mode if only using one queue
		if (queueFamilyIndices.size()<2)
			queueFamilyIndices.clear();

		graphicsQueue = device->getQueue(graphicsFamilyIndex, 0u);
		presentQueue = device->getQueue(presentFamilyIndex, 0u);

		// create a swapchain
		{
			video::ISwapchain::SCreationParams sc_params = {};
			sc_params.surface = surface;
			sc_params.minImageCount = minSwapchainImageCount;
			sc_params.surfaceFormat = surfaceFormat;
			sc_params.presentMode = presentMode;
			sc_params.width = WIN_W;
			sc_params.height = WIN_H;
			sc_params.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
			sc_params.queueFamilyIndices = queueFamilyIndices.data();
			sc_params.preTransform = video::ISurface::EST_IDENTITY_BIT;
			sc_params.compositeAlpha = video::ISurface::ECA_OPAQUE_BIT;
			sc_params.imageUsage = asset::IImage::EUF_COLOR_ATTACHMENT_BIT;
			sc_params.oldSwapchain = nullptr;
			swapchain = nbl::video::CVulkanSwapchain::create(std::move(device), std::move(sc_params));
		}

		// Create render pass
		video::IGPURenderpass::SCreationParams::SAttachmentDescription attachmentDescription = {};
		attachmentDescription.format = surfaceFormat.format; // this should be same as the imageview used for this attachment
		attachmentDescription.samples = asset::IImage::ESCF_1_BIT;
		attachmentDescription.loadOp = video::IGPURenderpass::ELO_CLEAR; // when the first subpass begins with this attachment, clear its color and depth components
		attachmentDescription.storeOp = video::IGPURenderpass::ESO_STORE; // when the last subpass ends with this attachment, store its results
		attachmentDescription.initialLayout = asset::IImage::EL_UNDEFINED;
		attachmentDescription.finalLayout = asset::IImage::EL_PRESENT_SRC;

		video::IGPURenderpass::SCreationParams::SSubpassDescription subpassDescription = {};
		subpassDescription.flags = video::IGPURenderpass::ESDF_NONE;
		subpassDescription.pipelineBindPoint = asset::EPBP_GRAPHICS;
		subpassDescription.inputAttachmentCount = 0u;
		subpassDescription.inputAttachments = nullptr;

		video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
		{
			colorAttRef.attachment = 0u;
			colorAttRef.layout = asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;
		}
		subpassDescription.colorAttachmentCount = 1u;
		subpassDescription.colorAttachments = &colorAttRef;
		subpassDescription.resolveAttachments = nullptr;
		subpassDescription.depthStencilAttachment = nullptr;
		subpassDescription.preserveAttachmentCount = 0u;
		subpassDescription.preserveAttachments = nullptr;

		video::IGPURenderpass::SCreationParams renderPassParams;
		renderPassParams.attachmentCount = 1u;
		renderPassParams.attachments = &attachmentDescription;
		renderPassParams.dependencies = nullptr;
		renderPassParams.dependencyCount = 0u;
		renderPassParams.subpasses = &subpassDescription;
		renderPassParams.subpassCount = 1u;

		renderpass = device->createRenderpass(renderPassParams);
		
		const auto swapchainImageCount = swapchain->getImageCount();
		fbos.resize(swapchainImageCount);
		for (uint32_t i = 0u; i < swapchainImageCount; ++i)
		{
			nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view = {};

			auto img = swapchain->createImage(i);
			{
				nbl::video::IGPUImageView::SCreationParams view_params;
				view_params.format = img->getCreationParameters().format;
				view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
				view_params.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
				view_params.subresourceRange.baseMipLevel = 0u;
				view_params.subresourceRange.levelCount = 1u;
				view_params.subresourceRange.baseArrayLayer = 0u;
				view_params.subresourceRange.layerCount = 1u;
				view_params.image = std::move(img);

				view = device->createImageView(std::move(view_params));
				assert(view);
			}

			video::IGPUFramebuffer::SCreationParams fbParams;
			fbParams.width = WIN_W;
			fbParams.height = WIN_H;
			fbParams.layers = 1u;
			fbParams.renderpass = renderpass;
			fbParams.flags = static_cast<video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
			fbParams.attachmentCount = renderpass->getAttachments().size();
			fbParams.attachments = &view;

			fbos[i] = device->createFramebuffer(std::move(fbParams));
		}

		for (uint32_t i = 0u; i < FRAMES_IN_FLIGHT; ++i)
		{
			m_imageAcquire[i] = device->createSemaphore();
			m_renderFinished[i] = device->createSemaphore();
		}

		for (size_t i = 0; i < FRAMES_IN_FLIGHT; i++)
		{
			auto cmdPool = device->createCommandPool(graphicsFamilyIndex, video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			device->createCommandBuffers(cmdPool.get(), video::IGPUCommandBuffer::EL_PRIMARY,
				1, m_cmdbuf + i);

		}

	}

	void onAppTerminated_impl() override
	{
		device->waitIdle();
	}

	void workLoopBody() override
	{
		++m_resourceIx;
		if (m_resourceIx >= FRAMES_IN_FLIGHT)
			m_resourceIx = 0;

		auto& commandBuffer = m_cmdbuf[m_resourceIx];
		auto& fence = m_frameComplete[m_resourceIx];

		if (fence)
		{
			device->blockForFences(1u, &fence.get(), true);
			device->resetFences(1u, &fence.get());
		}
		else
			fence = device->createFence(static_cast<video::IGPUFence::E_CREATE_FLAGS>(0));

		commandBuffer->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);  // TODO: Reset Frame's CommandPool

		swapchain->acquireNextImage(MAX_TIMEOUT, m_imageAcquire[m_resourceIx].get(), nullptr, &m_acquiredNextFBO);

		video::IGPUCommandBuffer::SRenderpassBeginInfo beginInfo = {};
		{
			VkRect2D area;
			area.offset = { 0,0 };
			area.extent = { WIN_W, WIN_H };
			asset::SClearValue clear[2] = {};
			clear[0].color.float32[0] = 1.f;
			clear[0].color.float32[1] = 0.f;
			clear[0].color.float32[2] = 0.f;
			clear[0].color.float32[3] = 1.f;
			clear[1].depthStencil.depth = 0.f;

			beginInfo.clearValueCount = 2u;
			beginInfo.framebuffer = fbos[m_acquiredNextFBO];
			beginInfo.renderpass = renderpass;
			beginInfo.renderArea = area;
			beginInfo.clearValues = clear;
		}
		commandBuffer->beginRenderPass(&beginInfo, nbl::asset::ESC_INLINE);

		// Do nothing

		commandBuffer->endRenderPass();
		commandBuffer->end();

		asset::E_PIPELINE_STAGE_FLAGS waitDstStageFlags = asset::E_PIPELINE_STAGE_FLAGS::EPSF_COLOR_ATTACHMENT_OUTPUT_BIT;

		video::IGPUQueue::SSubmitInfo submitInfo = {};
		submitInfo.waitSemaphoreCount = 1u;
		submitInfo.pWaitSemaphores = &m_imageAcquire[m_resourceIx].get();
		submitInfo.pWaitDstStageMask = &waitDstStageFlags;
		submitInfo.signalSemaphoreCount = 1u;
		submitInfo.pSignalSemaphores = &m_renderFinished[m_resourceIx].get();
		submitInfo.commandBufferCount = 1u;
		submitInfo.commandBuffers = &commandBuffer.get();
		graphicsQueue->submit(1u, &submitInfo, fence.get());

		video::ISwapchain::SPresentInfo presentInfo;
		presentInfo.waitSemaphoreCount = 1u;
		presentInfo.waitSemaphores = &m_renderFinished[m_resourceIx].get();
		presentInfo.imgIndex = m_acquiredNextFBO;
		swapchain->present(presentQueue, presentInfo);
	}

	bool keepRunning() override
	{
		return windowCb->isWindowOpen();
	}
};

int main(int argc, char** argv)
{
#ifndef _NBL_PLATFORM_ANDROID_
	system::path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
	nbl::system::path sharedInputCWD = CWD / "../../media/";
	nbl::system::path sharedOutputCWD = CWD / "../../tmp/";;
	nbl::system::path localInputCWD = CWD / "../";
	nbl::system::path localOutputCWD = CWD;

	auto app = nbl::core::template make_smart_refctd_ptr<HelloWorldSampleApp>(
		localInputCWD,
		localOutputCWD,
		sharedInputCWD,
		sharedOutputCWD);

	for (size_t i = 0; i < argc; ++i)
		app->argv.push_back(std::string(argv[i]));

	app->onAppInitialized();
	while (app->keepRunning())
	{
		app->workLoopBody();
	}
	app->onAppTerminated();
#endif

	return 0;
}

