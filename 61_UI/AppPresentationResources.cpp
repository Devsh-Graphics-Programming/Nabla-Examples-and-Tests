#include "app/App.hpp"

nbl::hlsl::uint32_t2 App::getPresentationRenderExtent() const
{
	if (m_cliRuntime.ciMode)
		return SCameraAppPresentationDefaults::CiWindowExtent;

	const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
	return nbl::hlsl::uint32_t2(dpyInfo.resX, dpyInfo.resY);
}

bool App::shouldMaximizePresentationWindow() const
{
	return !m_cliRuntime.ciMode;
}

core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> App::getSurfaces() const
{
	if (!m_surface)
	{
		const auto presentationExtent = getPresentationRenderExtent();
		auto windowCallback = core::make_smart_refctd_ptr<examples::CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));

		IWindow::SCreationParams params = {};
		params.callback = windowCallback;
		params.width = presentationExtent.x;
		params.height = presentationExtent.y;
		params.x = SCameraAppPresentationDefaults::WindowOrigin.x;
		params.y = SCameraAppPresentationDefaults::WindowOrigin.y;
		params.flags = IWindow::ECF_INPUT_FOCUS | IWindow::ECF_CAN_RESIZE | IWindow::ECF_CAN_MAXIMIZE | IWindow::ECF_CAN_MINIMIZE;
		params.windowCaption = "[Nabla Engine] UI App";

		const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
		auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
		const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSmoothResizeSurface<CSwapchainResources>::create(std::move(surface));
	}

	if (!m_surface)
		return {};

	if (shouldMaximizePresentationWindow())
		m_window->getManager()->maximize(m_window.get());
	m_window->getCursorControl()->setVisible(true);
	return { {m_surface->getSurface()} };
}

bool App::initializePresentationResources()
{
	m_semaphore = m_device->createSemaphore(m_realFrameIx);
	if (!m_semaphore)
		return logFail("Failed to Create a Semaphore!");

	const auto format = asset::EF_R8G8B8A8_SRGB;
	const auto samples = IGPUImage::ESCF_1_BIT;

	{
		IGPURenderpass::SCreationParams params = {};
		const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
			{{
				{
					.format = format,
					.samples = samples,
					.mayAlias = false
				},
				/*.loadOp = */IGPURenderpass::LOAD_OP::CLEAR,
				/*.storeOp = */IGPURenderpass::STORE_OP::STORE,
				/*.initialLayout = */IGPUImage::LAYOUT::UNDEFINED,
				/*.finalLayout = */ IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL
			}},
			IGPURenderpass::SCreationParams::ColorAttachmentsEnd
		};
		params.colorAttachments = colorAttachments;
		IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
			{},
			IGPURenderpass::SCreationParams::SubpassesEnd
		};
		subpasses[0].colorAttachments[0] = { .render = { .attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL } };
		params.subpasses = subpasses;
		const IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
			{
				.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.dstSubpass = 0,
				.memoryBarrier = {
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
					.srcAccessMask = asset::ACCESS_FLAGS::NONE,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.dstAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
				}
			},
			{
				.srcSubpass = 0,
				.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
				.memoryBarrier = {
					.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
					.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
					.dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
					.dstAccessMask = asset::ACCESS_FLAGS::NONE
				}
			},
			IGPURenderpass::SCreationParams::DependenciesEnd
		};
		params.dependencies = dependencies;
		m_renderpass = m_device->createRenderpass(std::move(params));
		if (!m_renderpass)
			return logFail("Failed to Create a Renderpass!");
	}

	if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get()), std::make_unique<CSwapchainResources>(), { .imageUsage = IGPUImage::EUF_TRANSFER_SRC_BIT }))
		return logFail("Failed to Create a Swapchain!");

	const auto presentationExtent = getPresentationRenderExtent();
	for (uint32_t i = 0u; i < MaxFramesInFlight; i++)
	{
		auto& image = m_tripleBuffers[i];
		{
			IGPUImage::SCreationParams params = {};
			params = asset::IImage::SCreationParams{
				.type = IGPUImage::ET_2D,
				.samples = samples,
				.format = format,
				.extent = { presentationExtent.x,presentationExtent.y,1 },
				.mipLevels = 1,
				.arrayLayers = 1,
				.flags = IGPUImage::ECF_NONE,
				.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT
			};
			image = m_device->createImage(std::move(params));
			if (!image)
				return logFail("Failed to Create Triple Buffer Image!");

			if (!m_device->allocate(image->getMemoryReqs(), image.get()).isValid())
				return logFail("Failed to allocate Device Memory for Image %d", i);
		}
		image->setObjectDebugName(("Triple Buffer Image " + std::to_string(i)).c_str());

		auto imageView = m_device->createImageView({
			.flags = IGPUImageView::ECF_NONE,
			.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT | IGPUImage::EUF_TRANSFER_SRC_BIT,
			.image = core::smart_refctd_ptr(image),
			.viewType = IGPUImageView::ET_2D,
			.format = format
		});
		const auto& imageParams = image->getCreationParameters();
		IGPUFramebuffer::SCreationParams params = { {
			.renderpass = core::smart_refctd_ptr(m_renderpass),
			.depthStencilAttachments = nullptr,
			.colorAttachments = &imageView.get(),
			.width = imageParams.extent.width,
			.height = imageParams.extent.height,
			.layers = imageParams.arrayLayers
		} };
		m_framebuffers[i] = m_device->createFramebuffer(std::move(params));
		if (!m_framebuffers[i])
			return logFail("Failed to Create a Framebuffer for Image %d", i);
	}

	auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
	if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { m_cmdBufs.data(),MaxFramesInFlight }, core::smart_refctd_ptr(m_logger)))
		return logFail("Failed to Create CommandBuffers!");

	return true;
}
