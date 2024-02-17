#include "CommonAPI.h"

#if 0
void CommonAPI::dropRetiredSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, const uint64_t completedFrameId)
{
	while (!qRetiredSwapchainResources.empty() && qRetiredSwapchainResources.front()->retiredFrameId < completedFrameId)
	{
		std::cout << "Dropping resource scheduled at " << qRetiredSwapchainResources.front()->retiredFrameId << " with completedFrameId " << completedFrameId << "\n";
		delete(qRetiredSwapchainResources.front());
		qRetiredSwapchainResources.pop_front();
	}
}

void CommonAPI::retireSwapchainResources(nbl::core::deque<IRetiredSwapchainResources*>& qRetiredSwapchainResources, IRetiredSwapchainResources* retired)
{
	qRetiredSwapchainResources.push_back(retired);
}

nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> CommonAPI::createRenderpass(const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device, nbl::asset::E_FORMAT colorAttachmentFormat, nbl::asset::E_FORMAT baseDepthFormat)
{
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = device->getPhysicalDevice()->promoteImageFormat(
			{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
			nbl::video::IGPUImage::ET_OPTIMAL
		);
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	nbl::video::IGPURenderpass::SCreationParams::SAttachmentDescription attachments[2];
	attachments[0].initialLayout = asset::IImage::EL_UNDEFINED;
	attachments[0].finalLayout = asset::IImage::EL_PRESENT_SRC;
	attachments[0].format = colorAttachmentFormat;
	attachments[0].samples = asset::IImage::ESCF_1_BIT;
	attachments[0].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[0].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	attachments[1].initialLayout = asset::IImage::EL_UNDEFINED;
	attachments[1].finalLayout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	attachments[1].format = depthFormat;
	attachments[1].samples = asset::IImage::ESCF_1_BIT;
	attachments[1].loadOp = nbl::video::IGPURenderpass::ELO_CLEAR;
	attachments[1].storeOp = nbl::video::IGPURenderpass::ESO_STORE;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef colorAttRef;
	colorAttRef.attachment = 0u;
	colorAttRef.layout = asset::IImage::EL_COLOR_ATTACHMENT_OPTIMAL;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription::SAttachmentRef depthStencilAttRef;
	depthStencilAttRef.attachment = 1u;
	depthStencilAttRef.layout = asset::IImage::EL_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	nbl::video::IGPURenderpass::SCreationParams::SSubpassDescription sp;
	sp.pipelineBindPoint = asset::EPBP_GRAPHICS;
	sp.colorAttachmentCount = 1u;
	sp.colorAttachments = &colorAttRef;
	if (useDepth) {
		sp.depthStencilAttachment = &depthStencilAttRef;
	}
	else {
		sp.depthStencilAttachment = nullptr;
	}
	sp.flags = nbl::video::IGPURenderpass::ESDF_NONE;
	sp.inputAttachmentCount = 0u;
	sp.inputAttachments = nullptr;
	sp.preserveAttachmentCount = 0u;
	sp.preserveAttachments = nullptr;
	sp.resolveAttachments = nullptr;

	nbl::video::IGPURenderpass::SCreationParams rp_params;
	rp_params.attachmentCount = (useDepth) ? 2u : 1u;
	rp_params.attachments = attachments;
	rp_params.dependencies = nullptr;
	rp_params.dependencyCount = 0u;
	rp_params.subpasses = &sp;
	rp_params.subpassCount = 1u;

	return device->createRenderpass(rp_params);
}

nbl::core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>> CommonAPI::createFBOWithSwapchainImages(
	size_t imageCount, uint32_t width, uint32_t height,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain> swapchain,
	nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass,
	nbl::asset::E_FORMAT baseDepthFormat
) {
	using namespace nbl;

	bool useDepth = baseDepthFormat != nbl::asset::EF_UNKNOWN;
	nbl::asset::E_FORMAT depthFormat = nbl::asset::EF_UNKNOWN;
	if (useDepth)
	{
		depthFormat = baseDepthFormat;
		//depthFormat = device->getPhysicalDevice()->promoteImageFormat(
		//	{ baseDepthFormat, nbl::video::IPhysicalDevice::SFormatImageUsages::SUsage(nbl::asset::IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT) },
		//	nbl::asset::IImage::ET_OPTIMAL
		//);
		// TODO error reporting
		assert(depthFormat != nbl::asset::EF_UNKNOWN);
	}

	auto fbo = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<nbl::core::smart_refctd_ptr<nbl::video::IGPUFramebuffer>>>(imageCount);
	for (uint32_t i = 0u; i < imageCount; ++i)
	{
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> view[2] = {};

		auto img = swapchain->createImage(i);
		{
			nbl::video::IGPUImageView::SCreationParams view_params = {};
			view_params.format = img->getCreationParameters().format;
			view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
			view_params.subresourceRange.aspectMask = asset::IImage::EAF_COLOR_BIT;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(img);

			view[0] = device->createImageView(std::move(view_params));
			assert(view[0]);
		}

		if (useDepth) {
			nbl::video::IGPUImage::SCreationParams imgParams = {};
			imgParams.flags = static_cast<asset::IImage::E_CREATE_FLAGS>(0u);
			imgParams.type = asset::IImage::ET_2D;
			imgParams.format = depthFormat;
			imgParams.extent = { width, height, 1 };
			imgParams.usage = asset::IImage::E_USAGE_FLAGS::EUF_DEPTH_STENCIL_ATTACHMENT_BIT;
			imgParams.mipLevels = 1u;
			imgParams.arrayLayers = 1u;
			imgParams.samples = asset::IImage::ESCF_1_BIT;

			auto depthImg = device->createImage(std::move(imgParams));
			auto depthImgMemReqs = depthImg->getMemoryReqs();
			depthImgMemReqs.memoryTypeBits &= device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
			auto depthImgMem = device->allocate(depthImgMemReqs, depthImg.get());

			nbl::video::IGPUImageView::SCreationParams view_params;
			view_params.format = depthFormat;
			view_params.viewType = asset::IImageView<nbl::video::IGPUImage>::ET_2D;
			view_params.subresourceRange.aspectMask = asset::IImage::EAF_DEPTH_BIT;
			view_params.subresourceRange.baseMipLevel = 0u;
			view_params.subresourceRange.levelCount = 1u;
			view_params.subresourceRange.baseArrayLayer = 0u;
			view_params.subresourceRange.layerCount = 1u;
			view_params.image = std::move(depthImg);

			view[1] = device->createImageView(std::move(view_params));
			assert(view[1]);
		}

		nbl::video::IGPUFramebuffer::SCreationParams fb_params;
		fb_params.width = width;
		fb_params.height = height;
		fb_params.layers = 1u;
		fb_params.renderpass = renderpass;
		fb_params.flags = static_cast<nbl::video::IGPUFramebuffer::E_CREATE_FLAGS>(0);
		fb_params.attachmentCount = (useDepth) ? 2u : 1u;
		fb_params.attachments = view;

		fbo->begin()[i] = device->createFramebuffer(std::move(fb_params));
		assert(fbo->begin()[i]);
	}
	return fbo;
}
#endif