
#include "CommonAPI.h"

nbl::video::IPhysicalDevice* CommonAPI::CDefaultPhysicalDeviceSelector::selectPhysicalDevice(const nbl::core::set<nbl::video::IPhysicalDevice*>& suitablePhysicalDevices)
{

	if (suitablePhysicalDevices.empty())
		return nullptr;

	for (auto itr = suitablePhysicalDevices.begin(); itr != suitablePhysicalDevices.end(); ++itr)
	{
		nbl::video::IPhysicalDevice* physdev = *itr;
		if (physdev->getProperties().driverID == preferredDriver)
			return physdev;
	}

	return *suitablePhysicalDevices.begin();
}

nbl::video::ISwapchain::SCreationParams CommonAPI::computeSwapchainCreationParams(
	uint32_t& imageCount,
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>& device,
	const nbl::core::smart_refctd_ptr<nbl::video::ISurface>& surface,
	nbl::asset::IImage::E_USAGE_FLAGS imageUsage,
	// Acceptable settings, ordered by preference.
	const nbl::asset::E_FORMAT* acceptableSurfaceFormats, uint32_t acceptableSurfaceFormatCount,
	const nbl::asset::E_COLOR_PRIMARIES* acceptableColorPrimaries, uint32_t acceptableColorPrimaryCount,
	const nbl::asset::ELECTRO_OPTICAL_TRANSFER_FUNCTION* acceptableEotfs, uint32_t acceptableEotfCount,
	const nbl::video::ISurface::E_PRESENT_MODE* acceptablePresentModes, uint32_t acceptablePresentModeCount,
	const nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS* acceptableSurfaceTransforms, uint32_t acceptableSurfaceTransformCount
)
{
	using namespace nbl;

	nbl::video::ISurface::SFormat surfaceFormat;
	nbl::video::ISurface::E_PRESENT_MODE presentMode = nbl::video::ISurface::EPM_UNKNOWN;
	nbl::video::ISurface::E_SURFACE_TRANSFORM_FLAGS surfaceTransform = nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM;

	if (device->getAPIType() == nbl::video::EAT_VULKAN)
	{
		nbl::video::ISurface::SCapabilities surfaceCapabilities;
		surface->getSurfaceCapabilitiesForPhysicalDevice(device->getPhysicalDevice(), surfaceCapabilities);

		for (uint32_t i = 0; i < acceptableSurfaceTransformCount; i++)
		{
			auto testSurfaceTransform = acceptableSurfaceTransforms[i];
			if (surfaceCapabilities.currentTransform == testSurfaceTransform)
			{
				surfaceTransform = testSurfaceTransform;
				break;
			}
		}
		assert(surfaceTransform != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM); // currentTransform must be supported in acceptableSurfaceTransforms

		auto availablePresentModes = surface->getAvailablePresentModesForPhysicalDevice(device->getPhysicalDevice());
		for (uint32_t i = 0; i < acceptablePresentModeCount; i++)
		{
			auto testPresentMode = acceptablePresentModes[i];
			if ((availablePresentModes & testPresentMode) == testPresentMode)
			{
				presentMode = testPresentMode;
				break;
			}
		}
		assert(presentMode != nbl::video::ISurface::EST_FLAG_BITS_MAX_ENUM);

		constexpr uint32_t MAX_SURFACE_FORMAT_COUNT = 1000u;
		uint32_t availableFormatCount;
		nbl::video::ISurface::SFormat availableFormats[MAX_SURFACE_FORMAT_COUNT];
		surface->getAvailableFormatsForPhysicalDevice(device->getPhysicalDevice(), availableFormatCount, availableFormats);

		for (uint32_t i = 0; i < availableFormatCount; ++i)
		{
			auto testsformat = availableFormats[i];
			bool supportsFormat = false;
			bool supportsEotf = false;
			bool supportsPrimary = false;

			for (uint32_t i = 0; i < acceptableSurfaceFormatCount; i++)
			{
				if (testsformat.format == acceptableSurfaceFormats[i])
				{
					supportsFormat = true;
					break;
				}
			}
			for (uint32_t i = 0; i < acceptableEotfCount; i++)
			{
				if (testsformat.colorSpace.eotf == acceptableEotfs[i])
				{
					supportsEotf = true;
					break;
				}
			}
			for (uint32_t i = 0; i < acceptableColorPrimaryCount; i++)
			{
				if (testsformat.colorSpace.primary == acceptableColorPrimaries[i])
				{
					supportsPrimary = true;
					break;
				}
			}

			if (supportsFormat && supportsEotf && supportsPrimary)
			{
				surfaceFormat = testsformat;
				break;
			}
		}
		// Require at least one of the acceptable options to be present
		assert(surfaceFormat.format != nbl::asset::EF_UNKNOWN &&
			surfaceFormat.colorSpace.primary != nbl::asset::ECP_COUNT &&
			surfaceFormat.colorSpace.eotf != nbl::asset::EOTF_UNKNOWN);

		imageCount = std::max(surfaceCapabilities.minImageCount, std::min(surfaceCapabilities.maxImageCount, imageCount));
	}
	else
	{
		// Temporary path until OpenGL reports properly!
		surfaceFormat = nbl::video::ISurface::SFormat(acceptableSurfaceFormats[0], acceptableColorPrimaries[0], acceptableEotfs[0]);
		presentMode = nbl::video::ISurface::EPM_IMMEDIATE;
		surfaceTransform = nbl::video::ISurface::EST_HORIZONTAL_MIRROR_ROTATE_180_BIT;
	}
	nbl::video::ISwapchain::SCreationParams sc_params = {};
	sc_params.arrayLayers = 1u;
	sc_params.minImageCount = imageCount;
	sc_params.presentMode = presentMode;
	sc_params.imageUsage = imageUsage;
	sc_params.surface = surface;
	sc_params.preTransform = surfaceTransform;
	sc_params.compositeAlpha = nbl::video::ISurface::ECA_OPAQUE_BIT;
	sc_params.surfaceFormat = surfaceFormat;

	return sc_params;
}

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

bool CommonAPI::createSwapchain(
	const nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice>&& device,
	nbl::video::ISwapchain::SCreationParams& params,
	uint32_t width, uint32_t height,
	// nullptr for initial creation, old swapchain for eventual resizes
	nbl::core::smart_refctd_ptr<nbl::video::ISwapchain>& swapchain
)
{
	auto oldSwapchain = swapchain;

	nbl::video::ISwapchain::SCreationParams paramsCp = params;
	paramsCp.width = width;
	paramsCp.height = height;
	paramsCp.oldSwapchain = oldSwapchain;

	assert(device->getAPIType() == nbl::video::EAT_VULKAN);
	swapchain = nbl::video::CVulkanSwapchain::create(std::move(device), std::move(paramsCp));
	assert(swapchain);
	assert(swapchain != oldSwapchain);

	return true;
}