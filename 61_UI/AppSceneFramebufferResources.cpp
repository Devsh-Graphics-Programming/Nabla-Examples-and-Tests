#include "app/App.hpp"

namespace
{

smart_refctd_ptr<IGPUImageView> createSceneAttachmentView(ILogicalDevice* device, E_FORMAT format, uint32_t width, uint32_t height, const char* debugName)
{
	if (!device)
		return nullptr;

	const bool isDepth = isDepthOrStencilFormat(format);
	auto usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT;
	if (!isDepth)
		usage |= IGPUImage::EUF_SAMPLED_BIT;

	auto image = device->createImage({{
		.type = IGPUImage::ET_2D,
		.samples = IGPUImage::ESCF_1_BIT,
		.format = format,
		.extent = { width, height, 1u },
		.mipLevels = 1u,
		.arrayLayers = 1u,
		.usage = usage
	}});
	if (!image)
		return nullptr;

	image->setObjectDebugName(debugName);
	if (!device->allocate(image->getMemoryReqs(), image.get()).isValid())
		return nullptr;

	IGPUImageView::SCreationParams params = {
		.subUsages = usage,
		.image = std::move(image),
		.viewType = IGPUImageView::ET_2D,
		.format = format
	};
	params.subresourceRange.aspectMask = isDepth ? IGPUImage::EAF_DEPTH_BIT : IGPUImage::EAF_COLOR_BIT;
	return device->createImageView(std::move(params));
}

smart_refctd_ptr<IGPUFramebuffer> createSceneFramebuffer(
	ILogicalDevice* device,
	IGPURenderpass* renderpass,
	IGPUImageView* colorView,
	IGPUImageView* depthView)
{
	if (!device || !renderpass || !colorView || !depthView)
		return nullptr;

	const auto& imageParams = colorView->getCreationParameters().image->getCreationParameters();
	IGPUFramebuffer::SCreationParams params = { {
		.renderpass = core::smart_refctd_ptr<IGPURenderpass>(renderpass),
		.depthStencilAttachments = &depthView,
		.colorAttachments = &colorView,
		.width = imageParams.extent.width,
		.height = imageParams.extent.height,
		.layers = imageParams.arrayLayers
	} };
	return device->createFramebuffer(std::move(params));
}

} // namespace

bool App::initializeSceneRenderpass()
{
	IGPURenderpass::SCreationParams params = {};
	const IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
		{{
			{
				.format = SCameraAppRenderDefaults::SceneDepthFormat,
				.samples = IGPUImage::ESCF_1_BIT,
				.mayAlias = false
			},
			{ IGPURenderpass::LOAD_OP::CLEAR },
			{ IGPURenderpass::STORE_OP::STORE },
			{ IGPUImage::LAYOUT::UNDEFINED },
			{ IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL }
		}},
		IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
	};
	params.depthStencilAttachments = depthAttachments;
	const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
		{{
			{
				.format = SCameraAppRenderDefaults::FinalSceneFormat,
				.samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
				.mayAlias = false
			},
			IGPURenderpass::LOAD_OP::CLEAR,
			IGPURenderpass::STORE_OP::STORE,
			IGPUImage::LAYOUT::UNDEFINED,
			IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
		}},
		IGPURenderpass::SCreationParams::ColorAttachmentsEnd
	};
	params.colorAttachments = colorAttachments;
	IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
		{},
		IGPURenderpass::SCreationParams::SubpassesEnd
	};
	subpasses[0].depthStencilAttachment = { { .render = { .attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL } } };
	subpasses[0].colorAttachments[0] = { .render = { .attachmentIndex = 0, .layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL } };
	params.subpasses = subpasses;
	static constexpr IGPURenderpass::SCreationParams::SSubpassDependency dependencies[] = {
		{
			.srcSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
			.dstSubpass = 0,
			.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::LATE_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT,
				.srcAccessMask = ACCESS_FLAGS::NONE,
				.dstStageMask = PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT | PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstAccessMask = ACCESS_FLAGS::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT
			}
		},
		{
			.srcSubpass = 0,
			.dstSubpass = IGPURenderpass::SCreationParams::SSubpassDependency::External,
			.memoryBarrier = {
				.srcStageMask = PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
				.dstStageMask = PIPELINE_STAGE_FLAGS::FRAGMENT_SHADER_BIT | PIPELINE_STAGE_FLAGS::EARLY_FRAGMENT_TESTS_BIT,
				.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT
			}
		},
		IGPURenderpass::SCreationParams::DependenciesEnd
	};
	params.dependencies = dependencies;
	m_debugScene.renderpass = m_device->createRenderpass(std::move(params));
	return m_debugScene.renderpass || logFail("Failed to create Scene Renderpass!");
}

bool App::initializeWindowSceneFramebufferResources()
{
	const auto presentationExtent = getPresentationRenderExtent();
	for (uint32_t i = 0u; i < m_viewports.windowBindings.size(); ++i)
	{
		auto& binding = m_viewports.windowBindings[i];
		binding.sceneColorView = createSceneAttachmentView(m_device.get(), SCameraAppRenderDefaults::FinalSceneFormat, presentationExtent.x, presentationExtent.y, "UI Scene Color Attachment");
		binding.sceneDepthView = createSceneAttachmentView(m_device.get(), SCameraAppRenderDefaults::SceneDepthFormat, presentationExtent.x, presentationExtent.y, "UI Scene Depth Attachment");
		binding.sceneFramebuffer = createSceneFramebuffer(m_device.get(), m_debugScene.renderpass.get(), binding.sceneColorView.get(), binding.sceneDepthView.get());
		if (!binding.sceneFramebuffer)
			return logFail("Could not create geometry creator scene[%d]!", i);
	}

	return true;
}
