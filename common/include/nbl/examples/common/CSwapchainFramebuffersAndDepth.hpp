// Copyright (C) 2023-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_C_SWAPCHAIN_FRAMEBUFFERS_AND_DEPTH_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_C_SWAPCHAIN_FRAMEBUFFERS_AND_DEPTH_HPP_INCLUDED_

// Build on top of the previous one
#include "nbl/application_templates/BasicMultiQueueApplication.hpp"

namespace nbl::examples
{
	
class CSwapchainFramebuffersAndDepth final : public video::CDefaultSwapchainFramebuffers
{
		using base_t = CDefaultSwapchainFramebuffers;

	public:
		template<typename... Args>
		inline CSwapchainFramebuffersAndDepth(video::ILogicalDevice* device, const asset::E_FORMAT _desiredDepthFormat, Args&&... args) : base_t(device,std::forward<Args>(args)...)
		{
			// user didn't want any depth
			if (_desiredDepthFormat==asset::EF_UNKNOWN)
				return;

			using namespace nbl::asset;
			using namespace nbl::video;
			const IPhysicalDevice::SImageFormatPromotionRequest req = {
				.originalFormat = _desiredDepthFormat,
				.usages = {IGPUImage::EUF_RENDER_ATTACHMENT_BIT}
			};
			m_depthFormat = m_device->getPhysicalDevice()->promoteImageFormat(req,IGPUImage::TILING::OPTIMAL);

			const static IGPURenderpass::SCreationParams::SDepthStencilAttachmentDescription depthAttachments[] = {
				{{
					{
						.format = m_depthFormat,
						.samples = IGPUImage::ESCF_1_BIT,
						.mayAlias = false
					},
					/*.loadOp = */{IGPURenderpass::LOAD_OP::CLEAR},
					/*.storeOp = */{IGPURenderpass::STORE_OP::STORE},
					/*.initialLayout = */{IGPUImage::LAYOUT::UNDEFINED}, // because we clear we don't care about contents
					/*.finalLayout = */{IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}
				}},
				IGPURenderpass::SCreationParams::DepthStencilAttachmentsEnd
			};
			m_params.depthStencilAttachments = depthAttachments;

			static IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
				m_params.subpasses[0],
				IGPURenderpass::SCreationParams::SubpassesEnd
			};
			subpasses[0].depthStencilAttachment.render = { .attachmentIndex = 0,.layout = IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL };
			m_params.subpasses = subpasses;
		}

	protected:
		inline bool onCreateSwapchain_impl(const uint8_t qFam) override
		{
			using namespace nbl::asset;
			using namespace nbl::video;
			if (m_depthFormat!=asset::EF_UNKNOWN)
			{
				// DOCS: why are we not using `m_device` here? any particular reason?
				auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

				const auto depthFormat = m_renderpass->getCreationParameters().depthStencilAttachments[0].format;
				const auto& sharedParams = getSwapchain()->getCreationParameters().sharedParams;
				auto image = device->createImage({ IImage::SCreationParams{
					.type = IGPUImage::ET_2D,
					.samples = IGPUImage::ESCF_1_BIT,
					.format = depthFormat,
					.extent = {sharedParams.width,sharedParams.height,1},
					.mipLevels = 1,
					.arrayLayers = 1,
					.depthUsage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT
				} });

				device->allocate(image->getMemoryReqs(), image.get());

				m_depthBuffer = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = std::move(image),
					.viewType = IGPUImageView::ET_2D,
					.format = depthFormat,
					.subresourceRange = {IGPUImage::EAF_DEPTH_BIT,0,1,0,1}
					});
			}
			const auto retval = base_t::onCreateSwapchain_impl(qFam);
			m_depthBuffer = nullptr;
			return retval;
		}

		inline core::smart_refctd_ptr<video::IGPUFramebuffer> createFramebuffer(video::IGPUFramebuffer::SCreationParams&& params) override
		{
			if (m_depthBuffer)
				params.depthStencilAttachments = &m_depthBuffer.get();
			return m_device->createFramebuffer(std::move(params));
		}

		asset::E_FORMAT m_depthFormat = asset::EF_UNKNOWN;
		// only used to pass a parameter from `onCreateSwapchain_impl` to `createFramebuffer`
		core::smart_refctd_ptr<video::IGPUImageView> m_depthBuffer;
};

}
#endif