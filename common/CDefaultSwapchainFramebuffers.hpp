// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_C_DEFAULT_SWAPCHAIN_FRAMEBUFFERS_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_C_DEFAULT_SWAPCHAIN_FRAMEBUFFERS_HPP_INCLUDED_


// Build on top of the previous one
#include "nabla.h"


namespace nbl::examples
{
	
// Just a class to hold framebuffers derived from swapchain images
// WARNING: It assumes the format won't change between swapchain recreates!
class CDefaultSwapchainFramebuffers : public video::ISimpleManagedSurface::ISwapchainResources
{
	public:
		inline CDefaultSwapchainFramebuffers(core::smart_refctd_ptr<video::IGPURenderpass>&& _renderpass) : m_renderpass(std::move(_renderpass)) {}

		inline video::IGPUFramebuffer* getFrambuffer(const uint8_t imageIx)
		{
			if (imageIx<m_framebuffers.size())
				return m_framebuffers[imageIx].get();
			return nullptr;
		}

	protected:
		virtual inline void invalidate_impl()
		{
			std::fill(m_framebuffers.begin(),m_framebuffers.end(),nullptr);
		}

		// For creating extra per-image or swapchain resources you might need
		virtual inline bool onCreateSwapchain_impl(const uint8_t qFam)
		{
			using namespace nbl::video;
			auto device = const_cast<ILogicalDevice*>(m_renderpass->getOriginDevice());

			const auto swapchain = getSwapchain();
			const auto& sharedParams = swapchain->getCreationParameters().sharedParams;
			const auto count = swapchain->getImageCount();
			for (uint8_t i=0u; i<count; i++)
			{
				auto imageView = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = core::smart_refctd_ptr<video::IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = swapchain->getCreationParameters().surfaceFormat.format
				});
				m_framebuffers[i] = device->createFramebuffer({{
					.renderpass = core::smart_refctd_ptr(m_renderpass),
					.colorAttachments = &imageView.get(),
					.width = sharedParams.width,
					.height = sharedParams.height
				}});
				if (!m_framebuffers[i])
					return false;
			}
			return true;
		}

		core::smart_refctd_ptr<video::IGPURenderpass> m_renderpass;
		std::array<core::smart_refctd_ptr<video::IGPUFramebuffer>,video::ISwapchain::MaxImages> m_framebuffers;
};

}
#endif