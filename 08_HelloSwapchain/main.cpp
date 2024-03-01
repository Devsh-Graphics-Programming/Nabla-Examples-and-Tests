// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "../common/SimpleWindowedApplication.hpp"

//
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanSwapchain.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace video;

//
class CSwapchainResources final : public IResizableSurface::ISwapchainResources
{
	public:
		// Because we blit to the swapchain image asynchronously, we need a queue which can not only present but also perform graphics commands.
		// If we for example used a compute shader to tonemap and MSAA resolve, we'd request the COMPUTE_BIT here. 
		constexpr static inline IQueue::FAMILY_FLAGS RequiredQueueFlags = IQueue::FAMILY_FLAGS::GRAPHICS_BIT;

		// If these get used, they will indirectly find their way into the `frameResources` argument of the `present` method.
		core::smart_refctd_ptr<IGPURenderpass> defaultRenderpass = {};
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> defaultFramebuffers = {};

	protected:
		inline void invalidate_impl() override
		{
			// Framebuffers hold onto the renderpass they were created from, and swapchain images (swapchain itself indirectly)
			std::fill_n(defaultFramebuffers.data(),ISwapchain::MaxImages,nullptr);
			defaultRenderpass = nullptr;
		}

		inline bool onCreateSwapchain_impl(const uint8_t qFam) override
		{
			auto device = const_cast<ILogicalDevice*>(swapchain->getOriginDevice());
			const auto& swapchainParams = swapchain->getCreationParameters();
			const auto masterFormat = swapchainParams.surfaceFormat.format;

			// create renderpass
			{
				const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
					{{
						.format = masterFormat,
						.samples = IGPUImage::ESCF_1_BIT,
						.mayAlias = false,
						.loadOp = IGPURenderpass::LOAD_OP::CLEAR,
						.storeOp = IGPURenderpass::STORE_OP::STORE,
						.initialLayout = IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents
						.finalLayout = IGPUImage::LAYOUT::PRESENT_SRC
					}},
					IGPURenderpass::SCreationParams::ColorAttachmentsEnd
				};
				IGPURenderpass::SCreationParams::SSubpassDescription subpasses[] = {
					{},
					IGPURenderpass::SCreationParams::SubpassesEnd
				};
				subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};

				IGPURenderpass::SCreationParams params = {};
				params.colorAttachments = colorAttachments;
				params.subpasses = subpasses;
				// no subpass dependencies
				defaultRenderpass = device->createRenderpass(params);
				if (!defaultRenderpass)
					return false;
			}

			// create framebuffers for the images
			for (auto i=0u; i<swapchain->getImageCount(); i++)
			{
				auto imageView = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = core::smart_refctd_ptr<IGPUImage>(getImage(i)),
					.viewType = IGPUImageView::ET_2D,
					.format = masterFormat
				});
				const auto& swapchainSharedParams = swapchainParams.sharedParams;
				IGPUFramebuffer::SCreationParams params = {{
					.renderpass = core::smart_refctd_ptr(defaultRenderpass),
					.depthStencilAttachments = nullptr,
					.colorAttachments = &imageView.get(),
					.width = swapchainSharedParams.width,
					.height = swapchainSharedParams.height,
					.layers = swapchainSharedParams.arrayLayers
				}};
				defaultFramebuffers[i] = device->createFramebuffer(std::move(params));
				if (!defaultFramebuffers[i])
					return false;
			}

			return true;
		}
		
		inline asset::PIPELINE_STAGE_FLAGS tripleBufferPresent(IGPUCommandBuffer* cmdbuf, const IResizableSurface::image_barrier_t& source, const uint8_t imageindex) override
		{
			bool success = true;

			const auto blitSrcLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
			const auto blitDstLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};

#if 0
			// barrier before
			const image_barrier_t preBarriers[2] = {
				{
					.barrier = {
						.dep = {
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE, // acquire isn't a stage
							.srcAccessMask = asset::ACCESS_FLAGS::NONE, // performs no accesses
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
							.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_WRITE_BIT
						}
					},
					.image = acquiredImage,
					.subresourceRange = {
						.aspectMask = IGPUImage::EAF_COLOR_BIT,
						.baseMipLevel = 0,
						.levelCount = 1,
						.baseArrayLayer = 0,
						.layerCount = 1
					},
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED, // I do not care about previous contents
					.newLayout = blitDstLayout
				},
				{
					.barrier = {
						.dep = {
							.srcStageMask = contents.barrier.dep.srcStageMask,
							.srcAccessMask = contents.barrier.dep.srcAccessMask,
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
							.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
						.otherQueueFamilyIndex = contents.barrier.otherQueueFamilyIndex
					},
					.image = contents.image,
					.subresourceRange = contents.subresourceRange,
					.oldLayout = contents.oldLayout,
					.newLayout = blitSrcLayout
				}
			};
			depInfo.imgBarriers = preBarriers;
			retval &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);
		
			// TODO: Implement scaling modes other than plain STRETCH, and allow for using subrectangles of the initial contents
			{
				const auto srcExtent = contents.image->getCreationParameters().extent;
				const auto dstExtent = acquiredImage->getCreationParameters().extent;
				const IGPUCommandBuffer::SImageBlit regions[1] = {{
					.srcMinCoord = {0,0,0},
					.srcMaxCoord = {srcExtent.width,srcExtent.height,1},
					.dstMinCoord = {0,0,0},
					.dstMaxCoord = {dstExtent.width,dstExtent.height,1},
					.layerCount = acquiredImage->getCreationParameters().arrayLayers,
					.srcBaseLayer = 0, // TODO
					.dstBaseLayer = 0,
					.srcMipLevel = 0 // TODO
				}};
				retval &= cmdbuf->blitImage(contents.image,blitSrcLayout,acquiredImage,blitDstLayout,regions,IGPUSampler::ETF_LINEAR);
			}

			// barrier after
			const image_barrier_t postBarriers[2] = {
				{
					.barrier = {
						// When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
						// or perform any visibility operations (as vkQueuePresentKHR performs automatic visibility operations).
						// To achieve this, the dstAccessMask member of the VkImageMemoryBarrier should be set to 0, and the dstStageMask parameter should be set to VK_PIPELINE_STAGE_2_NONE
						.dep = preBarriers[0].barrier.dep.nextBarrier(asset::PIPELINE_STAGE_FLAGS::NONE,asset::ACCESS_FLAGS::NONE)
					},
					.image = preBarriers[0].image,
					.subresourceRange = preBarriers[0].subresourceRange,
					.oldLayout = preBarriers[0].newLayout,
					.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
				},
				{
					.barrier = {
						.dep = preBarriers[1].barrier.dep.nextBarrier(contents.barrier.dep.dstStageMask,contents.barrier.dep.dstAccessMask),
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
						.otherQueueFamilyIndex = contents.barrier.otherQueueFamilyIndex
					},
					.image = preBarriers[1].image,
					.subresourceRange = preBarriers[1].subresourceRange,
					.oldLayout = preBarriers[1].newLayout,
					.newLayout = contents.newLayout
				}
			};
			depInfo.imgBarriers = postBarriers;
			retval &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);
#endif
			auto framebuffer = defaultFramebuffers[imageindex].get();

			const auto& framebufferParams = framebuffer->getCreationParameters();
			const IGPUCommandBuffer::SClearColorValue clearValue = {.float32={1.f,0.f,0.f,1.f}};
			const IGPUCommandBuffer::SRenderpassBeginInfo info = {
				.framebuffer = framebuffer,
				.colorClearValues = &clearValue,
				.depthStencilClearValues = nullptr,
				.renderArea = {.offset={0,0},.extent={framebufferParams.width,framebufferParams.height}}
			};
			cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
			cmdbuf->endRenderPass();

			return success ? asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT:asset::PIPELINE_STAGE_FLAGS::NONE;
		}
};

class HelloSwapchainApp final : public examples::SimpleWindowedApplication
{
		using base_t = examples::SimpleWindowedApplication;
		using clock_t = std::chrono::steady_clock;

	public:
		using base_t::base_t;
		
		// Will get called mid-initialization, via `filterDevices` between when the API Connection is created and Physical Device is chosen
		core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
		{
			// So let's create our Window and Surface then!
			if (!m_surface)
			{
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<nbl::video::IResizableSurface::ICallback>();
				params.width = 640;
				params.height = 480;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_INPUT_FOCUS|ui::IWindow::ECF_RESIZABLE|ui::IWindow::ECF_CAN_MAXIMIZE|ui::IWindow::ECF_CAN_MINIMIZE;
				params.windowCaption = "HelloSwapchainApp";
				auto window = m_winMgr->createWindow(std::move(params));
				// uncomment for some nasty testing of swapchain creation!
				//m_winMgr->minimize(window.get());
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CResizableSurface<CSwapchainResources>::create(video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(m_api), core::move_and_static_cast<ui::IWindowWin32>(window)));
			}
			return {{m_surface->getSurface()/*,EQF_NONE*/}};
		}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			// First create the resources that don't depend on a swapchain
			m_semaphore = m_device->createSemaphore(m_realFrameIx);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// We just live life in easy mode and have the Swapchain Creation Parameters get deduced from the surface.
			// We don't need any control over the format of the swapchain because we'll be only using Renderpasses this time!
			// TODO: improve the queue allocation/choice and allocate a dedicated presentation queue to improve responsiveness and race to present.
 			if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get())))
				return logFail("Failed to Create a Swapchain!");

			// When a swapchain gets recreated (resize or mode change) the number of images might change.
			// So we need the maximum possible number of in-flight resources so we never have too little.
			m_maxFramesInFlight = m_surface->getMaxFramesInFlight();

			// Normally you'd want to recreate these images whenever the swapchain is resized in some increment, like 64 pixels or something.
			// But I'm super lazy here and will just create "worst case sized images" and waste all the VRAM I can get.
			const auto dpyInfo = m_winMgr->getPrimaryDisplayInfo();
			for (auto i=0; i<m_maxFramesInFlight; i++)
			{
				IGPUImage::SCreationParams params = {};
				params = asset::IImage::SCreationParams{
					.type = IGPUImage::ET_2D,
					// you could be more clever and use the copy Triple Buffer to Swapchain as an opportunity to do a MSAA resolve or something
					.samples = IGPUImage::ESCF_1_BIT,
					// nice thing about having a triple buffer is that you don't need to do acrobatics to account for the formats available to the surface
					// you can transcode to the swapchain's format while copying, I actually recommend to do surface rotation, tonemapping and OETF application there.
					.format = asset::EF_R8G8B8A8_SRGB,
					.extent = {dpyInfo.resX,dpyInfo.resY,1},
					.mipLevels = 1,
					.arrayLayers = 1,
					.flags = IGPUImage::ECF_NONE,
					// in this example I'll be using a renderpass to clear the image, and then a blit to copy it to the swapchain
					.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT|IGPUImage::EUF_TRANSFER_SRC_BIT
				};
// TODO: concurrent sharing? No!
				auto& image = m_tripleBuffers[i];
				image = m_device->createImage(std::move(params));
				if (!image)
					return logFail("Failed to Create Triple Buffer Image!");

				// use dedicated allocations, we have plenty of memory
				auto allocation = m_device->allocate(image->getMemoryReqs(),image.get());
				if (!allocation.isValid())
					return logFail("Failed to allocate Device Memory for Image %d",i);
			}

			// This time we'll creaate all CommandBuffers from one CommandPool, to keep life simple. However the Pool must support individually resettable CommandBuffers
			// because they cannot be pre-recorded because the fraembuffers/swapchain images they use will change when a swapchain recreates.
			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data(),m_maxFramesInFlight},core::smart_refctd_ptr(m_logger)))
				return logFail("Failed to Create CommandBuffers!");

			// Help the CI a bit by providing a timeout option
			// TODO: @Hazardu maybe we should make a unified argument parser/handler for all examples?
			if (base_t::argv.size()>=3 && argv[1]=="-timeout_seconds")
				timeout = std::chrono::seconds(std::atoi(argv[2].c_str()));
			start = clock_t::now();
			return true;
		}

		// We do a very simple thing, and just keep on clearing the swapchain image to red and present
		void workLoopBody() override
		{
			// Can't reset a cmdbuffer before the previous use of commandbuffer is finished!
			if (m_realFrameIx>=m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{ 
						.semaphore = m_semaphore.get(),
						.value = m_realFrameIx+1-m_maxFramesInFlight
					}
				};
				m_device->blockForSemaphores(cmdbufDonePending);
			}
			
			// You explicitly should not use `getAcquireCount()` see the comment on `m_realFrameIx`
			const auto resourceIx = m_realFrameIx%m_maxFramesInFlight;

			// Now re-record it
			auto cmdbuf = m_cmdBufs[resourceIx].get();
			{
				cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
#if 0
				const auto& framebufferParams = framebuffer->getCreationParameters();
				const IGPUCommandBuffer::SClearColorValue clearValue = {.float32={1.f,0.f,0.f,1.f}};
				const IGPUCommandBuffer::SRenderpassBeginInfo info = {
					.framebuffer = framebuffer,
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = {.offset={0,0},.extent={framebufferParams.width,framebufferParams.height}}
				};
				cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				cmdbuf->endRenderPass();
#endif
				cmdbuf->end();
			}
			
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered = {
				.semaphore = m_semaphore.get(),
				.value = ++m_realFrameIx,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT // wait for renderpass/subpass to finish before handing over to blit
			};

			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{
				.cmdbuf = cmdbuf
			}};
			const IQueue::SSubmitInfo submitInfos[1] = {
				{
					.waitSemaphores = {}, // we already waited on the CPU to be able to reset the commandbuffer
					.commandBuffers = cmdbufs,
					.signalSemaphores = {&rendered,1}
				}
			};
			getGraphicsQueue()->submit(submitInfos);

			IResizableSurface::SPresentInfo presentInfo = {
				.source = m_tripleBuffers[resourceIx].get(),
				.wait = rendered,
				.frameResources = cmdbuf
			};
			m_surface->present(presentInfo);
		}

		//
		inline bool keepRunning() override
		{
			if (duration_cast<decltype(timeout)>(clock_t::now()-start)>timeout)
				return false;

			return !m_surface || !m_surface->irrecoverable();
		}

		virtual bool onAppTerminated() override
		{
			// We actually need to wait on a semaphore to finish the example nicely, otherwise we risk destroying a semaphore currently in use for a frame that hasn't finished yet.
			ISemaphore::SWaitInfo infos[1] = {
				{.semaphore=m_semaphore.get(),.value=m_realFrameIx}
			};
			m_device->blockForSemaphores(infos);

			// These are optional, the example will still work fine, but all the destructors would kick in (refcounts would drop to 0) AFTER we would have exited this function.
			m_semaphore = nullptr;
			std::fill_n(m_cmdBufs.data(),ISwapchain::MaxImages,nullptr);
			m_surface = nullptr;

			return base_t::onAppTerminated();
		}

	protected:
		// Just like in the HelloUI app we add a timeout
		std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point start;
		// In this example we have just one Window & Swapchain
		smart_refctd_ptr<CResizableSurface<CSwapchainResources>> m_surface;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		// At least two timelines must be used.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Use a separate counter to cycle through our resources because `getAcquireCount()` increases upon spontaneous resizes with immediate blit-presents 
		uint64_t m_realFrameIx : 59 = 0;
		// Maximum frames which can be simultaneously rendered
		uint64_t m_maxFramesInFlight : 5;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
		// Our own persistent images that don't get recreated with the swapchain
		std::array<smart_refctd_ptr<IGPUImage>,ISwapchain::MaxImages> m_tripleBuffers;
};

// define an entry point as always!
NBL_MAIN_FUNC(HelloSwapchainApp)