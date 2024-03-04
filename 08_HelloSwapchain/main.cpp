// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "../common/SimpleWindowedApplication.hpp"

//
#include "nbl/video/surface/CSurfaceVulkan.h"
<<<<<<< HEAD

#include "../common/BasicMultiQueueApplication.hpp"

namespace nbl::examples
{
// Virtual Inheritance because apps might end up doing diamond inheritance
class WindowedApplication : public virtual BasicMultiQueueApplication
{
		using base_t = BasicMultiQueueApplication;

	public:
		using base_t::base_t;

		// We inherit from an application that tries to find Graphics and Compute queues
		// because applications with presentable images often want to perform Graphics family operations
		virtual bool isComputeOnly() const {return false;}

		virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable() override
		{
			auto retval = base_t::getAPIFeaturesToEnable();
			// We only support one swapchain mode, surface, the other one is Display which we have not implemented yet.
			retval.swapchainMode = video::E_SWAPCHAIN_MODE::ESM_SURFACE;
			return retval;
		}

		// New function, we neeed to know about surfaces to create ahead of time
		virtual core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const = 0;

		// We have a very simple heuristic, the device must be able to render to all windows!
		// (want to make something more complex? you're on your own!)
		virtual void filterDevices(core::set<video::IPhysicalDevice*>& physicalDevices) const
		{
			base_t::filterDevices(physicalDevices);

			video::SPhysicalDeviceFilter deviceFilter = {};
			
			auto surfaces = getSurfaces();
			deviceFilter.requiredSurfaceCompatibilities = {surfaces};

			return deviceFilter(physicalDevices);
		}
		
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

		#ifdef _NBL_PLATFORM_WINDOWS_
			m_winMgr = nbl::ui::IWindowManagerWin32::create();
		#else
			#error "Unimplemented!"
		#endif
			return true;
		}

		core::smart_refctd_ptr<ui::IWindowManager> m_winMgr;
};


// Before we get onto creating a window, we need to discuss how Nabla handles input, clipboards and cursor control
class IWindowClosedCallback : public virtual nbl::ui::IWindow::IEventCallback
{
	public:
		IWindowClosedCallback() : m_gotWindowClosedMsg(false) {}

		// unless you create a separate callback per window, both will "trip" this condition
		bool windowGotClosed() const {return m_gotWindowClosedMsg;}

	private:
		bool onWindowClosed_impl() override
		{
			m_gotWindowClosedMsg = true;
			return true;
		}

		bool m_gotWindowClosedMsg;
};

// We inherit from an application that tries to find Graphics and Compute queues
// because applications with presentable images often want to perform Graphics family operations
// Virtual Inheritance because apps might end up doing diamond inheritance
class SingleNonResizableWindowApplication : public virtual WindowedApplication
{
		using base_t = WindowedApplication;

	public:
		using base_t::base_t;

		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_window = m_winMgr->createWindow(getWindowCreationParams());
			m_surface = video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(m_api),core::smart_refctd_ptr_static_cast<ui::IWindowWin32>(m_window));
			return true;
		}

		virtual core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const
		{
			return {{m_surface.get()/*,EQF_NONE*/}};
		}

		virtual bool keepRunning() override
		{
			if (!m_window || reinterpret_cast<const IWindowClosedCallback*>(m_window->getEventCallback())->windowGotClosed())
				return false;

			return true;
		}

	protected:
		virtual ui::IWindow::SCreationParams getWindowCreationParams() const
		{
			ui::IWindow::SCreationParams params = {};
			params.callback = core::make_smart_refctd_ptr<IWindowClosedCallback>();
			params.width = 640;
			params.height = 480;
			params.x = 32;
			params.y = 32;
			params.flags = ui::IWindow::ECF_NONE;
			params.windowCaption = "SingleNonResizableWindowApplication";
			return params;
		}

		core::smart_refctd_ptr<ui::IWindow> m_window;
		core::smart_refctd_ptr<video::ISurfaceVulkan> m_surface;
};
}

=======
>>>>>>> vulkan_1_3
#include "nbl/video/CVulkanSwapchain.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace video;

// We don't do anything weird in this example like present from mips of different layers
constexpr IGPUImage::SSubresourceRange TripleBufferUsedSubresourceRange = {
	.aspectMask = IGPUImage::EAF_COLOR_BIT,
	.baseMipLevel = 0,
	.levelCount = 1,
	.baseArrayLayer = 0,
	.layerCount = 1
};

//
class CSwapchainResources final : public IResizableSurface::ISwapchainResources
{
	public:
		// Because we blit to the swapchain image asynchronously, we need a queue which can not only present but also perform graphics commands.
		// If we for example used a compute shader to tonemap and MSAA resolve, we'd request the COMPUTE_BIT here. 
		constexpr static inline IQueue::FAMILY_FLAGS RequiredQueueFlags = IQueue::FAMILY_FLAGS::GRAPHICS_BIT;

	protected:		
		inline asset::PIPELINE_STAGE_FLAGS tripleBufferPresent(IGPUCommandBuffer* cmdbuf, const IResizableSurface::SPresentSource& source, const uint8_t imageIndex, const uint32_t qFamToAcquireSrcFrom) override
		{
			bool success = true;
			auto acquiredImage = getImage(imageIndex);

			const auto blitDstLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL;
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};

			// barrier before to transition the swapchain image layout
			using image_barrier_t = decltype(depInfo.imgBarriers)::element_type;
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
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED, // I do not care about previous contents of the swapchain
					.newLayout = blitDstLayout
				},
				{
					.barrier = {
						.dep = {
							// when acquiring ownership the source masks don't matter
							.srcStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
							.srcAccessMask = asset::ACCESS_FLAGS::NONE,
							.dstStageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT,
							.dstAccessMask = asset::ACCESS_FLAGS::TRANSFER_READ_BIT
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
						.otherQueueFamilyIndex = qFamToAcquireSrcFrom
					},
					.image = source.image,
					.subresourceRange = TripleBufferUsedSubresourceRange
					// no layout transition, already in the correct layout for the blit
				}
			};
			// We only barrier the source image if we need to acquire ownership, otherwise thanks to Timeline Semaphores all sync is good
			depInfo.imgBarriers = {preBarriers,qFamToAcquireSrcFrom!=IQueue::FamilyIgnored ? 2ull:1ull};
			success &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);
		
			// TODO: Implement scaling modes other than a plain STRETCH, and allow for using subrectangles of the initial contents
			{
				const auto srcOffset = source.rect.offset;
				const auto srcExtent = source.rect.extent;
				const auto dstExtent = acquiredImage->getCreationParameters().extent;
				const IGPUCommandBuffer::SImageBlit regions[1] = {{
					.srcMinCoord = {static_cast<uint32_t>(srcOffset.x),static_cast<uint32_t>(srcOffset.y),0},
					.srcMaxCoord = {srcExtent.width,srcExtent.height,1},
					.dstMinCoord = {0,0,0},
					.dstMaxCoord = {dstExtent.width,dstExtent.height,1},
					.layerCount = acquiredImage->getCreationParameters().arrayLayers,
					.srcBaseLayer = 0,
					.dstBaseLayer = 0,
					.srcMipLevel = 0
				}};
				success &= cmdbuf->blitImage(source.image,IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL,acquiredImage,blitDstLayout,regions,IGPUSampler::ETF_LINEAR);
			}

			// Barrier after, note that I don't care about preserving the contents of the Triple Buffer when the Render queue starts writing to it again.
			// Therefore no ownership release, and no layout transition.
			const image_barrier_t postBarrier[1] = {
				{
					.barrier = {
						// When transitioning the image to VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR or VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, there is no need to delay subsequent processing,
						// or perform any visibility operations (as vkQueuePresentKHR performs automatic visibility operations).
						// To achieve this, the dstAccessMask member of the VkImageMemoryBarrier should be set to 0, and the dstStageMask parameter should be set to VK_PIPELINE_STAGE_2_NONE
						.dep = preBarriers[0].barrier.dep.nextBarrier(asset::PIPELINE_STAGE_FLAGS::NONE,asset::ACCESS_FLAGS::NONE)
					},
					.image = preBarriers[0].image,
					.subresourceRange = preBarriers[0].subresourceRange,
					.oldLayout = blitDstLayout,
					.newLayout = IGPUImage::LAYOUT::PRESENT_SRC
				}
			};
			depInfo.imgBarriers = postBarrier;
			success &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);

			return success ? asset::PIPELINE_STAGE_FLAGS::BLIT_BIT:asset::PIPELINE_STAGE_FLAGS::NONE;
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

			// The nice thing about having a triple buffer is that you don't need to do acrobatics to account for the formats available to the surface.
			// You can transcode to the swapchain's format while copying, and I actually recommend to do surface rotation, tonemapping and OETF application there.
			const auto format = asset::EF_R8G8B8A8_SRGB;
			// Could be more clever and use the copy Triple Buffer to Swapchain as an opportunity to do a MSAA resolve or something
			const auto samples = IGPUImage::ESCF_1_BIT;

			// Create the renderpass
			{
				const IGPURenderpass::SCreationParams::SColorAttachmentDescription colorAttachments[] = {
					{{
						.format = format,
						.samples = samples,
						.mayAlias = false,
						.loadOp = IGPURenderpass::LOAD_OP::CLEAR,
						.storeOp = IGPURenderpass::STORE_OP::STORE,
						.initialLayout = IGPUImage::LAYOUT::UNDEFINED, // because we clear we don't care about contents when we grab the triple buffer img again
						.finalLayout = IGPUImage::LAYOUT::TRANSFER_SRC_OPTIMAL // put it already in the correct layout for the blit operation
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
				m_renderpass = m_device->createRenderpass(params);
				if (!m_renderpass)
					return logFail("Failed to Create a Renderpass!");
			}

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
				auto& image = m_tripleBuffers[i];
				{
					IGPUImage::SCreationParams params = {};
					params = asset::IImage::SCreationParams{
						.type = IGPUImage::ET_2D,
						.samples = samples,
						.format = format,
						.extent = {dpyInfo.resX,dpyInfo.resY,1},
						.mipLevels = 1,
						.arrayLayers = 1,
						.flags = IGPUImage::ECF_NONE,
						// in this example I'll be using a renderpass to clear the image, and then a blit to copy it to the swapchain
						.usage = IGPUImage::EUF_RENDER_ATTACHMENT_BIT|IGPUImage::EUF_TRANSFER_SRC_BIT
					};
					image = m_device->createImage(std::move(params));
					if (!image)
						return logFail("Failed to Create Triple Buffer Image!");

					// use dedicated allocations, we have plenty of allocations left, even on Win32
					if (!m_device->allocate(image->getMemoryReqs(),image.get()).isValid())
						return logFail("Failed to allocate Device Memory for Image %d",i);
				}

				// create framebuffers for the images
				{
					auto imageView = m_device->createImageView({
						.flags = IGPUImageView::ECF_NONE,
						// give it a Transfer SRC usage flag so we can transition to the Tranfer SRC layout with End Renderpass
						.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT|IGPUImage::EUF_TRANSFER_SRC_BIT,
						.image = core::smart_refctd_ptr(image),
						.viewType = IGPUImageView::ET_2D,
						.format = format
					});
					const auto& imageParams = image->getCreationParameters();
					IGPUFramebuffer::SCreationParams params = {{
						.renderpass = core::smart_refctd_ptr(m_renderpass),
						.depthStencilAttachments = nullptr,
						.colorAttachments = &imageView.get(),
						.width = imageParams.extent.width,
						.height = imageParams.extent.height,
						.layers = imageParams.arrayLayers
					}};
					m_framebuffers[i] = m_device->createFramebuffer(std::move(params));
					if (!m_framebuffers[i])
						return logFail("Failed to Create a Framebuffer for Image %d",i);
				}
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
				if (m_device->blockForSemaphores(cmdbufDonePending)!=ISemaphore::WAIT_RESULT::SUCCESS)
					return;
			}

			// Predict size of next render, and bail if nothing to do
			const auto currentSwapchainExtent = m_surface->getCurrentExtent();
			if (currentSwapchainExtent.width*currentSwapchainExtent.height<=0)
				return;
			// The extent of the swapchain might change between now and `present` but the blit should adapt nicely
			const VkRect2D currentRenderArea = {.offset={0,0},.extent=currentSwapchainExtent};
			
			// You explicitly should not use `getAcquireCount()` see the comment on `m_realFrameIx`
			const auto resourceIx = m_realFrameIx%m_maxFramesInFlight;

			// We will be using this command buffer to produce the frame
			auto frame = m_tripleBuffers[resourceIx].get();
			auto cmdbuf = m_cmdBufs[resourceIx].get();

			// Now re-record it
			bool willSubmit = true;
			{
				willSubmit &= cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);

				const IGPUCommandBuffer::SClearColorValue clearValue = {.float32={1.f,0.f,0.f,1.f}};
				const IGPUCommandBuffer::SRenderpassBeginInfo info = {
					.framebuffer = m_framebuffers[resourceIx].get(),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = currentRenderArea
				};
				willSubmit &= cmdbuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				willSubmit &= cmdbuf->endRenderPass();

				// If the Rendering and Blit/Present Queues don't come from the same family we need to transfer ownership, because we need to preserve contents between them.
				if (cmdbuf->getQueueFamilyIndex()!=m_surface->getAssignedQueue()->getFamilyIndex())
				{
					IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = {};
					const decltype(depInfo.imgBarriers)::element_type barrier[2] = {{
						.barrier = {
							.dep = {
								.srcStageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT,
								.srcAccessMask = asset::ACCESS_FLAGS::COLOR_ATTACHMENT_WRITE_BIT,
								// for a Queue Family Ownership Release the destination masks are irrelevant
								.dstStageMask = asset::PIPELINE_STAGE_FLAGS::NONE,
								.dstAccessMask = asset::ACCESS_FLAGS::NONE
							}
						},
						.image = frame,
						.subresourceRange = TripleBufferUsedSubresourceRange
						// there will be no layout transition, already done by the Renderpass End
					}};
					willSubmit &= cmdbuf->pipelineBarrier(asset::EDF_NONE,depInfo);
				}

				willSubmit &= cmdbuf->end();
			}
			
			// submit and present under a mutex ASAP
			if (willSubmit)
			{
				// We will signal a semaphore in the rendering queue, and await it with the presentation/blit queue
				const IQueue::SSubmitInfo::SSemaphoreInfo rendered = {
					.semaphore = m_semaphore.get(),
					.value = m_realFrameIx+1,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT // wait for renderpass/subpass to finish before handing over to blit
				};
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{
					.cmdbuf = cmdbuf
				}};
				// We need to wait on previous triple buffer blits/presents from our source image to complete
				auto* pBlitWaitValue = m_blitWaitValues.data()+resourceIx;
				auto swapchainLock = m_surface->pseudoAcquire(pBlitWaitValue);
				const IQueue::SSubmitInfo::SSemaphoreInfo blitted = {
					.semaphore = m_surface->getPresentSemaphore(),
					.value = pBlitWaitValue->load(),
					.stageMask = asset::PIPELINE_STAGE_FLAGS::BLIT_BIT // same mask as returned from tripleBufferPresent
				};
				const IQueue::SSubmitInfo submitInfos[1] = {
					{
						.waitSemaphores = {&blitted,1},
						.commandBuffers = cmdbufs,
						.signalSemaphores = {&rendered,1}
					}
				};
				if (getGraphicsQueue()->submit(submitInfos)!=IQueue::RESULT::SUCCESS)
					return;
				m_realFrameIx++;

				// only present if there's successful content to show
				const IResizableSurface::SPresentInfo presentInfo = {
					.source = {.image=frame,.rect=currentRenderArea},
					.wait = rendered,
					.pPresentSemaphoreWaitValue = pBlitWaitValue,
					// The Graphics Queue will be the the most recent owner just before it releases ownership
					.mostRecentFamilyOwningSource = cmdbuf->getQueueFamilyIndex(),
					.frameResources = cmdbuf
				};
				m_surface->present(std::move(swapchainLock),presentInfo);
			}
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
				{.semaphore=m_semaphore.get(),.value=m_realFrameIx},
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
		// We'll write to the Triple Buffer with a Renderpass
		core::smart_refctd_ptr<IGPURenderpass> m_renderpass = {};
		// These are atomic counters where the Surface lets us know what's the latest Blit timeline semaphore value which will be signalled on the resource
		std::array<std::atomic_uint64_t,ISwapchain::MaxImages> m_blitWaitValues;
		// Enough Command Buffers and other resources for all frames in flight!
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
		// Our own persistent images that don't get recreated with the swapchain
		std::array<smart_refctd_ptr<IGPUImage>,ISwapchain::MaxImages> m_tripleBuffers;
		// Resources derived from the images
		std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> m_framebuffers = {};
};

// define an entry point as always!
NBL_MAIN_FUNC(HelloSwapchainApp)