// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "../common/SimpleWindowedApplication.hpp"


//
#include "nbl/video/surface/CSurfaceVulkan.h"
#include "nbl/video/CVulkanSwapchain.h"

// TODO: move to a common header
namespace nbl::video
{

// Simple wrapper class intended for Single Threaded usage
class SimpleResizableSurface : public ISimpleManagedSurface
{
	public:
		// Simple callback to facilitate detection of window being closed
		class ICallback : public ISimpleManagedSurface::ICallback
		{
			protected:
				inline virtual bool onWindowResized_impl(uint32_t w, uint32_t h) override
				{
#if 0
					std::unique_lock guard(m_swapchainPtrMutex);
					// recreate the swapchain with a new size
					if (m_swapchainResources.recreateSwapchain(m_surface.get(),const_cast<ILogicalDevice*>(m_queue->getOriginDevice())))
						break;
					// wait for last presented frame to finish rendering
					block(m_lastPresentWait);
					immediateImagePresent(m_queue,);
#endif
					return true;
				}
		};

		// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `ICallback` declared just above
		static inline core::smart_refctd_ptr<SimpleResizableSurface> create(core::smart_refctd_ptr<ISurface>&& _surface)
		{
			if (!_surface)
				return nullptr;

			auto _window = _surface->getWindow();
			if (!_window)
				return nullptr;

			auto cb = dynamic_cast<ICallback*>(_window->getEventCallback());
			if (!cb)
				return nullptr;

			return core::smart_refctd_ptr<SimpleResizableSurface>(new SimpleResizableSurface(std::move(_surface),cb),core::dont_grab);
		}

		//
		class CSwapchainResources final : public ISimpleManagedSurface::ISwapchainResources
		{
			friend class SimpleResizableSurface;
				inline void setStatus(const STATUS _status) {status=_status;}

			public:
				// Last parameter is for when you want to recreate and immediately present with the contents of some `IGPUImage`
				bool recreateSwapchain(ISurface* surface, ILogicalDevice* device, IGPUImage* initPresent=nullptr, IQueue* blitPresentQueue=nullptr);

				// Because the surface can start minimized (extent={0,0}) we might not be able to create the swapchain right away, so store creation parameters until we can create it.
				ISwapchain::SSharedCreationParams sharedParams = {};
				// If these get used, they will indirectly find their way into the `frameResources` argument of the `present` method.
				core::smart_refctd_ptr<IGPURenderpass> defaultRenderpass = {};
				std::array<core::smart_refctd_ptr<IGPUFramebuffer>, ISwapchain::MaxImages> defaultFramebuffers = {};

			protected:
				inline void invalidate_impl() override
				{
					// Framebuffers hold onto the renderpass they were created from, and swapchain images (swapchain itself indirectly)
					std::fill_n(defaultFramebuffers.data(), ISwapchain::MaxImages, nullptr);
					defaultRenderpass = nullptr;
				}

				inline bool onCreateSwapchain() override
				{
					auto fail = [&]()->bool{invalidate_impl();return false;};

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
							return fail();
					}

					// create framebuffers for the images
					for (auto i=0u; i<swapchain->getImageCount(); i++)
					{
						auto imageView = device->createImageView({
							.flags = IGPUImageView::ECF_NONE,
							.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
							.image = swapchain->createImage(i),
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
							return fail();
					}

					return true;
				}
		};
		inline const CSwapchainResources& getSwapchainResources() const {return m_swapchainResources;}

	protected:
		using ISimpleManagedSurface::ISimpleManagedSurface;

		inline ISimpleManagedSurface::ISwapchainResources& getSwapchainResources() override {return m_swapchainResources;}
		inline bool init_impl(IQueue* queue, const ISwapchain::SSharedCreationParams& sharedParams={}) override
		{
			switch (getSurface()->getAPIType()) // TODO: move to concrete class
			{
				case EAT_VULKAN:
					break;
				default:
					return false;
			}

			m_swapchainResources.sharedParams = sharedParams;
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			if (!m_swapchainResources.sharedParams.deduce(device->getPhysicalDevice(),getSurface()))
				return false;

			m_swapchainResources.setStatus(ISwapchainResources::STATUS::NOT_READY);
			return true;
		}

		inline bool handleNotReady() override
		{
			return m_swapchainResources.recreateSwapchain(getSurface(),const_cast<ILogicalDevice*>(getAssignedQueue()->getOriginDevice()));
		}
		inline int8_t handleOutOfDate() override
		{
			// try again, will re-create swapchain
			return acquireNextImage();
		}

		CSwapchainResources m_swapchainResources = {};
};
// TODO: move comment
// Window/Surface got closed, but won't actually disappear UNTIL the swapchain gets dropped,
// which is outside of our control here as there is a nice chain of lifetimes of:
// `ExternalCmdBuf -via usage of-> Swapchain Image -memory provider-> Swapchain -created from-> Window/Surface`
// Only when the last user of the swapchain image drops it, will the window die.
bool SimpleResizableSurface::CSwapchainResources::recreateSwapchain(ISurface* surface, ILogicalDevice* device, IGPUImage* initPresent, IQueue* blitPresentQueue)
{
	auto fail = [&]()->bool {becomeIrrecoverable(); return false;};

	// create new swapchain
	{
		// dont assign straight to `swapchain` because of complex refcounting and cycles
		core::smart_refctd_ptr<ISwapchain> newSwapchain;
		// Question: should we re-query the supported queues, formats, present modes, etc. just-in-time??
		if (swapchain ? swapchain->deduceRecreationParams(sharedParams):sharedParams.deduce(device->getPhysicalDevice(),surface))
		{
			// super special case, we can't re-create the swapchain 
			if (sharedParams.width==0 || sharedParams.height==0)
			{
				// we need to keep the old-swapchain around, but can drop the rest
				invalidate();
				return false;
			}
			// now we have to succeed in creation
			if (swapchain)
				newSwapchain = swapchain->recreate(sharedParams);
			else
			{
				ISwapchain::SCreationParams params = {
					.surface = core::smart_refctd_ptr<ISurface>(surface),
					.surfaceFormat = {},
					.sharedParams = sharedParams
					// we're not going to support concurrent sharing in this simple class
				};
				if (params.deduceFormat(device->getPhysicalDevice()))
					newSwapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device),std::move(params));
			}
		}

		if (!newSwapchain)
			return fail();
		swapchain = std::move(newSwapchain);
	}

	if (initPresent)
	{
#if 0
		if (!blitPresentQueue)
			blitPresentQueue = pickQueue(BLIT|PRESENT);

		auto semaphore = device->createSemaphore(0);

		const IQueue::SSubmitInfo::SSemaphoreInfo acquired[1] = {
			{
				.semaphore=semaphore.get(),
				.value=1
			}
		};

		uint32_t imageIndex;
		switch (m_swapchainResources.swapchain->acquireNextImage({.queue=blitPresentQueue,.signalSemaphores=acquired},&imageIndex))
		{
			case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
			case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
				break;
			case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
			case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
				assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
				break;
			case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
				// try again, will re-create swapchain
				invalidate();
				return false;
			default:
				return fail();
		}
					
		switch (m_swapchainResources.swapchain->present({.queue=blitPresentQueue,.imgIndex=imageIndex,.waitSemaphores=blitted},std::move(frameResources)))
		{
			case ISwapchain::PRESENT_RESULT::SUBOPTIMAL: [[fallthrough]];
			case ISwapchain::PRESENT_RESULT::SUCCESS:
				return true;
			case ISwapchain::PRESENT_RESULT::OUT_OF_DATE:
				m_swapchainResources.invalidate();
				break;
			default:
				m_swapchainResources.becomeIrrecoverable();
				break;
		}
#endif
	}

	onCreateSwapchain();

	status = STATUS::USABLE;
	return true;
}

}

using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;
using namespace video;

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
				params.callback = core::make_smart_refctd_ptr<nbl::video::SimpleResizableSurface::ICallback>();
				params.width = 640;
				params.height = 480;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_INPUT_FOCUS|ui::IWindow::ECF_RESIZABLE|ui::IWindow::ECF_CAN_MAXIMIZE|ui::IWindow::ECF_CAN_MINIMIZE;
				params.windowCaption = "HelloSwapchainApp";
				auto window = m_winMgr->createWindow(std::move(params));
				// uncomment for some nasty testing of swapchain creation!
				//m_winMgr->minimize(window.get());
				const_cast<core::smart_refctd_ptr<SimpleResizableSurface>&>(m_surface) = SimpleResizableSurface::create(
					video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(m_api),core::move_and_static_cast<ui::IWindowWin32>(window))
				);
			}
			return {{m_surface->getSurface()/*,EQF_NONE*/}};
		}

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			// First create the resources that don't depend on a swapchain
			m_semaphore = m_device->createSemaphore(0);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// When a swapchain gets recreated (resize or mode change) the number of images might change.
			// So we need the maximum possible number of in-flight resources so we never have too little.
			m_maxFramesInFlight = ISwapchain::MaxImages;// m_surface->getMaxFramesInFlight();

			// This time we'll creaate all CommandBuffers from one CommandPool, to keep life simple. However the Pool must support individually resettable CommandBuffers
			// because they cannot be pre-recorded because the fraembuffers/swapchain images they use will change when a swapchain recreates.
			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data(),m_maxFramesInFlight},core::smart_refctd_ptr(m_logger)))
				return logFail("Failed to Create CommandBuffers!");

			// We just live life in easy mode and have the Swapchain Creation Parameters get deduced from the surface.
			// We don't need any control over the format of the swapchain because we'll be only using Renderpasses this time!
 			if (!m_surface || !m_surface->init(m_surface->pickQueue(m_device.get())))
				return logFail("Failed to Create a Swapchain!");

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
			// To proceed, you need to acquire to know what swapchain image to use.
			const int8_t acquiredImageIx = m_surface->acquireNextImage();
			// We might fail to acquire an image, e.g. because of a zero-sized window (minimized on win32)
			if (acquiredImageIx <0)
				return;

			const auto nextFrameIx = m_surface->getAcquireCount();

			// You explicitly should not use `m_cmdBufs[acquiredImageIx]` because after swapchain re-creation you might
			// end up acquiring an image index colliding with old swapchain images not yet presented.
			auto cmdbuf = m_cmdBufs[(nextFrameIx-1)%m_maxFramesInFlight].get();

			// Can't reset a cmdbuffer before the previous use of commandbuffer is finished!
			if (nextFrameIx>m_maxFramesInFlight)
			{
				const ISemaphore::SWaitInfo cmdbufDonePending[] = {
					{ 
						.semaphore = m_semaphore.get(),
						.value = nextFrameIx-m_maxFramesInFlight
					}
				};
				m_device->blockForSemaphores(cmdbufDonePending);
			}

			// Now re-record it
			{
				const SimpleResizableSurface* surface = m_surface.get(); // to get around some missing const conversions of smart_refctd_ptr
				auto framebuffer = surface->getSwapchainResources().defaultFramebuffers[acquiredImageIx].get();

				cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
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
				cmdbuf->end();
			}

			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{
				.cmdbuf = cmdbuf
			}};
			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[1] = {{
				.semaphore = m_semaphore.get(),
				.value = nextFrameIx,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT // wait for renderpass/subpass to finish before handing over to Present
			}};
			// acquired swapchain image so can start rendering into it and present it
			const IQueue::SSubmitInfo::SSemaphoreInfo acquired[] = {
				{
					.semaphore = m_surface->getAcquireSemaphore(),
					.value = m_surface->getAcquireCount(),
					.stageMask = asset::PIPELINE_STAGE_FLAGS::NONE // presentation engine usage isn't a stage
				}
			};
			const IQueue::SSubmitInfo submitInfos[1] = {
				{
					.waitSemaphores = acquired,
					.commandBuffers = cmdbufs,
					.signalSemaphores = rendered
				}
			};
			getGraphicsQueue()->submit(submitInfos);
			m_surface->present(static_cast<uint8_t>(acquiredImageIx),rendered,smart_refctd_ptr<IGPUCommandBuffer>(cmdbufs->cmdbuf));
		}

		//
		inline bool keepRunning() override
		{
			if (duration_cast<decltype(timeout)>(clock_t::now()-start)>timeout)
				return false;

			const SimpleResizableSurface* surface = m_surface.get(); // to get around some missing const conversions of smart_refctd_ptr
			return surface && surface->getSwapchainResources().getStatus()!=ISimpleManagedSurface::ISwapchainResources::STATUS::IRRECOVERABLE;
		}

		virtual bool onAppTerminated() override
		{
			// We actually need to wait on a semaphore to finish the example nicely, otherwise we risk destroying a semaphore currently in use for a frame that hasn't finished yet.
			ISemaphore::SWaitInfo infos[1] = {
				{.semaphore=m_semaphore.get(),.value=m_surface->getAcquireCount()}
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
		smart_refctd_ptr<SimpleResizableSurface> m_surface;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		// At least two timelines must be used.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Enough Command Buffers for all frames in flight!
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
		uint8_t m_maxFramesInFlight;
};

// define an entry point as always!
NBL_MAIN_FUNC(HelloSwapchainApp)