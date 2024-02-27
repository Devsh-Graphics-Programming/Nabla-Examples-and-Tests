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
class SimpleResizableSurface : public core::IReferenceCounted
{
	public:
		// Simple callback to facilitate detection of window being closed
		class ICallback : public ui::IWindow::IEventCallback
		{
			public:
				inline ICallback() : m_gotWindowClosedMsg(false) {}

				// unless you create a separate callback per window, both will "trip" this condition
				inline bool windowGotClosed() const {return m_gotWindowClosedMsg;}

			private:
				inline bool onWindowResized_impl(uint32_t w, uint32_t h) override
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

				inline bool onWindowClosed_impl() override
				{
					m_gotWindowClosedMsg = true;
					return true;
				}

				bool m_gotWindowClosedMsg;
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
		inline const ISurface* getSurface() const {return m_surface.get();}

		// A small utility for the boilerplate
		inline uint8_t pickQueueFamily(ILogicalDevice* device) const
		{
			uint8_t qFam = 0u;
			for (; qFam<ILogicalDevice::MaxQueueFamilies; qFam++)
			if (device->getQueueCount(qFam) && m_surface->isSupportedForPhysicalDevice(device->getPhysicalDevice(),qFam))
				break;
			return qFam;
		}

		// Just pick the first queue within the first compatible family
		inline IQueue* pickQueue(ILogicalDevice* device) const
		{
			return device->getThreadSafeQueue(pickQueueFamily(device),0);
		}

		// We need to defer the swapchain creation till the Physical Device is chosen and Queues are created together with the Logical Device
		inline bool init(IQueue* queue, const ISwapchain::SSharedCreationParams& sharedParams={})
		{
			m_swapchainResources.status = ISwapchainResources::STATUS::IRRECOVERABLE;
			if (!queue)
				return false;

			switch (m_surface->getAPIType())
			{
				case EAT_VULKAN:
					break;
				default:
					return false;
			}

			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			if (!m_acquireSemaphore)
			{
				m_acquireSemaphore = device->createSemaphore(0u);
				if (!m_acquireSemaphore)
					return false;
			}

			m_swapchainResources.sharedParams = sharedParams;
			if (!m_swapchainResources.sharedParams.deduce(device->getPhysicalDevice(),m_surface.get()))
				return false;

			m_swapchainResources.status = ISwapchainResources::STATUS::NOT_READY;
			m_queue = queue;			
			return true;
		}

		//
		inline IQueue* getAssignedQueue() {return m_queue;}

		// An interesting solution to the "Frames In Flight", our tiny wrapper class will have its own Timeline Semaphore incrementing with each acquire, and thats it.
		inline uint64_t getAcquireCount() {return m_acquireCount;}
		inline ISemaphore* getAcquireSemaphore() {return m_acquireSemaphore.get();}

		// A class to hold resources that can die/invalidate spontaneously
		struct ISwapchainResources : core::InterfaceUnmovable
		{
			// If window gets minimized on some platforms or more rarely if it gets resized weirdly, the render area becomes 0 so its impossible to recreate a swapchain.
			// So we need to defer the swapchain re-creation until we can resize to a valid extent.
			enum class STATUS : int8_t
			{
				IRRECOVERABLE = -1,
				USABLE,
				NOT_READY = 1
			};

			core::smart_refctd_ptr<ISwapchain> swapchain = {};
			// If these get used, they should indirectly find their way into the `frameResources` argument of the `present` method.
			core::smart_refctd_ptr<IGPURenderpass> defaultRenderpass = {};
			std::array<core::smart_refctd_ptr<IGPUFramebuffer>,ISwapchain::MaxImages> defaultFramebuffers = {};
			// very important variable
			STATUS status = STATUS::NOT_READY;
		};
		inline const ISwapchainResources& getSwapchainResources() const {return m_swapchainResources;}

		// RETURNS: Negative on failure, otherwise its the acquired image's index.
		inline int8_t acquireNextImage()
		{
			using status_t = ISwapchainResources::STATUS;
			switch (m_swapchainResources.status)
			{
				case status_t::NOT_READY:
					if (m_swapchainResources.recreateSwapchain(m_surface.get(),const_cast<ILogicalDevice*>(m_queue->getOriginDevice())))
						break;
					[[fallthrough]];
				case status_t::IRRECOVERABLE:
					return -1;
				default:
					break;
			}

			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfos[1] = {
				{
					.semaphore=m_acquireSemaphore.get(),
					.value=m_acquireCount+1
				}
			};

			uint32_t imageIndex;
			// We don't support resizing (swapchain recreation) in this example, so a failure to acquire is a failure to keep running
			switch (m_swapchainResources.swapchain->acquireNextImage({.queue=m_queue,.signalSemaphores=signalInfos},&imageIndex))
			{
				case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
				case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
					// the semaphore will only get signalled upon a successful acquire
					m_acquireCount++;
					return imageIndex;
				case ISwapchain::ACQUIRE_IMAGE_RESULT::TIMEOUT: [[fallthrough]];
				case ISwapchain::ACQUIRE_IMAGE_RESULT::NOT_READY: // don't throw our swapchain away just because of a timeout XD
					assert(false); // shouldn't happen though cause we use uint64_t::max() as the timeout
					break;
				case ISwapchain::ACQUIRE_IMAGE_RESULT::OUT_OF_DATE:
					// try again, will re-create swapchain
					return acquireNextImage();
				default:
					break;
			}
			m_swapchainResources.becomeIrrecoverable();
			return -1;
		}

		// Frame Resources are not optional!
		inline bool present(const uint8_t imageIndex, const std::span<const IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores, core::smart_refctd_ptr<core::IReferenceCounted>&& frameResources)
		{
			if (m_swapchainResources.status!=ISwapchainResources::STATUS::USABLE)
				return false;

			const ISwapchain::SPresentInfo info = {
				.queue = m_queue,
				.imgIndex = imageIndex,
				.waitSemaphores = waitSemaphores
			};
			switch (m_swapchainResources.swapchain->present(info,std::move(frameResources)))
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
			return false;
		}

	protected:
		inline SimpleResizableSurface(core::smart_refctd_ptr<ISurface>&& _surface, ICallback* _cb) : m_surface(std::move(_surface)), m_cb(_cb) {}
		inline ~SimpleResizableSurface()
		{
			// just to avoid deadlocks due to circular refcounting
			m_swapchainResources.becomeIrrecoverable();
		}

		core::smart_refctd_ptr<ISurface> m_surface;
		ICallback* m_cb;
		IQueue* m_queue;
		core::smart_refctd_ptr<ISemaphore> m_acquireSemaphore;
		// You don't want to use `m_swapchainResources.swapchain->getAcquireCount()` because it resets when swapchain gets recreated
		uint64_t m_acquireCount = 0;
		// these can die spontaneously and get recreated
		struct CSwapchainResources : ISwapchainResources
		{
			inline void invalidate()
			{
				if (status==STATUS::NOT_READY)
					return;

				// Framebuffers hold onto the renderpass they were created from, and swapchain images (swapchain itself indirectly)
				std::fill_n(defaultFramebuffers.data(),ISwapchain::MaxImages,nullptr);
				defaultRenderpass = nullptr;

				status = STATUS::NOT_READY;
			}

			inline void becomeIrrecoverable()
			{
				if (status==STATUS::IRRECOVERABLE)
					return;

				// Want to nullify things in an order that leads to fastest drops (if possible) and shallowest callstacks when refcounting
				invalidate();

				// We need to call this method manually to make sure resources latched on swapchain images are dropped and cycles broken, otherwise its
				// EXTERMELY LIKELY (if you don't reset CommandBuffers) that you'll end up with a circular reference:
				// `CommandBuffer -> SC Image[i] -> Swapchain -> FrameResource[i] -> CommandBuffer`
				// and a memory leak of: Swapchain and its Images, CommandBuffer and its pool CommandPool, and any resource used by the CommandBuffer.
				while (swapchain->acquiredImagesAwaitingPresent()) {}
				swapchain = nullptr;

				status = STATUS::IRRECOVERABLE;
			}	

			inline bool recreateSwapchain(ISurface* surface, ILogicalDevice* device)
			{
				auto fail = [&]()->bool {becomeIrrecoverable(); return false;};
				using namespace video;

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

				const auto& swapchainParams = swapchain->getCreationParameters();
				const auto masterFormat = swapchainParams.surfaceFormat.format;
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
				assert(subpasses[1]==IGPURenderpass::SCreationParams::SubpassesEnd);
				subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};

				IGPURenderpass::SCreationParams params = {};
				params.colorAttachments = colorAttachments;
				params.subpasses = subpasses;
				// no subpass dependencies
				defaultRenderpass = device->createRenderpass(params);
				if (!defaultRenderpass)
					return fail();

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

				status = STATUS::USABLE;
				return true;
			}

			ISwapchain::SSharedCreationParams sharedParams;
		} m_swapchainResources = {};
};
// TODO: move comment
// Window/Surface got closed, but won't actually disappear UNTIL the swapchain gets dropped,
// which is outside of our control here as there is a nice chain of lifetimes of:
// `ExternalCmdBuf -via usage of-> Swapchain Image -memory provider-> Swapchain -created from-> Window/Surface`
// Only when the last user of the swapchain image drops it, will the window die.

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
				auto framebuffer = m_surface->getSwapchainResources().defaultFramebuffers[acquiredImageIx].get();

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

			return m_surface && m_surface->getSwapchainResources().status!=SimpleResizableSurface::ISwapchainResources::STATUS::IRRECOVERABLE;
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