// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//
#include "nbl/video/surface/CSurfaceVulkan.h"

#include "../common/BasicMultiQueueApplication.hpp"

namespace nbl::examples
{
// Virtual Inheritance because apps might end up doing diamond inheritance
class SimpleWindowedApplication : public virtual BasicMultiQueueApplication
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

		// virtual function so you can override as needed for some example father down the line
		virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
		{
			auto retval = base_t::getRequiredDeviceFeatures();
			retval.swapchainMode = video::E_SWAPCHAIN_MODE::ESM_SURFACE;
			return retval;
		}
		
		virtual bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			// want to have a usable system and logger first
			if (!MonoSystemMonoLoggerApplication::onAppInitialized(std::move(system)))
				return false;
			
		#ifdef _NBL_PLATFORM_WINDOWS_
			m_winMgr = nbl::ui::IWindowManagerWin32::create();
		#else
			#error "Unimplemented!"
		#endif
			if (!m_winMgr)
				return false;

			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(core::smart_refctd_ptr(m_system)))
				return false;

			return true;
		}

	protected:
		core::smart_refctd_ptr<ui::IWindowManager> m_winMgr;
};

}


#include "nbl/video/CVulkanSwapchain.h"
namespace nbl::examples
{

// Simple wrapper class intended for Single Threaded usage
class SimpleNonResizableSurface : public core::IReferenceCounted
{
	public:
		// Simple callback to facilitate detection of window being closed
		class IClosedCallback : public nbl::ui::IWindow::IEventCallback
		{
			public:
				inline IClosedCallback() : m_gotWindowClosedMsg(false) {}

				// unless you create a separate callback per window, both will "trip" this condition
				inline bool windowGotClosed() const {return m_gotWindowClosedMsg;}

			private:
				inline bool onWindowClosed_impl() override
				{
					m_gotWindowClosedMsg = true;
					return true;
				}

				bool m_gotWindowClosedMsg;
		};

		// Factory method so we can fail, requires a `_surface` created from a window and with a callback that inherits from `IClosedCallback` declared just above
		static inline core::smart_refctd_ptr<SimpleNonResizableSurface> create(core::smart_refctd_ptr<video::ISurface>&& _surface)
		{
			if (!_surface)
				return nullptr;

			auto _window = _surface->getWindow();
			if (!_window)
				return nullptr;

			auto cb = dynamic_cast<IClosedCallback*>(_window->getEventCallback());
			if (!cb)
				return nullptr;

			return core::smart_refctd_ptr<SimpleNonResizableSurface>(new SimpleNonResizableSurface(std::move(_surface),cb),core::dont_grab);
		}

		//
		inline const video::ISurface* getSurface() const {return m_surface.get();}

		// A small utility for the boilerplate
		inline uint8_t pickQueueFamily(video::ILogicalDevice* device) const
		{
			uint8_t qFam = 0u;
			for (; qFam<video::ILogicalDevice::MaxQueueFamilies; qFam++)
			if (device->getQueueCount(qFam) && m_surface->isSupportedForPhysicalDevice(device->getPhysicalDevice(),qFam))
				break;
			return qFam;
		}

		// Just pick the first queue within the first compatible family
		inline video::IQueue* pickQueue(video::ILogicalDevice* device) const
		{
			return device->getThreadSafeQueue(pickQueueFamily(device),0);
		}

		// We need to defer the swapchain creation till the Physical Device is chosen and Queues are created together with the Logical Device
		inline bool createSwapchain(video::IQueue* queue, const video::ISwapchain::SSharedCreationParams& sharedParams={}, const video::ISurface::SFormat& surfaceFormat={})
		{
			if (!queue)
				return false;

			using namespace nbl::video;
			ISwapchain::SCreationParams params = {
				.surface = core::smart_refctd_ptr(m_surface),
				.surfaceFormat = surfaceFormat,
				.sharedParams = sharedParams
				// we're not going to support concurrent sharing in this simple class
			};
			
			auto device = const_cast<ILogicalDevice*>(queue->getOriginDevice());
			const auto physDev = device->getPhysicalDevice();
			if (!params.sharedParams.deduce(physDev,m_surface.get()))
				return false;
			const auto swapchainSharedParams = params.sharedParams;

			if (!params.deduceFormat(physDev))
				return false;
			const auto masterFormat = params.surfaceFormat.format;
			
			if (m_swapchainResources.acquireSemaphore=device->createSemaphore(0u))
			switch (m_surface->getAPIType())
			{
				case EAT_VULKAN:
					m_swapchainResources.swapchain = CVulkanSwapchain::create(core::smart_refctd_ptr<const ILogicalDevice>(device),std::move(params));
					if (m_swapchainResources.swapchain)
						break;
					[[fallthrough]];
				default:
					m_swapchainResources = {};
					return false;
			}
			
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
				assert(subpasses[1]==IGPURenderpass::SCreationParams::SubpassesEnd);
				subpasses[0].colorAttachments[0] = {.render={.attachmentIndex=0,.layout=IGPUImage::LAYOUT::ATTACHMENT_OPTIMAL}};

				IGPURenderpass::SCreationParams params = {};
				params.colorAttachments = colorAttachments;
				params.subpasses = subpasses;
				// no subpass dependencies
				m_swapchainResources.defaultRenderpass = device->createRenderpass(params);
				if (!m_swapchainResources.defaultRenderpass)
				{
					m_swapchainResources = {};
					return false;
				}
			}
			for (auto i=0u; i<m_swapchainResources.swapchain->getImageCount(); i++)
			{
				auto imageView = device->createImageView({
					.flags = IGPUImageView::ECF_NONE,
					.subUsages = IGPUImage::EUF_RENDER_ATTACHMENT_BIT,
					.image = m_swapchainResources.swapchain->createImage(i),
					.viewType = IGPUImageView::ET_2D,
					.format = masterFormat
				});
				IGPUFramebuffer::SCreationParams params = {{
					.renderpass = core::smart_refctd_ptr(m_swapchainResources.defaultRenderpass),
					.depthStencilAttachments = nullptr,
					.colorAttachments = &imageView.get(),
					.width = swapchainSharedParams.width,
					.height = swapchainSharedParams.height,
					.layers = swapchainSharedParams.arrayLayers
				}};
				m_swapchainResources.defaultFramebuffers[i] = device->createFramebuffer(std::move(params));
				if (!m_swapchainResources.defaultFramebuffers[i])
				{
					m_swapchainResources = {};
					return false;
				}
			}
			m_queue = queue;
			return true;
		}

		//
		inline video::IQueue* getAssignedQueue() {return m_queue;}

		//
		inline video::ISwapchain* getSwapchain() {return m_swapchainResources.swapchain.get();}
		inline const video::ISwapchain* getSwapchain() const {return m_swapchainResources.swapchain.get();}

		// An interesting solution to the "Frames In Flight", our tiny wrapper class will have its own Timeline Semaphore incrementing with each acquire, and thats it.
		inline video::ISemaphore* getAcquireSemaphore() {return m_swapchainResources.acquireSemaphore.get();}

		//
		inline video::IGPURenderpass* getDefaultRenderpass() {return m_swapchainResources.defaultRenderpass.get();}
		//
		inline video::IGPUFramebuffer* getDefaultFramebuffer(const uint8_t imageIndex) {return m_swapchainResources.defaultFramebuffers[imageIndex].get();}

		// RETURNS: Negative on failure, otherwise its the acquired image's index.
		inline int8_t acquireNextImage()
		{
			if (!m_swapchainResources || m_cb->windowGotClosed())
			{
				m_swapchainResources = {};
				return -1;
			}
			const auto frameIx = m_swapchainResources.swapchain->getAcquireCount();

			using namespace nbl::video;
			const IQueue::SSubmitInfo::SSemaphoreInfo signalInfos[1] = {
				{.semaphore=m_swapchainResources.acquireSemaphore.get(),.value=frameIx+1}
			};

			uint32_t imageIndex;
			// We don't support resizing (swapchain recreation) in this example, so a failure to acquire is a failure to keep running
			switch (m_swapchainResources.swapchain->acquireNextImage({.queue=m_queue,.signalSemaphores=signalInfos},&imageIndex))
			{
				case ISwapchain::ACQUIRE_IMAGE_RESULT::SUBOPTIMAL: [[fallthrough]];
				case ISwapchain::ACQUIRE_IMAGE_RESULT::SUCCESS:
					return imageIndex;
				default:
					break;
			}
			m_swapchainResources = {};
			return -1;
		}

		// Frame Resources are not optional!
		inline bool present(const uint8_t imageIndex, const std::span<const video::IQueue::SSubmitInfo::SSemaphoreInfo> waitSemaphores, core::smart_refctd_ptr<core::IReferenceCounted>&& frameResources)
		{
			if (!m_swapchainResources || m_cb->windowGotClosed())
			{
				m_swapchainResources = {};
				return false;
			}

			using namespace nbl::video;
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
				default:
					break;
			}
			m_swapchainResources = {};
			return false;
		}

	protected:
		inline SimpleNonResizableSurface(core::smart_refctd_ptr<video::ISurface>&& _surface, IClosedCallback* _cb)
			: m_surface(std::move(_surface)), m_cb(_cb) {}

		core::smart_refctd_ptr<video::ISurface> m_surface;
		IClosedCallback* m_cb;
		video::IQueue* m_queue;
		// these can die spontaneously
		struct SwapchainResources
		{
			inline operator bool() const {return bool(swapchain);}

			core::smart_refctd_ptr<video::ISwapchain> swapchain = {};
			core::smart_refctd_ptr<video::ISemaphore> acquireSemaphore = {};
			// If these get used, they should indirectly find their way into the `frameResources` argument of the `present` method.
			core::smart_refctd_ptr<video::IGPURenderpass> defaultRenderpass = {};
			std::array<core::smart_refctd_ptr<video::IGPUFramebuffer>,video::ISwapchain::MaxImages> defaultFramebuffers = {};
		} m_swapchainResources = {};
};

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
				params.callback = core::make_smart_refctd_ptr<examples::SimpleNonResizableSurface::IClosedCallback>();
				params.width = 640;
				params.height = 480;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_NONE;
				params.windowCaption = "HelloSwapchainApp";
				auto window = m_winMgr->createWindow(std::move(params));
				const_cast<core::smart_refctd_ptr<examples::SimpleNonResizableSurface>&>(m_surface) = examples::SimpleNonResizableSurface::create(
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

			//
			m_semaphore = m_device->createSemaphore(0);
			if (!m_semaphore)
				return logFail("Failed to Create a Semaphore!");

			// We just live life in easy mode and have the Swapchain Creation Parameters get deduced from the surface.
			// We don't need any control over the format of the swapchain because we'll be using Renderpasses this time!
 			if (!m_surface || !m_surface->createSwapchain(m_surface->pickQueue(m_device.get())))
				return logFail("Failed to Create a Swapchain!");

			//
			const auto* swapchain = m_surface->getSwapchain();
			const auto imageCount = swapchain->getImageCount();
			const auto& swapchainParams = swapchain->getCreationParameters();
			const auto& swapchainSharedParams = swapchainParams.sharedParams;

			// This time we'll experiment with pre-recorded and reusable CommandBuffers, with one for each swapchain image.
			auto pool = m_device->createCommandPool(getGraphicsQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE);
			if (!pool || !pool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{m_cmdBufs.data(),imageCount},core::smart_refctd_ptr(m_logger)))
				return logFail("Failed to Create CommandBuffers!");

			// We'll just clear the whole window to red in this example
			IGPUCommandBuffer::SClearColorValue clearValue = {.float32={1.f,0.f,0.f,1.f}};
			for (auto i=0u; i<imageCount; i++)
			{
				auto cmdBuf = m_cmdBufs[i].get();
				cmdBuf->begin(IGPUCommandBuffer::USAGE::NONE);
				const IGPUCommandBuffer::SRenderpassBeginInfo info = {
					.framebuffer = m_surface->getDefaultFramebuffer(i),
					.colorClearValues = &clearValue,
					.depthStencilClearValues = nullptr,
					.renderArea = {.offset={0,0},.extent={swapchainSharedParams.width,swapchainSharedParams.height}}
				};
				cmdBuf->beginRenderPass(info,IGPUCommandBuffer::SUBPASS_CONTENTS::INLINE);
				cmdBuf->endRenderPass();
				cmdBuf->end();
			}

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
			const IQueue::SSubmitInfo::SSemaphoreInfo renderingDone[1] = {{
				.semaphore = m_semaphore.get(),
				.value = ++m_frameIx,
				.stageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT
			}};
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{
				.cmdbuf = m_cmdBufs[m_currentAcquire].get()
			}};
			const uint64_t framesInFlight = m_surface->getSwapchain()->getImageCount();
			const IQueue::SSubmitInfo::SSemaphoreInfo waitSemaphores[] = {
				{ // acquired swapchain image so can start rendering into it
					.semaphore = m_surface->getAcquireSemaphore(),
					.value = m_surface->getSwapchain()->getAcquireCount(),
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COLOR_ATTACHMENT_OUTPUT_BIT // wait till we output
				},
				{ // previous use of commandbuffer is finished (we created cmbufs without the simultaneous use flag)
					.semaphore = m_semaphore.get(),
					.value = m_frameIx-framesInFlight,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS // wait till we start processing
				}
			};

			const IQueue::SSubmitInfo submitInfos[1] = {
				{
					.waitSemaphores = {waitSemaphores,m_frameIx>framesInFlight ? 2ull:1ull},
					.commandBuffers = cmdbufs,
					.signalSemaphores = renderingDone
				}
			};
			getGraphicsQueue()->submit(submitInfos);

			m_surface->present(static_cast<uint8_t>(m_currentAcquire),renderingDone,smart_refctd_ptr<IGPUCommandBuffer>(cmdbufs->cmdbuf));
		}

		//
		inline bool keepRunning() override
		{
			if (duration_cast<decltype(timeout)>(clock_t::now()-start)>timeout)
				return false;

			return (m_currentAcquire=m_surface->acquireNextImage())>=0;
		}

		//
		virtual bool onAppTerminated() override
		{
			ISemaphore::SWaitInfo infos[1] = {
				{.semaphore=m_semaphore.get(),.value=m_frameIx}
			};
			m_device->blockForSemaphores(infos);
			m_surface = nullptr;

			return base_t::onAppTerminated();
		}

	protected:
		// Just like in the HelloUI app we add a timeout
		std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point start;
		// In this example we have just one Window & Swapchain
		smart_refctd_ptr<examples::SimpleNonResizableSurface> m_surface;
		// and pre-recorded re-usable Command Buffers
		std::array<smart_refctd_ptr<IGPUCommandBuffer>,ISwapchain::MaxImages> m_cmdBufs;
		// We can't use the same semaphore for acquire and present, because that would disable "Frames in Flight" by syncing previous present against next acquire.
		// At least two timelines must be used.
		smart_refctd_ptr<ISemaphore> m_semaphore;
		// Current acquired Image Index
		int8_t m_currentAcquire;
		// Technically the same as `m_surface->getSwapchain()->getAcquireCount()` but using a seaparate variable for clarity
		uint64_t m_frameIx = 0;
};

// define an entry point as always!
NBL_MAIN_FUNC(HelloSwapchainApp)