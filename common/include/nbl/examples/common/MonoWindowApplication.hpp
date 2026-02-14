// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_MONO_WINDOW_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_MONO_WINDOW_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "nbl/examples/common/SimpleWindowedApplication.hpp"
#include "nbl/examples/common/CSwapchainFramebuffersAndDepth.hpp"
#include "nbl/examples/common/CEventCallback.hpp"

namespace nbl::examples
{
	
// Virtual Inheritance because apps might end up doing diamond inheritance
class MonoWindowApplication : public virtual SimpleWindowedApplication
{
		using base_t = SimpleWindowedApplication;

	public:
		// Maximum frames which can be simultaneously submitted, used to cycle through our per-frame resources like command buffers
		constexpr static inline uint8_t MaxFramesInFlight = 3;

		template<typename... Args>
		MonoWindowApplication(const hlsl::uint16_t2 _initialResolution, const asset::E_FORMAT _depthFormat, Args&&... args) :
			base_t(std::forward<Args>(args)...), m_initialResolution(_initialResolution), m_depthFormat(_depthFormat) {}

		//
		inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override final
		{
			if (!m_surface)
			{
				using namespace nbl::core;
				using namespace nbl::ui;
				using namespace nbl::video;
				{
					auto windowCallback = make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem),smart_refctd_ptr(m_logger));
					IWindow::SCreationParams params = {};
					params.callback = make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
					params.width = m_initialResolution[0];
					params.height = m_initialResolution[1];
					params.x = 32;
					params.y = 32;
					params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE | IWindow::ECF_CAN_MINIMIZE;
					params.windowCaption = "MonoWindowApplication";
					params.callback = windowCallback;
					const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
				}

				auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
				const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>::create(std::move(surface));
			}

			if (m_surface)
				return { {m_surface->getSurface()/*,EQF_NONE*/} };

			return {};
		}
		
		virtual inline bool onAppInitialized(core::smart_refctd_ptr<system::ISystem>&& system) override
		{
			using namespace nbl::core;
			using namespace nbl::video;
			// want to have a usable system and logger first
			if (!MonoSystemMonoLoggerApplication::onAppInitialized(std::move(system)))
				return false;

			m_inputSystem = make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));
			if (!base_t::onAppInitialized(std::move(system)))
				return false;
			
			ISwapchain::SCreationParams swapchainParams = { .surface = smart_refctd_ptr<ISurface>(m_surface->getSurface()) };
			if (!swapchainParams.deduceFormat(m_physicalDevice))
				return logFail("Could not choose a Surface Format for the Swapchain!");
			
			// TODO: option without depth
			auto scResources = std::make_unique<CSwapchainFramebuffersAndDepth>(m_device.get(),m_depthFormat,swapchainParams.surfaceFormat.format,getDefaultSubpassDependencies());
			auto* renderpass = scResources->getRenderpass();

			if (!renderpass)
				return logFail("Failed to create Renderpass!");

			auto gQueue = getGraphicsQueue();
			if (!m_surface || !m_surface->init(gQueue,std::move(scResources),swapchainParams.sharedParams))
				return logFail("Could not create Window & Surface or initialize the Surface!");
			
			m_winMgr->setWindowSize(m_window.get(),m_initialResolution[0],m_initialResolution[1]);
			m_surface->recreateSwapchain();

			return true;
		}

		// we do slight inversion of control here
		inline void workLoopBody() override final
		{
			using namespace nbl::core;
			using namespace nbl::video;
			// framesInFlight: ensuring safe execution of command buffers and acquires, `framesInFlight` only affect semaphore waits, don't use this to index your resources because it can change with swapchain recreation.
			const uint32_t framesInFlightCount = hlsl::min(MaxFramesInFlight,m_surface->getMaxAcquiresInFlight());
			// We block for semaphores for 2 reasons here:
				// A) Resource: Can't use resource like a command buffer BEFORE previous use is finished! [MaxFramesInFlight]
				// B) Acquire: Can't have more acquires in flight than a certain threshold returned by swapchain or your surface helper class. [MaxAcquiresInFlight]
			if (m_framesInFlight.size()>=framesInFlightCount)
			{
				const ISemaphore::SWaitInfo framesDone[] =
				{
					{
						.semaphore = m_framesInFlight.front().semaphore.get(),
						.value = m_framesInFlight.front().value
					}
				};
				if (m_device->blockForSemaphores(framesDone)!=ISemaphore::WAIT_RESULT::SUCCESS)
					return;
				m_framesInFlight.pop_front();
			}

			auto updatePresentationTimestamp = [&]()
			{
				m_currentImageAcquire = m_surface->acquireNextImage();

				// TODO: better frame pacing than this
				oracle.reportEndFrameRecord();
				const auto timestamp = oracle.getNextPresentationTimeStamp();
				oracle.reportBeginFrameRecord();

				return timestamp;
			};

			const auto nextPresentationTimestamp = updatePresentationTimestamp();

			if (!m_currentImageAcquire)
				return;

			const IQueue::SSubmitInfo::SSemaphoreInfo rendered[] = {renderFrame(nextPresentationTimestamp)};
			m_surface->present(m_currentImageAcquire.imageIndex,rendered);
			if (rendered->semaphore)
				m_framesInFlight.emplace_back(smart_refctd_ptr<ISemaphore>(rendered->semaphore),rendered->value);
		}

		//
		virtual inline bool keepRunning() override
		{
			if (m_surface->irrecoverable())
				return false;

			return true;
		}

		//
		virtual inline bool onAppTerminated()
		{
			m_inputSystem = nullptr;
			m_device->waitIdle();
			m_framesInFlight.clear();
			m_surface = nullptr;
			m_window = nullptr;
			return base_t::onAppTerminated();
		}

	protected:
		inline void onAppInitializedFinish()
		{
			m_winMgr->show(m_window.get());
			oracle.reportBeginFrameRecord();
		}
		inline const auto& getCurrentAcquire() const {return m_currentImageAcquire;}

		virtual const video::IGPURenderpass::SCreationParams::SSubpassDependency* getDefaultSubpassDependencies() const = 0;
		virtual video::IQueue::SSubmitInfo::SSemaphoreInfo renderFrame(const std::chrono::microseconds nextPresentationTimestamp) = 0;

		const hlsl::uint16_t2 m_initialResolution;
		const asset::E_FORMAT m_depthFormat;
		core::smart_refctd_ptr<InputSystem> m_inputSystem;
		core::smart_refctd_ptr<ui::IWindow> m_window;
		core::smart_refctd_ptr<video::CSimpleResizeSurface<CSwapchainFramebuffersAndDepth>> m_surface;

	private:
		struct SSubmittedFrame
		{
			core::smart_refctd_ptr<video::ISemaphore> semaphore;
			uint64_t value;
		};
		core::deque<SSubmittedFrame> m_framesInFlight;
		video::ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		video::CDumbPresentationOracle oracle;
};

}
#endif
