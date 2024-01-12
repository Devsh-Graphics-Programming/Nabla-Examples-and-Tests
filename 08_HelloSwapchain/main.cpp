// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

//
#include "nbl/video/surface/CSurfaceVulkan.h"

#include "../common/BasicMultiQueueApplication.hpp"

namespace nbl::examples
{
// Virtual Inheritance because apps might end up doing diamond inheritance
class WindowedApplication : public virtual BasicMultiQueueApplication
{
		using base_t = BasicMultiQueueApplication;

	public:
		using base_t::base_t;

		virtual video::IAPIConnection::SFeatures getAPIFeaturesToEnable() override
		{
			auto retval = base_t::getAPIFeaturesToEnable();
			// We only support one swapchain mode, surface, the other one is Display which we have not implemented yet.
			retval.swapchainMode = video::E_SWAPCHAIN_MODE::ESM_SURFACE;
			return retval;
		}

		// New function, we neeed to know about surfaces to create ahead of time
		virtual core::vector<const video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const = 0;

		virtual core::set<video::IPhysicalDevice*> filterDevices(const core::SRange<video::IPhysicalDevice* const>& physicalDevices) const
		{
			const auto firstFilter = base_t::filterDevices(physicalDevices);

			video::SPhysicalDeviceFilter deviceFilter = {};
			
			const auto surfaces = getSurfaces();
			deviceFilter.requiredSurfaceCompatibilities = surfaces.data();
			deviceFilter.requiredSurfaceCompatibilitiesCount = surfaces.size();

			return deviceFilter(physicalDevices);
		}
		
		virtual bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

		#ifdef _NBL_PLATFORM_WINDOWS_
			m_winMgr = nbl::ui::IWindowManagerWin32::create();
		#else
			#error "Unimplemented!"
		#endif
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

		virtual bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;

			m_window = m_winMgr->createWindow(getWindowCreationParams());
			m_surface = video::CSurfaceVulkanWin32::create(core::smart_refctd_ptr(m_api),core::smart_refctd_ptr_static_cast<ui::IWindowWin32>(m_window));
			return true;
		}

		virtual core::vector<const video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const
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
		virtual IWindow::SCreationParams getWindowCreationParams() const
		{
			IWindow::SCreationParams params = {};
			params.callback = make_smart_refctd_ptr<IWindowClosedCallback>();
			params.width = 640;
			params.height = 480;
			params.x = 32;
			params.y = 32;
			params.flags = IWindow::ECF_NONE;
			params.windowCaption = "SingleNonResizableWindowApplication";
			return params;
		}

		core::smart_refctd_ptr<ui::IWindow> m_window;
		core::smart_refctd_ptr<video::ISurfaceVulkan> m_surface;
};
}


using namespace nbl;
using namespace core;
using namespace system;
using namespace ui;

class HelloSwapchainApp final : public examples::SingleNonResizableWindowApplication
{
		using base_t = examples::SingleNonResizableWindowApplication;
		using clock_t = std::chrono::steady_clock;

	public:
		using base_t::base_t;

		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!base_t::onAppInitialized(std::move(system)))
				return false;
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
		}

		//
		bool keepRunning() override
		{
			if (duration_cast<decltype(timeout)>(clock_t::now()-start)>timeout)
				return false;

			return base_t::keepRunning();
		}

	protected:
		virtual IWindow::SCreationParams getWindowCreationParams() const
		{
			auto retval = base_t::getWindowCreationParams();
			retval.windowCaption = "HelloSwapchainApp";
			return retval;
		}

		// Just like in the HelloUI app we add a timeout
		std::chrono::seconds timeout = std::chrono::seconds(0x7fffFFFFu);
		clock_t::time_point start;
};

// define an entry point as always!
NBL_MAIN_FUNC(HelloSwapchainApp)