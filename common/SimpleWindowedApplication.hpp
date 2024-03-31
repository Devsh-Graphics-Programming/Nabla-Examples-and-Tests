// Copyright (C) 2023-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_COMMON_SIMPLE_WINDOWED_APPLICATION_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_SIMPLE_WINDOWED_APPLICATION_HPP_INCLUDED_

// Build on top of the previous one
#include "nbl/application_templates/BasicMultiQueueApplication.hpp"

namespace nbl::examples
{
	
// Virtual Inheritance because apps might end up doing diamond inheritance
class SimpleWindowedApplication : public virtual application_templates::BasicMultiQueueApplication
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

		// Just to run destructors in a nice order
		virtual bool onAppTerminated() override
		{
			m_winMgr = nullptr;
			return base_t::onAppTerminated();
		}

	protected:
		core::smart_refctd_ptr<ui::IWindowManager> m_winMgr;
};

}

#endif // _CAMERA_IMPL_