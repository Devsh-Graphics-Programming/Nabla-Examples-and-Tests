#ifndef __NBL_COMMON_API_H_INCLUDED__
#define __NBL_COMMON_API_H_INCLUDED__

#include <nabla.h>

#include "MonoSystemMonoLoggerApplication.hpp"

#include "nbl/ui/CGraphicalApplicationAndroid.h"
#include "nbl/ui/CWindowManagerAndroid.h"

// TODO: see TODO below
// TODO: make these include themselves via `nabla.h`

#include "nbl/video/utilities/SPhysicalDeviceFilter.h"

#if 0
class CommonAPI
{
	CommonAPI() = delete;
public:		
	class CommonAPIEventCallback : public nbl::ui::IWindow::IEventCallback
	{
	public:
		CommonAPIEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(inputSystem)), m_logger(std::move(logger)), m_gotWindowClosedMsg(false){}
		CommonAPIEventCallback() {}
		bool isWindowOpen() const {return !m_gotWindowClosedMsg;}
		void setLogger(nbl::system::logger_opt_smart_ptr& logger)
		{
			m_logger = logger;
		}
		void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& inputSystem)
		{
			m_inputSystem = std::move(inputSystem);
		}
	private:
		
		bool onWindowClosed_impl() override
		{
			m_logger.log("Window closed");
			m_gotWindowClosedMsg = true;
			return true;
		}

		void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
		{
			m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_mouse,std::move(mch));
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse,mch);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard,std::move(kbch));
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard,kbch);
		}

	private:
		nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
		bool m_gotWindowClosedMsg;
	};

	// old code from init
	{
		// ... 

		result.inputSystem = nbl::core::make_smart_refctd_ptr<InputSystem>(system::logger_opt_smart_ptr(nbl::core::smart_refctd_ptr(result.logger)));
		result.assetManager = nbl::core::make_smart_refctd_ptr<nbl::asset::IAssetManager>(nbl::core::smart_refctd_ptr(result.system), nbl::core::smart_refctd_ptr(result.compilerSet)); // we should let user choose it?

		if (!headlessCompute)
		{
			params.windowCb->setInputSystem(nbl::core::smart_refctd_ptr(result.inputSystem));
			if (!params.window)
			{
				#ifdef _NBL_PLATFORM_WINDOWS_
					result.windowManager = ui::IWindowManagerWin32::create(); // on the Windows path
				#elif defined(_NBL_PLATFORM_LINUX_)
					result.windowManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerX11>(); // on the Android path
				#else
					#error "Unsupported platform"
				#endif
				
				nbl::ui::IWindow::SCreationParams windowsCreationParams;
				windowsCreationParams.width = params.windowWidth;
				windowsCreationParams.height = params.windowHeight;
				windowsCreationParams.x = 64u;
				windowsCreationParams.y = 64u;
				windowsCreationParams.flags = nbl::ui::IWindow::ECF_RESIZABLE;
				windowsCreationParams.windowCaption = params.appName.data();
				windowsCreationParams.callback = params.windowCb;

				params.window = result.windowManager->createWindow(std::move(windowsCreationParams));
			}
			params.windowCb = nbl::core::smart_refctd_ptr<CommonAPIEventCallback>((CommonAPIEventCallback*) params.window->getEventCallback());
		}

		// ...
	}
};

#endif

#endif
