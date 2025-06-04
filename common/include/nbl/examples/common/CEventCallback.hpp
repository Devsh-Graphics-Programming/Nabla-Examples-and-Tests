#ifndef _NBL_EXAMPLES_COMMON_C_EVENT_CALLBACK_HPP_INCLUDED_
#define _NBL_EXAMPLES_COMMON_C_EVENT_CALLBACK_HPP_INCLUDED_


#include "nbl/video/utilities/CSimpleResizeSurface.h"

#include "nbl/examples/common/InputSystem.hpp"


namespace nbl::examples
{
class CEventCallback : public nbl::video::ISimpleManagedSurface::ICallback
{
	public:
		CEventCallback(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem, nbl::system::logger_opt_smart_ptr&& logger) : m_inputSystem(std::move(m_inputSystem)), m_logger(std::move(logger)) {}
		CEventCallback() {}

		void setLogger(nbl::system::logger_opt_smart_ptr& logger)
		{
			m_logger = logger;
		}
		void setInputSystem(nbl::core::smart_refctd_ptr<InputSystem>&& m_inputSystem)
		{
			m_inputSystem = std::move(m_inputSystem);
		}

	private:
		void onMouseConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IMouseEventChannel>&& mch) override
		{
			m_logger.log("A mouse %p has been connected", nbl::system::ILogger::ELL_INFO, mch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_mouse, std::move(mch));
		}
		void onMouseDisconnected_impl(nbl::ui::IMouseEventChannel* mch) override
		{
			m_logger.log("A mouse %p has been disconnected", nbl::system::ILogger::ELL_INFO, mch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_mouse, mch);
		}
		void onKeyboardConnected_impl(nbl::core::smart_refctd_ptr<nbl::ui::IKeyboardEventChannel>&& kbch) override
		{
			m_logger.log("A keyboard %p has been connected", nbl::system::ILogger::ELL_INFO, kbch.get());
			m_inputSystem.get()->add(m_inputSystem.get()->m_keyboard, std::move(kbch));
		}
		void onKeyboardDisconnected_impl(nbl::ui::IKeyboardEventChannel* kbch) override
		{
			m_logger.log("A keyboard %p has been disconnected", nbl::system::ILogger::ELL_INFO, kbch);
			m_inputSystem.get()->remove(m_inputSystem.get()->m_keyboard, kbch);
		}

	private:
		nbl::core::smart_refctd_ptr<InputSystem> m_inputSystem = nullptr;
		nbl::system::logger_opt_smart_ptr m_logger = nullptr;
};
}
#endif