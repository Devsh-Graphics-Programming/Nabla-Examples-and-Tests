#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include "IProjection.hpp"
#include "CGeneralPurposeGimbal.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl 
{

class ICameraController : virtual public core::IReferenceCounted
{
public:
    enum ControllerType : uint8_t
    {
        Keyboard = 0,
        Mouse = 1,

        Count
    };

    struct CKeyInfo
    {
        union
        {
            ui::E_KEY_CODE keyboardCode;
            ui::E_MOUSE_CODE mouseCode;
        };

        CKeyInfo(ControllerType _type, ui::E_KEY_CODE _keyboardCode) : keyboardCode(_keyboardCode), type(_type) {}
        CKeyInfo(ControllerType _type, ui::E_MOUSE_CODE _mouseCode) : mouseCode(_mouseCode), type(_type) {}

        ControllerType type;
    };

    struct CHashInfo
    {
        CHashInfo() {}
        CHashInfo(CVirtualGimbalEvent::VirtualEventType _type) : type(_type) {}
        ~CHashInfo() = default;

        CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::VirtualEventType::None;
        bool active = false;
        float64_t dtAction = {};
    };

    using keys_to_virtual_events_t = std::unordered_map<CKeyInfo, CHashInfo>;

    ICameraController() {}

    // Binds key codes to virtual events, the mapKeys lambda will be executed with controller keys_to_virtual_events_t table 
    void updateKeysToEvent(const std::function<void(keys_to_virtual_events_t&)>& mapKeys)
    {
        mapKeys(m_keysToVirtualEvents);
    }

    inline const keys_to_virtual_events_t& getKeysToVirtualEvents() { return m_keysToVirtualEvents; }


    void beginInputProcessing(const std::chrono::microseconds _nextPresentationTimeStamp)
    {
        nextPresentationTimeStamp = _nextPresentationTimeStamp;
    }

    void endInputProcessing()
    {
        lastVirtualUpTimeStamp = nextPresentationTimeStamp;
    }

private:
    keys_to_virtual_events_t m_keysToVirtualEvents;
    std::chrono::microseconds nextPresentationTimeStamp = {}, lastVirtualUpTimeStamp = {};
};

#if 0 // TOOD: update
template<typename R>
concept GimbalRange = GeneralPurposeRange<R> && requires 
{
    requires ProjectionMatrix<typename std::ranges::range_value_t<R>::projection_t>;
    requires std::same_as<std::ranges::range_value_t<R>, typename ICameraController<typename std::ranges::range_value_t<R>::projection_t>::CGimbal>;
};

template<GimbalRange Range>
class IGimbalRange : public IRange<typename Range>
{
public:
    using base_t = IRange<typename Range>;
    using range_t = typename base_t::range_t;
    using gimbal_t = typename base_t::range_value_t;

    IGimbalRange(range_t&& gimbals) : base_t(std::move(gimbals)) {}
    inline const range_t& getGimbals() const { return base_t::m_range; }

protected:
    inline range_t& getGimbals() const { return base_t::m_range; }
};

// TODO NOTE: eg. "follow camera" should use GimbalRange<std::array<ICameraController<T>::CGimbal, 2u>>, 
// one per camera itself and one for target it follows
#endif

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_