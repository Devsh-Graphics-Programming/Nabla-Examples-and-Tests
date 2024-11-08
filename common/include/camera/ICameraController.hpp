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
        Keyboard,
        Mouse,

        Count
    };

    struct CKeyInfo
    {
        union
        {
            ui::E_KEY_CODE keyboardCode;
            ui::E_MOUSE_CODE mouseCode;
        };

        CKeyInfo(ui::E_KEY_CODE _keyboardCode) : keyboardCode(_keyboardCode), type(Keyboard) {}
        CKeyInfo(ui::E_MOUSE_CODE _mouseCode) : mouseCode(_mouseCode), type(Mouse) {}

        ControllerType type;
    };

    struct CHashInfo
    {
        CHashInfo() {}
        CHashInfo(CVirtualGimbalEvent::VirtualEventType _type) : event({ .type = _type }) {}
        ~CHashInfo() = default;

        CVirtualGimbalEvent event = {};
        bool active = false;
    };

    void beginInputProcessing(const std::chrono::microseconds _nextPresentationTimeStamp)
    {
        nextPresentationTimeStamp = _nextPresentationTimeStamp;
    }

    void endInputProcessing()
    {
        lastVirtualUpTimeStamp = nextPresentationTimeStamp;
    }

protected:
    std::chrono::microseconds nextPresentationTimeStamp = {}, lastVirtualUpTimeStamp = {};
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_