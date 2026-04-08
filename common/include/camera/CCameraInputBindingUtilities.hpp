#ifndef _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
#define _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_

#include "ICamera.hpp"
#include "IGimbalBindingLayout.hpp"

namespace nbl::ui
{

//! Reusable keyboard, mouse, and ImGuizmo binding preset grouped for one camera kind.
struct SCameraInputBindingPreset
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t keyboard;
    IGimbalBindingLayout::mouse_to_virtual_events_t mouse;
    IGimbalBindingLayout::imguizmo_to_virtual_events_t imguizmo;
};

namespace impl
{

inline IGimbalBindingLayout::keyboard_to_virtual_events_t makeKeyboardPreset(
    std::initializer_list<std::pair<IGimbalBindingLayout::encode_keyboard_code_t, core::CVirtualGimbalEvent::VirtualEventType>> bindings)
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t preset;
    for (const auto& [code, event] : bindings)
        preset.emplace(code, IGimbalBindingLayout::CHashInfo(event));
    return preset;
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t makeMousePreset(
    std::initializer_list<std::pair<IGimbalBindingLayout::encode_mouse_code_t, core::CVirtualGimbalEvent::VirtualEventType>> bindings)
{
    IGimbalBindingLayout::mouse_to_virtual_events_t preset;
    for (const auto& [code, event] : bindings)
        preset.emplace(code, IGimbalBindingLayout::CHashInfo(event));
    return preset;
}

inline IGimbalBindingLayout::keyboard_to_virtual_events_t extendKeyboardPreset(
    IGimbalBindingLayout::keyboard_to_virtual_events_t preset,
    std::initializer_list<std::pair<IGimbalBindingLayout::encode_keyboard_code_t, core::CVirtualGimbalEvent::VirtualEventType>> bindings)
{
    for (const auto& [code, event] : bindings)
        preset.emplace(code, IGimbalBindingLayout::CHashInfo(event));
    return preset;
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t extendMousePreset(
    IGimbalBindingLayout::mouse_to_virtual_events_t preset,
    std::initializer_list<std::pair<IGimbalBindingLayout::encode_mouse_code_t, core::CVirtualGimbalEvent::VirtualEventType>> bindings)
{
    for (const auto& [code, event] : bindings)
        preset.emplace(code, IGimbalBindingLayout::CHashInfo(event));
    return preset;
}

inline IGimbalBindingLayout::keyboard_to_virtual_events_t makeWasdKeyboardPreset(
    const core::CVirtualGimbalEvent::VirtualEventType w,
    const core::CVirtualGimbalEvent::VirtualEventType s,
    const core::CVirtualGimbalEvent::VirtualEventType a,
    const core::CVirtualGimbalEvent::VirtualEventType d)
{
    return makeKeyboardPreset({
        { ui::E_KEY_CODE::EKC_W, w },
        { ui::E_KEY_CODE::EKC_S, s },
        { ui::E_KEY_CODE::EKC_A, a },
        { ui::E_KEY_CODE::EKC_D, d }
    });
}

inline IGimbalBindingLayout::keyboard_to_virtual_events_t appendIjklLookKeyboardPreset(
    IGimbalBindingLayout::keyboard_to_virtual_events_t preset,
    const core::CVirtualGimbalEvent::VirtualEventType i,
    const core::CVirtualGimbalEvent::VirtualEventType k,
    const core::CVirtualGimbalEvent::VirtualEventType j,
    const core::CVirtualGimbalEvent::VirtualEventType l)
{
    return extendKeyboardPreset(std::move(preset), {
        { ui::E_KEY_CODE::EKC_I, i },
        { ui::E_KEY_CODE::EKC_K, k },
        { ui::E_KEY_CODE::EKC_J, j },
        { ui::E_KEY_CODE::EKC_L, l }
    });
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t makeRelativeMousePreset(
    const core::CVirtualGimbalEvent::VirtualEventType positiveX,
    const core::CVirtualGimbalEvent::VirtualEventType negativeX,
    const core::CVirtualGimbalEvent::VirtualEventType positiveY,
    const core::CVirtualGimbalEvent::VirtualEventType negativeY)
{
    return makeMousePreset({
        { ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X, positiveX },
        { ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X, negativeX },
        { ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y, positiveY },
        { ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y, negativeY }
    });
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t appendSymmetricScrollPreset(
    IGimbalBindingLayout::mouse_to_virtual_events_t preset,
    const core::CVirtualGimbalEvent::VirtualEventType positive,
    const core::CVirtualGimbalEvent::VirtualEventType negative)
{
    return extendMousePreset(std::move(preset), {
        { ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL, positive },
        { ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL, positive },
        { ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL, negative },
        { ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL, negative }
    });
}

inline IGimbalBindingLayout::imguizmo_to_virtual_events_t makeImguizmoPreset(const uint32_t allowedVirtualEvents)
{
    IGimbalBindingLayout::imguizmo_to_virtual_events_t preset;
    for (const auto event : core::CVirtualGimbalEvent::VirtualEventsTypeTable)
    {
        if (event == core::CVirtualGimbalEvent::None)
            continue;
        if ((allowedVirtualEvents & event) != event)
            continue;
        preset.emplace(event, IGimbalBindingLayout::CHashInfo(event));
    }
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& emptyKeyboardPreset()
{
    static const auto preset = IGimbalBindingLayout::keyboard_to_virtual_events_t{};
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& emptyMousePreset()
{
    static const auto preset = IGimbalBindingLayout::mouse_to_virtual_events_t{};
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& fpsKeyboardPreset()
{
    static const auto preset = appendIjklLookKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        core::CVirtualGimbalEvent::TiltDown,
        core::CVirtualGimbalEvent::TiltUp,
        core::CVirtualGimbalEvent::PanLeft,
        core::CVirtualGimbalEvent::PanRight);
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& freeKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(fpsKeyboardPreset(), {
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::RollLeft },
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::RollRight }
    });
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& orbitKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        {
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveForward },
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveBackward }
        });
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& arcballKeyboardPreset()
{
    static const auto preset = appendIjklLookKeyboardPreset(
        extendKeyboardPreset(
            makeWasdKeyboardPreset(
                core::CVirtualGimbalEvent::MoveForward,
                core::CVirtualGimbalEvent::MoveBackward,
                core::CVirtualGimbalEvent::MoveLeft,
                core::CVirtualGimbalEvent::MoveRight),
            {
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveDown },
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveUp },
            }),
        core::CVirtualGimbalEvent::TiltDown,
        core::CVirtualGimbalEvent::TiltUp,
        core::CVirtualGimbalEvent::PanLeft,
        core::CVirtualGimbalEvent::PanRight);
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& turntableKeyboardPreset()
{
    static const auto preset = appendIjklLookKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::PanRight),
        core::CVirtualGimbalEvent::TiltDown,
        core::CVirtualGimbalEvent::TiltUp,
        core::CVirtualGimbalEvent::PanLeft,
        core::CVirtualGimbalEvent::PanRight);
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& topDownKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        {
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveBackward },
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveForward },
        { ui::E_KEY_CODE::EKC_J, core::CVirtualGimbalEvent::PanLeft },
        { ui::E_KEY_CODE::EKC_L, core::CVirtualGimbalEvent::PanRight }
        });
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& isometricKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        {
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveBackward },
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveForward }
        });
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& chaseKeyboardPreset()
{
    return arcballKeyboardPreset();
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& dollyKeyboardPreset()
{
    return arcballKeyboardPreset();
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& dollyZoomKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        {
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveForward },
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveBackward }
        });
    return preset;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& pathKeyboardPreset()
{
    static const auto preset = extendKeyboardPreset(
        makeWasdKeyboardPreset(
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight),
        {
        { ui::E_KEY_CODE::EKC_Q, core::CVirtualGimbalEvent::MoveDown },
        { ui::E_KEY_CODE::EKC_E, core::CVirtualGimbalEvent::MoveUp }
        });
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& fpsMousePreset()
{
    static const auto preset = makeRelativeMousePreset(
        core::CVirtualGimbalEvent::PanRight,
        core::CVirtualGimbalEvent::PanLeft,
        core::CVirtualGimbalEvent::TiltUp,
        core::CVirtualGimbalEvent::TiltDown);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& freeMousePreset()
{
    return fpsMousePreset();
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& orbitMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::MoveRight,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown),
        core::CVirtualGimbalEvent::MoveForward,
        core::CVirtualGimbalEvent::MoveBackward);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& arcballMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::TiltUp,
            core::CVirtualGimbalEvent::TiltDown),
        core::CVirtualGimbalEvent::MoveForward,
        core::CVirtualGimbalEvent::MoveBackward);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& turntableMousePreset()
{
    return arcballMousePreset();
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& topDownMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown),
        core::CVirtualGimbalEvent::MoveForward,
        core::CVirtualGimbalEvent::MoveBackward);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& isometricMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::MoveRight,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown),
        core::CVirtualGimbalEvent::MoveForward,
        core::CVirtualGimbalEvent::MoveBackward);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& chaseMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::TiltUp,
            core::CVirtualGimbalEvent::TiltDown),
        core::CVirtualGimbalEvent::MoveUp,
        core::CVirtualGimbalEvent::MoveDown);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& dollyMousePreset()
{
    static const auto preset = appendSymmetricScrollPreset(
        makeRelativeMousePreset(
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::TiltUp,
            core::CVirtualGimbalEvent::TiltDown),
        core::CVirtualGimbalEvent::MoveForward,
        core::CVirtualGimbalEvent::MoveBackward);
    return preset;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& dollyZoomMousePreset()
{
    return isometricMousePreset();
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& pathMousePreset()
{
    return isometricMousePreset();
}

} // namespace impl

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera& camera)
{
    switch (camera.getKind())
    {
        case core::ICamera::CameraKind::FPS: return impl::fpsKeyboardPreset();
        case core::ICamera::CameraKind::Free: return impl::freeKeyboardPreset();
        case core::ICamera::CameraKind::Orbit: return impl::orbitKeyboardPreset();
        case core::ICamera::CameraKind::Arcball: return impl::arcballKeyboardPreset();
        case core::ICamera::CameraKind::Turntable: return impl::turntableKeyboardPreset();
        case core::ICamera::CameraKind::TopDown: return impl::topDownKeyboardPreset();
        case core::ICamera::CameraKind::Isometric: return impl::isometricKeyboardPreset();
        case core::ICamera::CameraKind::Chase: return impl::chaseKeyboardPreset();
        case core::ICamera::CameraKind::Dolly: return impl::dollyKeyboardPreset();
        case core::ICamera::CameraKind::DollyZoom: return impl::dollyZoomKeyboardPreset();
        case core::ICamera::CameraKind::Path: return impl::pathKeyboardPreset();
        default: return impl::emptyKeyboardPreset();
    }
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera& camera)
{
    switch (camera.getKind())
    {
        case core::ICamera::CameraKind::FPS: return impl::fpsMousePreset();
        case core::ICamera::CameraKind::Free: return impl::freeMousePreset();
        case core::ICamera::CameraKind::Orbit: return impl::orbitMousePreset();
        case core::ICamera::CameraKind::Arcball: return impl::arcballMousePreset();
        case core::ICamera::CameraKind::Turntable: return impl::turntableMousePreset();
        case core::ICamera::CameraKind::TopDown: return impl::topDownMousePreset();
        case core::ICamera::CameraKind::Isometric: return impl::isometricMousePreset();
        case core::ICamera::CameraKind::Chase: return impl::chaseMousePreset();
        case core::ICamera::CameraKind::Dolly: return impl::dollyMousePreset();
        case core::ICamera::CameraKind::DollyZoom: return impl::dollyZoomMousePreset();
        case core::ICamera::CameraKind::Path: return impl::pathMousePreset();
        default: return impl::emptyMousePreset();
    }
}

inline IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const core::ICamera& camera)
{
    return impl::makeImguizmoPreset(camera.getAllowedVirtualEvents());
}

inline SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera& camera)
{
    SCameraInputBindingPreset preset;
    preset.keyboard = getDefaultCameraKeyboardMappingPreset(camera);
    preset.mouse = getDefaultCameraMouseMappingPreset(camera);
    preset.imguizmo = buildDefaultCameraImguizmoMappingPreset(camera);
    return preset;
}

inline void applyDefaultCameraInputBindingPreset(IGimbalBindingLayout& layout, const core::ICamera& camera)
{
    const auto preset = buildDefaultCameraInputBindingPreset(camera);
    layout.updateKeyboardMapping([&](auto& map) { map = preset.keyboard; });
    layout.updateMouseMapping([&](auto& map) { map = preset.mouse; });
    layout.updateImguizmoMapping([&](auto& map) { map = preset.imguizmo; });
}

} // namespace nbl::ui

#endif // _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
