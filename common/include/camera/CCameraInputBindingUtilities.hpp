#ifndef _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
#define _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_

#include <array>

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

struct SKeyboardPresetSpec
{
    std::array<core::CVirtualGimbalEvent::VirtualEventType, 4u> wasd = {
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None
    };
    std::array<core::CVirtualGimbalEvent::VirtualEventType, 2u> qe = {
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None
    };
    std::array<core::CVirtualGimbalEvent::VirtualEventType, 4u> ijkl = {
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None
    };
};

struct SMousePresetSpec
{
    std::array<core::CVirtualGimbalEvent::VirtualEventType, 4u> relative = {
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None
    };
    std::array<core::CVirtualGimbalEvent::VirtualEventType, 2u> scroll = {
        core::CVirtualGimbalEvent::None,
        core::CVirtualGimbalEvent::None
    };
};

template<typename Map, typename Codes, typename Events>
inline void appendBindingSpec(Map& preset, const Codes& codes, const Events& events)
{
    for (size_t i = 0u; i < codes.size() && i < events.size(); ++i)
    {
        const auto event = events[i];
        if (event == core::CVirtualGimbalEvent::None)
            continue;
        preset.emplace(codes[i], IGimbalBindingLayout::CHashInfo(event));
    }
}

inline IGimbalBindingLayout::keyboard_to_virtual_events_t buildKeyboardPreset(const SKeyboardPresetSpec& spec)
{
    static constexpr std::array KeyboardWasdCodes = {
        ui::E_KEY_CODE::EKC_W,
        ui::E_KEY_CODE::EKC_S,
        ui::E_KEY_CODE::EKC_A,
        ui::E_KEY_CODE::EKC_D
    };
    static constexpr std::array KeyboardQeCodes = {
        ui::E_KEY_CODE::EKC_Q,
        ui::E_KEY_CODE::EKC_E
    };
    static constexpr std::array KeyboardIjklCodes = {
        ui::E_KEY_CODE::EKC_I,
        ui::E_KEY_CODE::EKC_K,
        ui::E_KEY_CODE::EKC_J,
        ui::E_KEY_CODE::EKC_L
    };

    IGimbalBindingLayout::keyboard_to_virtual_events_t preset;
    appendBindingSpec(preset, KeyboardWasdCodes, spec.wasd);
    appendBindingSpec(preset, KeyboardQeCodes, spec.qe);
    appendBindingSpec(preset, KeyboardIjklCodes, spec.ijkl);
    return preset;
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t buildMousePreset(const SMousePresetSpec& spec)
{
    static constexpr std::array RelativeMouseCodes = {
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y
    };
    static constexpr std::array PositiveScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL
    };
    static constexpr std::array NegativeScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL
    };

    IGimbalBindingLayout::mouse_to_virtual_events_t preset;
    appendBindingSpec(preset, RelativeMouseCodes, spec.relative);
    if (spec.scroll[0] != core::CVirtualGimbalEvent::None)
    {
        appendBindingSpec(
            preset,
            PositiveScrollCodes,
            std::array<core::CVirtualGimbalEvent::VirtualEventType, 2u>{ spec.scroll[0], spec.scroll[0] });
    }
    if (spec.scroll[1] != core::CVirtualGimbalEvent::None)
    {
        appendBindingSpec(
            preset,
            NegativeScrollCodes,
            std::array<core::CVirtualGimbalEvent::VirtualEventType, 2u>{ spec.scroll[1], spec.scroll[1] });
    }
    return preset;
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

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& keyboardPresetForKind(const core::ICamera::CameraKind kind)
{
    switch (kind)
    {
        case core::ICamera::CameraKind::FPS:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .ijkl = {
                    core::CVirtualGimbalEvent::TiltDown,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Free:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::RollLeft,
                    core::CVirtualGimbalEvent::RollRight
                },
                .ijkl = {
                    core::CVirtualGimbalEvent::TiltDown,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Orbit:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveForward
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Arcball:
        case core::ICamera::CameraKind::Chase:
        case core::ICamera::CameraKind::Dolly:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::MoveDown,
                    core::CVirtualGimbalEvent::MoveUp
                },
                .ijkl = {
                    core::CVirtualGimbalEvent::TiltDown,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Turntable:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                },
                .ijkl = {
                    core::CVirtualGimbalEvent::TiltDown,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::TopDown:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveForward
                },
                .ijkl = {
                    core::CVirtualGimbalEvent::None,
                    core::CVirtualGimbalEvent::None,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::PanRight
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Isometric:
        case core::ICamera::CameraKind::DollyZoom:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveForward
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Path:
        {
            static const auto preset = buildKeyboardPreset({
                .wasd = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveRight
                },
                .qe = {
                    core::CVirtualGimbalEvent::MoveDown,
                    core::CVirtualGimbalEvent::MoveUp
                }
            });
            return preset;
        }
        default:
            return emptyKeyboardPreset();
    }
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePresetForKind(const core::ICamera::CameraKind kind)
{
    switch (kind)
    {
        case core::ICamera::CameraKind::FPS:
        case core::ICamera::CameraKind::Free:
        {
            static const auto preset = buildMousePreset({
                .relative = {
                    core::CVirtualGimbalEvent::PanRight,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::TiltDown
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Orbit:
        case core::ICamera::CameraKind::Isometric:
        case core::ICamera::CameraKind::DollyZoom:
        case core::ICamera::CameraKind::Path:
        {
            static const auto preset = buildMousePreset({
                .relative = {
                    core::CVirtualGimbalEvent::MoveRight,
                    core::CVirtualGimbalEvent::MoveLeft,
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown
                },
                .scroll = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Arcball:
        case core::ICamera::CameraKind::Turntable:
        case core::ICamera::CameraKind::Dolly:
        {
            static const auto preset = buildMousePreset({
                .relative = {
                    core::CVirtualGimbalEvent::PanRight,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::TiltDown
                },
                .scroll = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::TopDown:
        {
            static const auto preset = buildMousePreset({
                .relative = {
                    core::CVirtualGimbalEvent::PanRight,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown
                },
                .scroll = {
                    core::CVirtualGimbalEvent::MoveForward,
                    core::CVirtualGimbalEvent::MoveBackward
                }
            });
            return preset;
        }
        case core::ICamera::CameraKind::Chase:
        {
            static const auto preset = buildMousePreset({
                .relative = {
                    core::CVirtualGimbalEvent::PanRight,
                    core::CVirtualGimbalEvent::PanLeft,
                    core::CVirtualGimbalEvent::TiltUp,
                    core::CVirtualGimbalEvent::TiltDown
                },
                .scroll = {
                    core::CVirtualGimbalEvent::MoveUp,
                    core::CVirtualGimbalEvent::MoveDown
                }
            });
            return preset;
        }
        default:
            return emptyMousePreset();
    }
}

} // namespace impl

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera& camera)
{
    return impl::keyboardPresetForKind(camera.getKind());
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera& camera)
{
    return impl::mousePresetForKind(camera.getKind());
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
