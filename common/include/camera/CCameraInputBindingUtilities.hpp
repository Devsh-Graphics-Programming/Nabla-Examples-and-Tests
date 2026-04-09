#ifndef _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
#define _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_

#include <array>
#include <tuple>

#include "CCameraKindUtilities.hpp"
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

//! Shared physical input bundles reused by default presets and smoke probing.
struct SCameraInputBindingPhysicalGroups final
{
    static inline constexpr std::array KeyboardWasdCodes = {
        ui::E_KEY_CODE::EKC_W,
        ui::E_KEY_CODE::EKC_S,
        ui::E_KEY_CODE::EKC_A,
        ui::E_KEY_CODE::EKC_D
    };
    static inline constexpr std::array KeyboardQeCodes = {
        ui::E_KEY_CODE::EKC_Q,
        ui::E_KEY_CODE::EKC_E
    };
    static inline constexpr std::array KeyboardIjklCodes = {
        ui::E_KEY_CODE::EKC_I,
        ui::E_KEY_CODE::EKC_K,
        ui::E_KEY_CODE::EKC_J,
        ui::E_KEY_CODE::EKC_L
    };
    static inline constexpr auto KeyboardProbeMoveCodes = KeyboardWasdCodes;
    static inline constexpr auto KeyboardProbeLookCodes = KeyboardIjklCodes;
    static inline constexpr std::array KeyboardProbeExtraCodes = {
        ui::E_KEY_CODE::EKC_U,
        ui::E_KEY_CODE::EKC_O
    };
    static inline constexpr std::array KeyboardProbeCodes = {
        ui::E_KEY_CODE::EKC_W,
        ui::E_KEY_CODE::EKC_S,
        ui::E_KEY_CODE::EKC_A,
        ui::E_KEY_CODE::EKC_D,
        ui::E_KEY_CODE::EKC_Q,
        ui::E_KEY_CODE::EKC_E,
        ui::E_KEY_CODE::EKC_I,
        ui::E_KEY_CODE::EKC_K,
        ui::E_KEY_CODE::EKC_J,
        ui::E_KEY_CODE::EKC_L,
        ui::E_KEY_CODE::EKC_U,
        ui::E_KEY_CODE::EKC_O
    };
    static inline constexpr std::array RelativeMouseCodes = {
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
        ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y,
        ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y
    };
    static inline constexpr std::array PositiveScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL
    };
    static inline constexpr std::array NegativeScrollCodes = {
        ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL,
        ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL
    };
};

struct CCameraInputBindingUtilities final
{
public:
    static inline bool hasMouseRelativeMovementBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
    {
        return containsBindingForAnyCodeGroups(mousePreset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes);
    }

    static inline bool hasMouseScrollBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
    {
        return containsBindingForAnyCodeGroups(
            mousePreset,
            SCameraInputBindingPhysicalGroups::PositiveScrollCodes,
            SCameraInputBindingPhysicalGroups::NegativeScrollCodes);
    }

    static inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera::CameraKind kind)
    {
        return interactionBindingPresetForKind(kind).keyboard;
    }

    static inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera& camera)
    {
        return getDefaultCameraKeyboardMappingPreset(camera.getKind());
    }

    static inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera::CameraKind kind)
    {
        return interactionBindingPresetForKind(kind).mouse;
    }

    static inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera& camera)
    {
        return getDefaultCameraMouseMappingPreset(camera.getKind());
    }

    static inline IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const uint32_t allowedVirtualEvents)
    {
        return makeImguizmoPreset(allowedVirtualEvents);
    }

    static inline IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const core::ICamera& camera)
    {
        return buildDefaultCameraImguizmoMappingPreset(camera.getAllowedVirtualEvents());
    }

    static inline SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera::CameraKind kind, const uint32_t allowedVirtualEvents)
    {
        SCameraInputBindingPreset preset;
        preset.keyboard = getDefaultCameraKeyboardMappingPreset(kind);
        preset.mouse = getDefaultCameraMouseMappingPreset(kind);
        preset.imguizmo = buildDefaultCameraImguizmoMappingPreset(allowedVirtualEvents);
        return preset;
    }

    static inline SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera& camera)
    {
        return buildDefaultCameraInputBindingPreset(camera.getKind(), camera.getAllowedVirtualEvents());
    }

    static inline void applyDefaultCameraInputBindingPreset(
        IGimbalBindingLayout& layout,
        const core::ICamera::CameraKind kind,
        const uint32_t allowedVirtualEvents)
    {
        const auto preset = buildDefaultCameraInputBindingPreset(kind, allowedVirtualEvents);
        layout.updateKeyboardMapping([&](auto& map) { map = preset.keyboard; });
        layout.updateMouseMapping([&](auto& map) { map = preset.mouse; });
        layout.updateImguizmoMapping([&](auto& map) { map = preset.imguizmo; });
    }

    static inline void applyDefaultCameraInputBindingPreset(IGimbalBindingLayout& layout, const core::ICamera& camera)
    {
        applyDefaultCameraInputBindingPreset(layout, camera.getKind(), camera.getAllowedVirtualEvents());
    }

private:
    using virtual_event_t = core::CVirtualGimbalEvent::VirtualEventType;
    using keyboard_axis_group_t = std::array<virtual_event_t, 4u>;
    using mouse_axis_group_t = std::array<virtual_event_t, 4u>;
    using scalar_axis_pair_t = std::array<virtual_event_t, 2u>;

    struct SKeyboardPresetSpec final
    {
        keyboard_axis_group_t wasd = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None
        };
        scalar_axis_pair_t qe = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None
        };
        keyboard_axis_group_t ijkl = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None
        };
    };

    struct SMousePresetSpec final
    {
        mouse_axis_group_t relative = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None
        };
        scalar_axis_pair_t scroll = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None
        };
    };

    //! Shared virtual-event bundles reused across interaction families.
    struct SCameraInputBindingEventGroups final
    {
        static inline constexpr std::array FpsMove = {
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight
        };
        static inline constexpr std::array OrbitTranslate = {
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight
        };
        static inline constexpr std::array OrbitZoom = {
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward
        };
        static inline constexpr std::array VerticalMove = {
            core::CVirtualGimbalEvent::MoveDown,
            core::CVirtualGimbalEvent::MoveUp
        };
        static inline constexpr std::array PathRigAdvanceAndRadius = {
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveRight
        };
        static inline constexpr std::array PathRigHeight = VerticalMove;
        static inline constexpr std::array TurntableMove = {
            core::CVirtualGimbalEvent::MoveForward,
            core::CVirtualGimbalEvent::MoveBackward,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::PanRight
        };
        static inline constexpr std::array LookYawPitch = {
            core::CVirtualGimbalEvent::TiltDown,
            core::CVirtualGimbalEvent::TiltUp,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::PanRight
        };
        static inline constexpr std::array Roll = {
            core::CVirtualGimbalEvent::RollLeft,
            core::CVirtualGimbalEvent::RollRight
        };
        static inline constexpr std::array PanOnly = {
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::None,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::PanRight
        };
        static inline constexpr std::array RelativeLook = {
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::TiltUp,
            core::CVirtualGimbalEvent::TiltDown
        };
        static inline constexpr std::array RelativeOrbitTranslate = {
            core::CVirtualGimbalEvent::MoveRight,
            core::CVirtualGimbalEvent::MoveLeft,
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown
        };
        static inline constexpr std::array RelativeTopDown = {
            core::CVirtualGimbalEvent::PanRight,
            core::CVirtualGimbalEvent::PanLeft,
            core::CVirtualGimbalEvent::MoveUp,
            core::CVirtualGimbalEvent::MoveDown
        };
    };

    struct SCameraInteractionBindingSpec
    {
        SKeyboardPresetSpec keyboard = {};
        SMousePresetSpec mouse = {};
    };

    struct SCameraMappedInteractionBindingSpec
    {
        IGimbalBindingLayout::keyboard_to_virtual_events_t keyboard;
        IGimbalBindingLayout::mouse_to_virtual_events_t mouse;
    };

    template<typename Map, typename Codes>
    static inline bool containsBindingForAnyCode(const Map& preset, const Codes& codes)
    {
        for (const auto code : codes)
        {
            if (preset.find(code) != preset.end())
                return true;
        }
        return false;
    }

    template<typename Map, typename... Codes>
    static inline bool containsBindingForAnyCodeGroups(const Map& preset, const Codes&... codes)
    {
        return (containsBindingForAnyCode(preset, codes) || ...);
    }

    static inline constexpr size_t interactionFamilyIndex(const core::ECameraInteractionFamily family)
    {
        return static_cast<size_t>(family);
    }

    template<typename Map, typename Codes, typename Events>
    static inline void appendBindingSpec(Map& preset, const Codes& codes, const Events& events)
    {
        for (size_t i = 0u; i < codes.size() && i < events.size(); ++i)
        {
            const auto event = events[i];
            if (event == core::CVirtualGimbalEvent::None)
                continue;
            preset.emplace(codes[i], IGimbalBindingLayout::CHashInfo(event));
        }
    }

    template<typename Map, typename Codes>
    static inline void appendMirroredBindingSpec(Map& preset, const Codes& codes, const virtual_event_t event)
    {
        if (event == core::CVirtualGimbalEvent::None)
            return;

        std::array<virtual_event_t, std::tuple_size_v<Codes>> duplicatedEvents = {};
        duplicatedEvents.fill(event);
        appendBindingSpec(preset, codes, duplicatedEvents);
    }

    static inline IGimbalBindingLayout::keyboard_to_virtual_events_t buildKeyboardPreset(const SKeyboardPresetSpec& spec)
    {
        IGimbalBindingLayout::keyboard_to_virtual_events_t preset;
        appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardWasdCodes, spec.wasd);
        appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardQeCodes, spec.qe);
        appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardIjklCodes, spec.ijkl);
        return preset;
    }

    static inline IGimbalBindingLayout::mouse_to_virtual_events_t buildMousePreset(const SMousePresetSpec& spec)
    {
        IGimbalBindingLayout::mouse_to_virtual_events_t preset;
        appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes, spec.relative);
        appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::PositiveScrollCodes, spec.scroll[0]);
        appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::NegativeScrollCodes, spec.scroll[1]);
        return preset;
    }

    static inline IGimbalBindingLayout::imguizmo_to_virtual_events_t makeImguizmoPreset(const uint32_t allowedVirtualEvents)
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

    static inline constexpr SCameraInteractionBindingSpec EmptyInteractionBindingSpec = {};

    static inline constexpr SKeyboardPresetSpec FpsKeyboardSpec = {
        SCameraInputBindingEventGroups::FpsMove,
        {},
        SCameraInputBindingEventGroups::LookYawPitch
    };

    static inline constexpr SKeyboardPresetSpec FreeKeyboardSpec = {
        FpsKeyboardSpec.wasd,
        SCameraInputBindingEventGroups::Roll,
        FpsKeyboardSpec.ijkl
    };

    static inline constexpr SKeyboardPresetSpec OrbitKeyboardSpec = {
        SCameraInputBindingEventGroups::OrbitTranslate,
        SCameraInputBindingEventGroups::OrbitZoom,
        {}
    };

    static inline constexpr SKeyboardPresetSpec TargetRigKeyboardSpec = {
        FpsKeyboardSpec.wasd,
        SCameraInputBindingEventGroups::VerticalMove,
        FpsKeyboardSpec.ijkl
    };

    static inline constexpr SKeyboardPresetSpec TurntableKeyboardSpec = {
        SCameraInputBindingEventGroups::TurntableMove,
        {},
        FpsKeyboardSpec.ijkl
    };

    static inline constexpr SKeyboardPresetSpec TopDownKeyboardSpec = {
        OrbitKeyboardSpec.wasd,
        OrbitKeyboardSpec.qe,
        SCameraInputBindingEventGroups::PanOnly
    };

    static inline constexpr SKeyboardPresetSpec PathKeyboardSpec = {
        SCameraInputBindingEventGroups::PathRigAdvanceAndRadius,
        SCameraInputBindingEventGroups::PathRigHeight,
        {}
    };

    static inline constexpr SMousePresetSpec FpsMouseSpec = {
        SCameraInputBindingEventGroups::RelativeLook,
        {}
    };

    static inline constexpr SMousePresetSpec OrbitMouseSpec = {
        SCameraInputBindingEventGroups::RelativeOrbitTranslate,
        SCameraInputBindingEventGroups::OrbitZoom
    };

    static inline constexpr SMousePresetSpec TargetRigMouseSpec = {
        FpsMouseSpec.relative,
        OrbitMouseSpec.scroll
    };

    static inline constexpr SMousePresetSpec TopDownMouseSpec = {
        SCameraInputBindingEventGroups::RelativeTopDown,
        OrbitMouseSpec.scroll
    };

    static inline constexpr SCameraInteractionBindingSpec FpsInteractionBindingSpec = {
        FpsKeyboardSpec,
        FpsMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec FreeInteractionBindingSpec = {
        FreeKeyboardSpec,
        FpsMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec OrbitInteractionBindingSpec = {
        OrbitKeyboardSpec,
        OrbitMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec TargetRigInteractionBindingSpec = {
        TargetRigKeyboardSpec,
        TargetRigMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec TurntableInteractionBindingSpec = {
        TurntableKeyboardSpec,
        TargetRigMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec TopDownInteractionBindingSpec = {
        TopDownKeyboardSpec,
        TopDownMouseSpec
    };

    static inline constexpr SCameraInteractionBindingSpec PathInteractionBindingSpec = {
        PathKeyboardSpec,
        OrbitMouseSpec
    };

    template<typename Map, typename SpecArray, typename Builder>
    static inline auto makePresetCache(const SpecArray& specs, Builder&& builder)
    {
        std::array<Map, std::tuple_size_v<SpecArray>> cache = {};
        for (size_t i = 0u; i < specs.size(); ++i)
            cache[i] = builder(specs[i]);
        return cache;
    }

    static inline SCameraMappedInteractionBindingSpec mapInteractionBindingSpec(const SCameraInteractionBindingSpec& spec)
    {
        return {
            .keyboard = buildKeyboardPreset(spec.keyboard),
            .mouse = buildMousePreset(spec.mouse)
        };
    }

    static inline constexpr std::array<SCameraInteractionBindingSpec, 8u> InteractionFamilyPresetSpecs = {{
        EmptyInteractionBindingSpec,
        FpsInteractionBindingSpec,
        FreeInteractionBindingSpec,
        OrbitInteractionBindingSpec,
        TargetRigInteractionBindingSpec,
        TurntableInteractionBindingSpec,
        TopDownInteractionBindingSpec,
        PathInteractionBindingSpec
    }};

    static inline const SCameraMappedInteractionBindingSpec& interactionBindingPresetForKind(const core::ICamera::CameraKind kind)
    {
        const auto familyIx = interactionFamilyIndex(core::CCameraKindUtilities::getCameraInteractionFamily(kind));
        static const auto cache = makePresetCache<SCameraMappedInteractionBindingSpec>(
            InteractionFamilyPresetSpecs,
            [](const SCameraInteractionBindingSpec& spec) { return mapInteractionBindingSpec(spec); });
        return cache[familyIx < cache.size() ? familyIx : 0u];
    }
};

} // namespace nbl::ui

#endif // _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
