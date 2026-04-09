#ifndef _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
#define _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_

#include <array>
#include <tuple>

#include "CCameraKindUtilities.hpp"
#include "ICamera.hpp"
#include "IGimbalBindingLayout.hpp"

namespace nbl::ui
{

template<typename T, size_t N, size_t M>
inline constexpr std::array<T, N + M> concatBindingCodeArrays(
    const std::array<T, N>& lhs,
    const std::array<T, M>& rhs)
{
    std::array<T, N + M> output = {};
    for (size_t i = 0u; i < N; ++i)
        output[i] = lhs[i];
    for (size_t i = 0u; i < M; ++i)
        output[N + i] = rhs[i];
    return output;
}

template<typename T, size_t N, size_t M, size_t... RestN>
inline constexpr auto concatBindingCodeArrays(
    const std::array<T, N>& lhs,
    const std::array<T, M>& rhs,
    const std::array<T, RestN>&... rest)
{
    if constexpr (sizeof...(rest) == 0u)
        return concatBindingCodeArrays(lhs, rhs);
    else
        return concatBindingCodeArrays(concatBindingCodeArrays(lhs, rhs), rest...);
}

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
    static inline constexpr auto KeyboardProbeCodes = concatBindingCodeArrays(
        KeyboardProbeMoveCodes,
        KeyboardQeCodes,
        KeyboardProbeLookCodes,
        KeyboardProbeExtraCodes);
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

template<typename Map, typename Codes>
inline bool containsBindingForAnyCode(const Map& preset, const Codes& codes)
{
    for (const auto code : codes)
    {
        if (preset.find(code) != preset.end())
            return true;
    }
    return false;
}

template<typename Map, typename... Codes>
inline bool containsBindingForAnyCodeGroups(const Map& preset, const Codes&... codes)
{
    return (containsBindingForAnyCode(preset, codes) || ...);
}

inline bool hasMouseRelativeMovementBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
{
    return containsBindingForAnyCodeGroups(mousePreset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes);
}

inline bool hasMouseScrollBinding(const IGimbalBindingLayout::mouse_to_virtual_events_t& mousePreset)
{
    return containsBindingForAnyCodeGroups(
        mousePreset,
        SCameraInputBindingPhysicalGroups::PositiveScrollCodes,
        SCameraInputBindingPhysicalGroups::NegativeScrollCodes);
}

namespace impl
{
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

inline constexpr size_t interactionFamilyIndex(const core::ECameraInteractionFamily family)
{
    return static_cast<size_t>(family);
}

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

template<typename Map, typename Codes>
inline void appendMirroredBindingSpec(Map& preset, const Codes& codes, const virtual_event_t event)
{
    if (event == core::CVirtualGimbalEvent::None)
        return;

    std::array<virtual_event_t, std::tuple_size_v<Codes>> duplicatedEvents = {};
    duplicatedEvents.fill(event);
    appendBindingSpec(preset, codes, duplicatedEvents);
}

inline IGimbalBindingLayout::keyboard_to_virtual_events_t buildKeyboardPreset(const SKeyboardPresetSpec& spec)
{
    IGimbalBindingLayout::keyboard_to_virtual_events_t preset;
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardWasdCodes, spec.wasd);
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardQeCodes, spec.qe);
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::KeyboardIjklCodes, spec.ijkl);
    return preset;
}

inline IGimbalBindingLayout::mouse_to_virtual_events_t buildMousePreset(const SMousePresetSpec& spec)
{
    IGimbalBindingLayout::mouse_to_virtual_events_t preset;
    appendBindingSpec(preset, SCameraInputBindingPhysicalGroups::RelativeMouseCodes, spec.relative);
    appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::PositiveScrollCodes, spec.scroll[0]);
    appendMirroredBindingSpec(preset, SCameraInputBindingPhysicalGroups::NegativeScrollCodes, spec.scroll[1]);
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

inline constexpr SKeyboardPresetSpec makeKeyboardPresetSpec(
    const keyboard_axis_group_t& wasd = {},
    const scalar_axis_pair_t& qe = {},
    const keyboard_axis_group_t& ijkl = {})
{
    return {
        .wasd = wasd,
        .qe = qe,
        .ijkl = ijkl
    };
}

inline constexpr SKeyboardPresetSpec withKeyboardQe(
    const SKeyboardPresetSpec& base,
    const scalar_axis_pair_t& qe)
{
    auto preset = base;
    preset.qe = qe;
    return preset;
}

inline constexpr SKeyboardPresetSpec withKeyboardIjkl(
    const SKeyboardPresetSpec& base,
    const keyboard_axis_group_t& ijkl)
{
    auto preset = base;
    preset.ijkl = ijkl;
    return preset;
}

inline constexpr SMousePresetSpec makeMousePresetSpec(
    const mouse_axis_group_t& relative = {},
    const scalar_axis_pair_t& scroll = {})
{
    return {
        .relative = relative,
        .scroll = scroll
    };
}

inline constexpr SMousePresetSpec withMouseScroll(
    const SMousePresetSpec& base,
    const scalar_axis_pair_t& scroll)
{
    auto preset = base;
    preset.scroll = scroll;
    return preset;
}

inline constexpr SCameraInteractionBindingSpec makeInteractionBindingSpec(
    const SKeyboardPresetSpec& keyboard = {},
    const SMousePresetSpec& mouse = {})
{
    return {
        .keyboard = keyboard,
        .mouse = mouse
    };
}

inline constexpr SCameraInteractionBindingSpec withInteractionKeyboard(
    const SCameraInteractionBindingSpec& base,
    const SKeyboardPresetSpec& keyboard)
{
    auto spec = base;
    spec.keyboard = keyboard;
    return spec;
}

inline constexpr SCameraInteractionBindingSpec withInteractionMouse(
    const SCameraInteractionBindingSpec& base,
    const SMousePresetSpec& mouse)
{
    auto spec = base;
    spec.mouse = mouse;
    return spec;
}

inline constexpr SCameraInteractionBindingSpec EmptyInteractionBindingSpec = makeInteractionBindingSpec();

inline constexpr SKeyboardPresetSpec FpsKeyboardSpec = makeKeyboardPresetSpec(
    SCameraInputBindingEventGroups::FpsMove,
    {},
    SCameraInputBindingEventGroups::LookYawPitch);

inline constexpr SKeyboardPresetSpec FreeKeyboardSpec = withKeyboardQe(
    FpsKeyboardSpec,
    SCameraInputBindingEventGroups::Roll);

inline constexpr SKeyboardPresetSpec OrbitKeyboardSpec = makeKeyboardPresetSpec(
    SCameraInputBindingEventGroups::OrbitTranslate,
    SCameraInputBindingEventGroups::OrbitZoom);

inline constexpr SKeyboardPresetSpec TargetRigKeyboardSpec = withKeyboardQe(
    FpsKeyboardSpec,
    SCameraInputBindingEventGroups::VerticalMove);

inline constexpr SKeyboardPresetSpec TurntableKeyboardSpec = makeKeyboardPresetSpec(
    SCameraInputBindingEventGroups::TurntableMove,
    {},
    FpsKeyboardSpec.ijkl);

inline constexpr SKeyboardPresetSpec TopDownKeyboardSpec = withKeyboardIjkl(
    OrbitKeyboardSpec,
    SCameraInputBindingEventGroups::PanOnly);

inline constexpr SKeyboardPresetSpec PathKeyboardSpec = withKeyboardQe(
    makeKeyboardPresetSpec(SCameraInputBindingEventGroups::PathRigAdvanceAndRadius),
    SCameraInputBindingEventGroups::PathRigHeight);

inline constexpr SMousePresetSpec FpsMouseSpec = makeMousePresetSpec(
    SCameraInputBindingEventGroups::RelativeLook);

inline constexpr SMousePresetSpec OrbitMouseSpec = makeMousePresetSpec(
    SCameraInputBindingEventGroups::RelativeOrbitTranslate,
    SCameraInputBindingEventGroups::OrbitZoom);

inline constexpr SMousePresetSpec TargetRigMouseSpec = withMouseScroll(FpsMouseSpec, OrbitMouseSpec.scroll);

inline constexpr SMousePresetSpec TopDownMouseSpec = makeMousePresetSpec(
    SCameraInputBindingEventGroups::RelativeTopDown,
    OrbitMouseSpec.scroll);

inline constexpr SCameraInteractionBindingSpec FpsInteractionBindingSpec = makeInteractionBindingSpec(
    FpsKeyboardSpec,
    FpsMouseSpec);

inline constexpr SCameraInteractionBindingSpec FreeInteractionBindingSpec = withInteractionKeyboard(
    FpsInteractionBindingSpec,
    FreeKeyboardSpec);

inline constexpr SCameraInteractionBindingSpec OrbitInteractionBindingSpec = makeInteractionBindingSpec(
    OrbitKeyboardSpec,
    OrbitMouseSpec);

inline constexpr SCameraInteractionBindingSpec TargetRigInteractionBindingSpec = withInteractionKeyboard(
    withInteractionMouse(FpsInteractionBindingSpec, TargetRigMouseSpec),
    TargetRigKeyboardSpec);

inline constexpr SCameraInteractionBindingSpec TurntableInteractionBindingSpec = withInteractionKeyboard(
    TargetRigInteractionBindingSpec,
    TurntableKeyboardSpec);

inline constexpr SCameraInteractionBindingSpec TopDownInteractionBindingSpec = withInteractionKeyboard(
    withInteractionMouse(OrbitInteractionBindingSpec, TopDownMouseSpec),
    TopDownKeyboardSpec);

inline constexpr SCameraInteractionBindingSpec PathInteractionBindingSpec = withInteractionKeyboard(
    OrbitInteractionBindingSpec,
    PathKeyboardSpec);

template<typename Map, typename SpecArray, typename Builder>
inline auto makePresetCache(const SpecArray& specs, Builder&& builder)
{
    std::array<Map, std::tuple_size_v<SpecArray>> cache = {};
    for (size_t i = 0u; i < specs.size(); ++i)
        cache[i] = builder(specs[i]);
    return cache;
}

inline SCameraMappedInteractionBindingSpec mapInteractionBindingSpec(const SCameraInteractionBindingSpec& spec)
{
    return {
        .keyboard = buildKeyboardPreset(spec.keyboard),
        .mouse = buildMousePreset(spec.mouse)
    };
}

inline constexpr std::array<SCameraInteractionBindingSpec, interactionFamilyIndex(core::ECameraInteractionFamily::Path) + 1u> InteractionFamilyPresetSpecs = {{
    EmptyInteractionBindingSpec,
    FpsInteractionBindingSpec,
    FreeInteractionBindingSpec,
    OrbitInteractionBindingSpec,
    TargetRigInteractionBindingSpec,
    TurntableInteractionBindingSpec,
    TopDownInteractionBindingSpec,
    PathInteractionBindingSpec
}};

inline const SCameraMappedInteractionBindingSpec& interactionBindingPresetForKind(const core::ICamera::CameraKind kind)
{
    const auto familyIx = interactionFamilyIndex(core::getCameraInteractionFamily(kind));
    static const auto cache = makePresetCache<SCameraMappedInteractionBindingSpec>(
        InteractionFamilyPresetSpecs,
        [](const SCameraInteractionBindingSpec& spec) { return mapInteractionBindingSpec(spec); });
    return cache[familyIx < cache.size() ? familyIx : 0u];
}

} // namespace impl

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera::CameraKind kind)
{
    return impl::interactionBindingPresetForKind(kind).keyboard;
}

inline const IGimbalBindingLayout::keyboard_to_virtual_events_t& getDefaultCameraKeyboardMappingPreset(const core::ICamera& camera)
{
    return getDefaultCameraKeyboardMappingPreset(camera.getKind());
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera::CameraKind kind)
{
    return impl::interactionBindingPresetForKind(kind).mouse;
}

inline const IGimbalBindingLayout::mouse_to_virtual_events_t& getDefaultCameraMouseMappingPreset(const core::ICamera& camera)
{
    return getDefaultCameraMouseMappingPreset(camera.getKind());
}

inline IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const uint32_t allowedVirtualEvents)
{
    return impl::makeImguizmoPreset(allowedVirtualEvents);
}

inline IGimbalBindingLayout::imguizmo_to_virtual_events_t buildDefaultCameraImguizmoMappingPreset(const core::ICamera& camera)
{
    return buildDefaultCameraImguizmoMappingPreset(camera.getAllowedVirtualEvents());
}

inline SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera::CameraKind kind, const uint32_t allowedVirtualEvents)
{
    SCameraInputBindingPreset preset;
    preset.keyboard = getDefaultCameraKeyboardMappingPreset(kind);
    preset.mouse = getDefaultCameraMouseMappingPreset(kind);
    preset.imguizmo = buildDefaultCameraImguizmoMappingPreset(allowedVirtualEvents);
    return preset;
}

inline SCameraInputBindingPreset buildDefaultCameraInputBindingPreset(const core::ICamera& camera)
{
    return buildDefaultCameraInputBindingPreset(camera.getKind(), camera.getAllowedVirtualEvents());
}

inline void applyDefaultCameraInputBindingPreset(
    IGimbalBindingLayout& layout,
    const core::ICamera::CameraKind kind,
    const uint32_t allowedVirtualEvents)
{
    const auto preset = buildDefaultCameraInputBindingPreset(kind, allowedVirtualEvents);
    layout.updateKeyboardMapping([&](auto& map) { map = preset.keyboard; });
    layout.updateMouseMapping([&](auto& map) { map = preset.mouse; });
    layout.updateImguizmoMapping([&](auto& map) { map = preset.imguizmo; });
}

inline void applyDefaultCameraInputBindingPreset(IGimbalBindingLayout& layout, const core::ICamera& camera)
{
    applyDefaultCameraInputBindingPreset(layout, camera.getKind(), camera.getAllowedVirtualEvents());
}

} // namespace nbl::ui

#endif // _NBL_C_CAMERA_INPUT_BINDING_UTILITIES_HPP_
