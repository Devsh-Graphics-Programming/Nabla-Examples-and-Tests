#include "camera/CCameraScriptedRuntimePersistence.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

#include "nbl/ext/Cameras/CCameraFileUtilities.hpp"
#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"
#include "nbl/ext/Cameras/CCameraVirtualEventUtilities.hpp"
#include "nlohmann/json.hpp"

using json_t = nlohmann::json;

namespace nbl::this_example
{

namespace impl
{

template<typename T>
void readVector3(const json_t& entry, T& outValue)
{
    using scalar_t = std::remove_reference_t<decltype(outValue.x)>;
    const auto values = entry.get<std::array<scalar_t, 3>>();
    outValue = T(values[0], values[1], values[2]);
}

nbl::hlsl::float32_t4x4 composeScriptedImguizmoTransform(
    const std::array<float, 3>& translation,
    const std::array<float, 3>& rotationDeg,
    const std::array<float, 3>& scale)
{
    return nbl::hlsl::CCameraMathUtilities::composeTransformMatrix(
        nbl::hlsl::float32_t3(translation[0], translation[1], translation[2]),
        nbl::hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegrees(nbl::hlsl::float32_t3(rotationDeg[0], rotationDeg[1], rotationDeg[2])),
        nbl::hlsl::float32_t3(scale[0], scale[1], scale[2]));
}

nbl::hlsl::float32_t4x4 makeScriptedMatrixFromArray(const std::array<float, 16>& values)
{
    nbl::hlsl::float32_t4x4 out(1.f);
    for (uint32_t column = 0u; column < 4u; ++column)
    {
        for (uint32_t row = 0u; row < 4u; ++row)
            out[column][row] = values[column * 4u + row];
    }
    return out;
}

std::optional<nbl::system::CCameraScriptedInputEvent::KeyboardData::Action> parseScriptedKeyboardAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return nbl::system::CCameraScriptedInputEvent::KeyboardData::Action::Pressed;
    if (action == "released" || action == "release")
        return nbl::system::CCameraScriptedInputEvent::KeyboardData::Action::Released;
    return std::nullopt;
}

nbl::ui::E_KEY_CODE parseScriptedKeyCode(std::string_view key)
{
    auto parsed = nbl::ui::stringToKeyCode(key);
    if (parsed != nbl::ui::EKC_NONE)
        return parsed;

    constexpr std::string_view KeyPrefix = "KEY_";
    constexpr std::string_view EkcPrefix = "EKC_";
    if (key.starts_with(KeyPrefix))
        parsed = nbl::ui::stringToKeyCode(key.substr(KeyPrefix.size()));
    if (parsed == nbl::ui::EKC_NONE && key.starts_with(EkcPrefix))
        parsed = nbl::ui::stringToKeyCode(key.substr(EkcPrefix.size()));
    return parsed;
}

std::optional<nbl::ui::E_MOUSE_BUTTON> parseScriptedMouseButton(std::string_view button)
{
    auto tryParseCode = [](std::string_view code) -> std::optional<nbl::ui::E_MOUSE_BUTTON>
    {
        switch (nbl::ui::stringToMouseCode(code))
        {
            case nbl::ui::EMC_LEFT_BUTTON:
                return nbl::ui::EMB_LEFT_BUTTON;
            case nbl::ui::EMC_RIGHT_BUTTON:
                return nbl::ui::EMB_RIGHT_BUTTON;
            case nbl::ui::EMC_MIDDLE_BUTTON:
                return nbl::ui::EMB_MIDDLE_BUTTON;
            default:
                return std::nullopt;
        }
    };

    auto parsed = tryParseCode(button);
    if (parsed.has_value())
        return parsed;

    constexpr std::string_view ButtonPrefix = "BUTTON_";
    constexpr std::string_view EmbPrefix = "EMB_";
    if (button.starts_with(ButtonPrefix))
        parsed = tryParseCode(button.substr(ButtonPrefix.size()));
    if (!parsed.has_value() && button.starts_with(EmbPrefix))
        parsed = tryParseCode(button.substr(EmbPrefix.size()));

    return parsed;
}

std::optional<nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction> parseScriptedMouseClickAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction::Pressed;
    if (action == "released" || action == "release")
        return nbl::system::CCameraScriptedInputEvent::MouseData::ClickAction::Released;
    return std::nullopt;
}

void appendScriptedCaptureFrame(
    nbl::this_example::CCameraScriptedInputParseResult& out,
    const uint64_t frame,
    const bool captureFrame)
{
    if (captureFrame)
        out.timeline.captureFrames.emplace_back(frame);
}

void parseScriptedCaptureFramesJson(const json_t& script, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("capture_frames") || !script["capture_frames"].is_array())
        return;

    for (const auto& entry : script["capture_frames"])
    {
        if (entry.is_number_unsigned())
            out.timeline.captureFrames.emplace_back(entry.get<uint64_t>());
    }
}

void parseScriptedControlOverridesJson(const json_t& controls, nbl::this_example::CCameraScriptedControlOverrides& out)
{
    if (!controls.is_object())
        return;

    if (controls.contains("keyboard_scale"))
    {
        out.keyboardScale = controls["keyboard_scale"].get<float>();
        out.hasKeyboardScale = true;
    }
    if (controls.contains("mouse_move_scale"))
    {
        out.mouseMoveScale = controls["mouse_move_scale"].get<float>();
        out.hasMouseMoveScale = true;
    }
    if (controls.contains("mouse_scroll_scale"))
    {
        out.mouseScrollScale = controls["mouse_scroll_scale"].get<float>();
        out.hasMouseScrollScale = true;
    }
    if (controls.contains("translation_scale"))
    {
        out.translationScale = controls["translation_scale"].get<float>();
        out.hasTranslationScale = true;
    }
    if (controls.contains("rotation_scale"))
    {
        out.rotationScale = controls["rotation_scale"].get<float>();
        out.hasRotationScale = true;
    }
}

bool parseScriptedSequenceIfPresentJson(const json_t& script, nbl::this_example::CCameraScriptedInputParseResult& out, std::string* error)
{
    nbl::core::CCameraSequenceScript sequence;
    if (script.contains("segments"))
    {
        if (!nbl::system::CCameraSequenceScriptPersistenceUtilities::deserializeCameraSequenceScript(script.dump(), sequence, error))
            return false;
        out.sequence = std::move(sequence);
        return true;
    }

    if (script.contains("sequence"))
    {
        if (!nbl::system::CCameraSequenceScriptPersistenceUtilities::deserializeCameraSequenceScript(script["sequence"].dump(), sequence, error))
            return false;
        out.sequence = std::move(sequence);
    }

    return true;
}

void parseScriptedKeyboardEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("key") || !event.contains("action"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event missing \"key\" or \"action\".");
        return;
    }

    const auto keyText = event["key"].get<std::string>();
    const auto actionText = event["action"].get<std::string>();
    const auto key = parseScriptedKeyCode(keyText);
    if (key == nbl::ui::EKC_NONE)
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid key \"" + keyText + "\".");
        return;
    }

    const auto action = parseScriptedKeyboardAction(actionText);
    if (!action.has_value())
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid action \"" + actionText + "\".");
        return;
    }

    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Keyboard;
    entry.keyboard.key = key;
    entry.keyboard.action = action.value();
    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedMouseEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("kind"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted mouse event missing \"kind\".");
        return;
    }

    const auto kind = event["kind"].get<std::string>();
    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Mouse;

    if (kind == "move")
    {
        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Movement;
        entry.mouse.delta = nbl::hlsl::int16_t2(event.value("dx", 0), event.value("dy", 0));
    }
    else if (kind == "scroll")
    {
        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Scroll;
        entry.mouse.scroll = nbl::hlsl::int16_t2(event.value("v", 0), event.value("h", 0));
    }
    else if (kind == "click")
    {
        if (!event.contains("button") || !event.contains("action"))
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event missing \"button\" or \"action\".");
            return;
        }

        const auto buttonText = event["button"].get<std::string>();
        const auto actionText = event["action"].get<std::string>();
        const auto button = parseScriptedMouseButton(buttonText);
        if (!button.has_value())
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event has invalid button \"" + buttonText + "\".");
            return;
        }

        const auto action = parseScriptedMouseClickAction(actionText);
        if (!action.has_value())
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted click event has invalid action \"" + actionText + "\".");
            return;
        }

        entry.mouse.type = nbl::system::CCameraScriptedInputEvent::MouseData::Type::Click;
        entry.mouse.button = button.value();
        entry.mouse.action = action.value();
        entry.mouse.position = nbl::hlsl::int16_t2(event.value("x", 0), event.value("y", 0));
    }
    else
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted mouse event has invalid kind \"" + kind + "\".");
        return;
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedImguizmoEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    nbl::system::CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = nbl::system::CCameraScriptedInputEvent::Type::Imguizmo;

    if (event.contains("delta_trs"))
    {
        const auto matrix = event["delta_trs"].get<std::array<float, 16>>();
        entry.imguizmo = makeScriptedMatrixFromArray(matrix);
    }
    else
    {
        const auto translation = event.contains("translation") ? event["translation"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto rotation = event.contains("rotation_deg") ? event["rotation_deg"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto scale = event.contains("scale") ? event["scale"].get<std::array<float, 3>>() : std::array<float, 3>{1.f, 1.f, 1.f};
        entry.imguizmo = composeScriptedImguizmoTransform(translation, rotation, scale);
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

int32_t parseScriptedActionIntValue(const json_t& event)
{
    if (event.contains("value"))
        return event["value"].get<int32_t>();
    if (event.contains("index"))
        return event["index"].get<int32_t>();
    return 0;
}

bool parseScriptedProjectionActionValue(const json_t& event, nbl::this_example::CCameraScriptedActionEvent& action, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (event.contains("value") && event["value"].is_string())
    {
        const auto valueText = event["value"].get<std::string>();
        if (valueText == "perspective")
            action.value = static_cast<int32_t>(nbl::core::IPlanarProjection::CProjection::Perspective);
        else if (valueText == "orthographic")
            action.value = static_cast<int32_t>(nbl::core::IPlanarProjection::CProjection::Orthographic);
        else
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action projection type has invalid value \"" + valueText + "\".");
            return false;
        }
    }
    else
    {
        action.value = parseScriptedActionIntValue(event);
    }

    return true;
}

void parseScriptedActionEventJson(const json_t& event, const uint64_t frame, const bool captureFrame, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("action"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action event missing \"action\".");
        return;
    }

    const auto actionText = event["action"].get<std::string>();
    nbl::this_example::CCameraScriptedActionEvent entry = {
        .frame = frame
    };

    if (actionText == "set_active_render_window")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetActiveRenderWindow);
        entry.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_active_planar")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetActivePlanar);
        entry.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_projection_type")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetProjectionType);
        if (!parseScriptedProjectionActionValue(event, entry, out))
            return;
    }
    else if (actionText == "set_projection_index")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetProjectionIndex);
        entry.value = parseScriptedActionIntValue(event);
    }
    else if (actionText == "set_use_window")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetUseWindow);
        entry.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionText == "set_left_handed")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::SetLeftHanded);
        entry.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionText == "reset_active_camera")
    {
        entry.code = nbl::this_example::CCameraScriptedActionUtilities::toCode(nbl::this_example::ECameraScriptedActionCode::ResetActiveCamera);
        entry.value = 1;
    }
    else
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted action event has invalid action \"" + actionText + "\".");
        return;
    }

    out.actionEvents.emplace_back(std::move(entry));
    appendScriptedCaptureFrame(out, frame, captureFrame);
}

void parseScriptedInputEventJson(const json_t& event, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!event.contains("frame") || !event.contains("type"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted input event missing \"frame\" or \"type\".");
        return;
    }

    const auto frame = event["frame"].get<uint64_t>();
    const auto type = event["type"].get<std::string>();
    const bool captureFrame = event.value("capture", false);

    if (type == "keyboard")
        parseScriptedKeyboardEventJson(event, frame, captureFrame, out);
    else if (type == "mouse")
        parseScriptedMouseEventJson(event, frame, captureFrame, out);
    else if (type == "imguizmo")
        parseScriptedImguizmoEventJson(event, frame, captureFrame, out);
    else if (type == "action")
        parseScriptedActionEventJson(event, frame, captureFrame, out);
    else
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted input event has invalid type \"" + type + "\".");
}

void parseScriptedInputEventsJson(const json_t& script, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("events"))
        return;

    for (const auto& event : script["events"])
        parseScriptedInputEventJson(event, out);
}

bool parseScriptedImguizmoVirtualCheckJson(const json_t& check, nbl::system::CCameraScriptedInputCheck& outCheck, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    outCheck.kind = nbl::system::CCameraScriptedInputCheck::Kind::ImguizmoVirtual;
    outCheck.tolerance = check.value("tolerance", outCheck.tolerance);

    if (!check.contains("events"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check missing \"events\".");
        return false;
    }

    for (const auto& expectedEvent : check["events"])
    {
        if (!expectedEvent.contains("type") || !expectedEvent.contains("magnitude"))
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check event missing \"type\" or \"magnitude\".");
            continue;
        }

        const auto typeText = expectedEvent["type"].get<std::string>();
        const auto type = nbl::core::CVirtualGimbalEvent::stringToVirtualEvent(typeText);
        if (type == nbl::core::CVirtualGimbalEvent::None)
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Imguizmo virtual check event has invalid type \"" + typeText + "\".");
            continue;
        }

        nbl::system::CCameraScriptedInputCheck::ExpectedVirtualEvent expected;
        expected.type = type;
        expected.magnitude = expectedEvent["magnitude"].get<double>();
        outCheck.expectedVirtualEvents.emplace_back(expected);
    }

    return true;
}

bool parseScriptedCheckJson(const json_t& check, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!check.contains("frame") || !check.contains("kind"))
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted check missing \"frame\" or \"kind\".");
        return false;
    }

    const auto frame = check["frame"].get<uint64_t>();
    const auto kind = check["kind"].get<std::string>();

    nbl::system::CCameraScriptedInputCheck entry;
    entry.frame = frame;

    if (kind == "baseline")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::Baseline;
    }
    else if (kind == "imguizmo_virtual")
    {
        if (!parseScriptedImguizmoVirtualCheckJson(check, entry, out))
            return false;
    }
    else if (kind == "gimbal_near")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalNear;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);

        if (check.contains("position"))
        {
            readVector3(check["position"], entry.expectedPos);
            entry.hasExpectedPos = true;
        }
        if (check.contains("euler_deg"))
        {
            readVector3(check["euler_deg"], entry.expectedEulerDeg);
            entry.hasExpectedEuler = true;
        }
    }
    else if (kind == "gimbal_delta")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalDelta;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);
    }
    else if (kind == "gimbal_step")
    {
        entry.kind = nbl::system::CCameraScriptedInputCheck::Kind::GimbalStep;

        if (check.contains("min_pos_delta"))
        {
            entry.minPosDelta = check["min_pos_delta"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }
        if (check.contains("max_pos_delta"))
        {
            entry.posTolerance = check["max_pos_delta"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }
        else if (check.contains("pos_tolerance"))
        {
            entry.posTolerance = check["pos_tolerance"].get<float>();
            entry.hasPosDeltaConstraint = true;
        }

        if (check.contains("min_euler_delta_deg"))
        {
            entry.minEulerDeltaDeg = check["min_euler_delta_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }
        if (check.contains("max_euler_delta_deg"))
        {
            entry.eulerToleranceDeg = check["max_euler_delta_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }
        else if (check.contains("euler_tolerance_deg"))
        {
            entry.eulerToleranceDeg = check["euler_tolerance_deg"].get<float>();
            entry.hasEulerDeltaConstraint = true;
        }

        if (!entry.hasPosDeltaConstraint && !entry.hasEulerDeltaConstraint)
        {
            nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "gimbal_step check requires at least one delta constraint.");
            return false;
        }
    }
    else
    {
        nbl::this_example::CCameraScriptedRuntimePersistenceUtilities::appendScriptedInputParseWarning(out, "Scripted check has invalid kind \"" + kind + "\".");
        return false;
    }

    out.timeline.checks.emplace_back(std::move(entry));
    return true;
}

void parseScriptedChecksJson(const json_t& script, nbl::this_example::CCameraScriptedInputParseResult& out)
{
    if (!script.contains("checks"))
        return;

    for (const auto& check : script["checks"])
        parseScriptedCheckJson(check, out);
}

} // namespace impl

bool CCameraScriptedRuntimePersistenceUtilities::readCameraScriptedInput(std::string_view text, CCameraScriptedInputParseResult& out, std::string* error)
{
    json_t script;
    try
    {
        script = json_t::parse(text);
    }
    catch (const json_t::exception& e)
    {
        if (error)
            *error = e.what();
        return false;
    }

    out = {};

    if (script.contains("enabled"))
        out.enabled = script["enabled"].get<bool>();
    if (script.contains("log"))
    {
        out.hasLog = true;
        out.log = script["log"].get<bool>();
    }
    if (script.contains("hard_fail"))
        out.hardFail = script["hard_fail"].get<bool>();
    if (script.contains("visual_debug"))
        out.visualDebug = script["visual_debug"].get<bool>();
    if (script.contains("visual_debug_target_fps"))
        out.visualTargetFps = script["visual_debug_target_fps"].get<float>();
    if (script.contains("visual_debug_hold_seconds"))
        out.visualCameraHoldSeconds = script["visual_debug_hold_seconds"].get<float>();
    if (script.contains("enableActiveCameraMovement"))
    {
        out.hasEnableActiveCameraMovement = true;
        out.enableActiveCameraMovement = script["enableActiveCameraMovement"].get<bool>();
    }
    if (script.contains("exclusive_input"))
        out.exclusive = script["exclusive_input"].get<bool>() || out.exclusive;
    if (script.contains("exclusive"))
        out.exclusive = script["exclusive"].get<bool>() || out.exclusive;
    if (script.contains("capture_prefix"))
        out.capturePrefix = script["capture_prefix"].get<std::string>();
    if (out.capturePrefix.empty())
        out.capturePrefix = "script";

    impl::parseScriptedCaptureFramesJson(script, out);

    if (script.contains("camera_controls"))
        impl::parseScriptedControlOverridesJson(script["camera_controls"], out.cameraControls);

    if (!impl::parseScriptedSequenceIfPresentJson(script, out, error))
        return false;

    impl::parseScriptedInputEventsJson(script, out);
    impl::parseScriptedChecksJson(script, out);

    nbl::system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(out.timeline);
    nbl::this_example::CCameraScriptedActionUtilities::finalizeActionEvents(out.actionEvents);
    return true;
}

bool CCameraScriptedRuntimePersistenceUtilities::loadCameraScriptedInputFromFile(nbl::system::ISystem& system, const nbl::system::path& filePath, CCameraScriptedInputParseResult& out, std::string* error)
{
    std::string text;
    if (!nbl::system::CCameraFileUtilities::readTextFile(system, filePath, text, error, "Cannot open scripted input file."))
        return false;

    return readCameraScriptedInput(text, out, error);
}

} // namespace nbl::this_example
