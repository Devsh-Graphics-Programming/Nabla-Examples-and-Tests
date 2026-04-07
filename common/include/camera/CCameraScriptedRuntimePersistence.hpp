// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_
#define _C_CAMERA_SCRIPTED_RUNTIME_PERSISTENCE_HPP_

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "CCameraScriptedRuntime.hpp"
#include "CCameraSequenceScript.hpp"
#include "nbl/ui/SInputEvent.h"
#include "glm/glm/gtc/matrix_transform.hpp"
#include "glm/glm/gtc/quaternion.hpp"
#include "nlohmann/json.hpp"

namespace nbl::hlsl
{

/**
* Shared JSON parser for low-level scripted runtime payloads.
*
* This keeps legacy `events/checks/capture_frames` authored data reusable and backward-compatible
* without leaving parser glue inside one example.
*
* The parser can also detect compact `segments` and forward them into `CCameraSequenceScript`.
*/
inline float32_t4x4 composeScriptedImguizmoTransform(
    const std::array<float, 3>& translation,
    const std::array<float, 3>& rotationDeg,
    const std::array<float, 3>& scale)
{
    const auto translationMatrix = glm::translate(glm::mat4(1.f), glm::vec3(translation[0], translation[1], translation[2]));
    const auto rotationMatrix = glm::mat4_cast(glm::quat(glm::radians(glm::vec3(rotationDeg[0], rotationDeg[1], rotationDeg[2]))));
    const auto scaleMatrix = glm::scale(glm::mat4(1.f), glm::vec3(scale[0], scale[1], scale[2]));
    return float32_t4x4(translationMatrix * rotationMatrix * scaleMatrix);
}

inline float32_t4x4 makeScriptedMatrixFromArray(const std::array<float, 16>& values)
{
    float32_t4x4 out(1.f);
    for (uint32_t column = 0u; column < 4u; ++column)
    {
        for (uint32_t row = 0u; row < 4u; ++row)
            out[column][row] = values[column * 4u + row];
    }
    return out;
}

//! Optional runtime control-scale overrides parsed from low-level scripted JSON.
struct CCameraScriptedControlOverrides
{
    bool hasKeyboardScale = false;
    float keyboardScale = 1.f;
    bool hasMouseMoveScale = false;
    float mouseMoveScale = 1.f;
    bool hasMouseScrollScale = false;
    float mouseScrollScale = 1.f;
    bool hasTranslationScale = false;
    float translationScale = 1.f;
    bool hasRotationScale = false;
    float rotationScale = 1.f;
};

/**
* Parsed top-level scripted-runtime input including:
*
* - low-level expanded runtime payloads (`events`, `checks`, `capture_frames`)
* - compact camera-sequence data (`segments`) when present
* - optional runtime/debug policy flags used by scripted consumers
*/
struct CCameraScriptedInputParseResult
{
    bool enabled = true;
    bool hasLog = false;
    bool log = false;
    bool hardFail = false;
    bool visualDebug = false;
    float visualTargetFps = 0.f;
    float visualCameraHoldSeconds = 0.f;
    bool hasEnableActiveCameraMovement = false;
    bool enableActiveCameraMovement = true;
    bool exclusive = false;
    std::string capturePrefix = "script";
    CCameraScriptedControlOverrides cameraControls = {};
    CCameraScriptedTimeline timeline = {};
    std::optional<CCameraSequenceScript> sequence;
    std::vector<std::string> warnings;
};

inline void appendScriptedInputParseWarning(CCameraScriptedInputParseResult& out, std::string warning)
{
    out.warnings.emplace_back(std::move(warning));
}

inline ui::E_KEY_CODE parseScriptedKeyCode(std::string_view key)
{
    if (const auto parsed = ui::stringToKeyCode(key); parsed != ui::EKC_NONE)
        return parsed;

    constexpr std::string_view keyKeyPrefix = "KEY_KEY_";
    if (key.starts_with(keyKeyPrefix))
        return ui::stringToKeyCode(key.substr(keyKeyPrefix.size()));

    constexpr std::string_view ekcPrefix = "EKC_";
    if (key.starts_with(ekcPrefix))
        return ui::stringToKeyCode(key.substr(ekcPrefix.size()));

    return ui::EKC_NONE;
}

inline void appendScriptedInputCaptureFrame(CCameraScriptedInputParseResult& out, const uint64_t frame, const bool captureFrame)
{
    if (captureFrame)
        out.timeline.captureFrames.emplace_back(frame);
}

inline std::optional<ui::SKeyboardEvent::E_KEY_ACTION> parseScriptedKeyboardAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return ui::SKeyboardEvent::ECA_PRESSED;
    if (action == "released" || action == "release")
        return ui::SKeyboardEvent::ECA_RELEASED;
    return std::nullopt;
}

inline std::optional<ui::E_MOUSE_BUTTON> parseScriptedMouseButton(std::string_view button)
{
    if (button == "LEFT_BUTTON")
        return ui::EMB_LEFT_BUTTON;
    if (button == "RIGHT_BUTTON")
        return ui::EMB_RIGHT_BUTTON;
    if (button == "MIDDLE_BUTTON")
        return ui::EMB_MIDDLE_BUTTON;
    if (button == "BUTTON_4")
        return ui::EMB_BUTTON_4;
    if (button == "BUTTON_5")
        return ui::EMB_BUTTON_5;
    return std::nullopt;
}

inline std::optional<ui::SMouseEvent::SClickEvent::E_ACTION> parseScriptedMouseClickAction(std::string_view action)
{
    if (action == "pressed" || action == "press")
        return ui::SMouseEvent::SClickEvent::EA_PRESSED;
    if (action == "released" || action == "release")
        return ui::SMouseEvent::SClickEvent::EA_RELEASED;
    return std::nullopt;
}

inline void parseScriptedCaptureFrames(
    const nlohmann::json& script,
    CCameraScriptedInputParseResult& out)
{
    if (!script.contains("capture_frames"))
        return;

    for (const auto& frame : script["capture_frames"])
        out.timeline.captureFrames.emplace_back(frame.get<uint64_t>());
}

inline void parseScriptedControlOverrides(
    const nlohmann::json& controls,
    CCameraScriptedControlOverrides& out)
{
    if (controls.contains("keyboard_scale"))
    {
        out.hasKeyboardScale = true;
        out.keyboardScale = controls["keyboard_scale"].get<float>();
    }
    if (controls.contains("mouse_move_scale"))
    {
        out.hasMouseMoveScale = true;
        out.mouseMoveScale = controls["mouse_move_scale"].get<float>();
    }
    if (controls.contains("mouse_scroll_scale"))
    {
        out.hasMouseScrollScale = true;
        out.mouseScrollScale = controls["mouse_scroll_scale"].get<float>();
    }
    if (controls.contains("translation_scale"))
    {
        out.hasTranslationScale = true;
        out.translationScale = controls["translation_scale"].get<float>();
    }
    if (controls.contains("rotation_scale"))
    {
        out.hasRotationScale = true;
        out.rotationScale = controls["rotation_scale"].get<float>();
    }
}

inline bool parseScriptedSequenceIfPresent(
    const nlohmann::json& script,
    CCameraScriptedInputParseResult& out,
    std::string* error)
{
    if (!script.contains("segments"))
        return true;

    CCameraSequenceScript sequence;
    std::string sequenceError;
    if (!deserializeCameraSequenceScript(script, sequence, &sequenceError))
    {
        if (error)
            *error = std::move(sequenceError);
        return false;
    }

    out.sequence = std::move(sequence);
    return true;
}

inline void parseScriptedKeyboardEvent(
    const nlohmann::json& event,
    const uint64_t frame,
    const bool captureFrame,
    CCameraScriptedInputParseResult& out)
{
    if (!event.contains("key") || !event.contains("action"))
    {
        appendScriptedInputParseWarning(out, "Scripted keyboard event missing \"key\" or \"action\".");
        return;
    }

    const auto keyStr = event["key"].get<std::string>();
    const auto actionStr = event["action"].get<std::string>();
    const auto key = parseScriptedKeyCode(keyStr);
    if (key == ui::EKC_NONE)
    {
        appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid key \"" + keyStr + "\".");
        return;
    }

    const auto action = parseScriptedKeyboardAction(actionStr);
    if (!action.has_value())
    {
        appendScriptedInputParseWarning(out, "Scripted keyboard event has invalid action \"" + actionStr + "\".");
        return;
    }

    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Keyboard;
    entry.keyboard.key = key;
    entry.keyboard.action = action.value();
    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedInputCaptureFrame(out, frame, captureFrame);
}

inline void parseScriptedMouseEvent(
    const nlohmann::json& event,
    const uint64_t frame,
    const bool captureFrame,
    CCameraScriptedInputParseResult& out)
{
    if (!event.contains("kind"))
    {
        appendScriptedInputParseWarning(out, "Scripted mouse event missing \"kind\".");
        return;
    }

    const auto kind = event["kind"].get<std::string>();
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Mouse;

    if (kind == "move")
    {
        entry.mouse.type = ui::SMouseEvent::EET_MOVEMENT;
        entry.mouse.dx = event.value("dx", 0);
        entry.mouse.dy = event.value("dy", 0);
    }
    else if (kind == "scroll")
    {
        entry.mouse.type = ui::SMouseEvent::EET_SCROLL;
        entry.mouse.v = event.value("v", 0);
        entry.mouse.h = event.value("h", 0);
    }
    else if (kind == "click")
    {
        if (!event.contains("button") || !event.contains("action"))
        {
            appendScriptedInputParseWarning(out, "Scripted click event missing \"button\" or \"action\".");
            return;
        }

        const auto buttonStr = event["button"].get<std::string>();
        const auto actionStr = event["action"].get<std::string>();
        const auto button = parseScriptedMouseButton(buttonStr);
        if (!button.has_value())
        {
            appendScriptedInputParseWarning(out, "Scripted click event has invalid button \"" + buttonStr + "\".");
            return;
        }

        const auto action = parseScriptedMouseClickAction(actionStr);
        if (!action.has_value())
        {
            appendScriptedInputParseWarning(out, "Scripted click event has invalid action \"" + actionStr + "\".");
            return;
        }

        entry.mouse.type = ui::SMouseEvent::EET_CLICK;
        entry.mouse.button = button.value();
        entry.mouse.action = action.value();
        entry.mouse.x = event.value("x", 0);
        entry.mouse.y = event.value("y", 0);
    }
    else
    {
        appendScriptedInputParseWarning(out, "Scripted mouse event has invalid kind \"" + kind + "\".");
        return;
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedInputCaptureFrame(out, frame, captureFrame);
}

inline void parseScriptedImguizmoEvent(
    const nlohmann::json& event,
    const uint64_t frame,
    const bool captureFrame,
    CCameraScriptedInputParseResult& out)
{
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Imguizmo;

    if (event.contains("delta_trs"))
    {
        const auto matrix = event["delta_trs"].get<std::array<float, 16>>();
        entry.imguizmo = makeScriptedMatrixFromArray(matrix);
    }
    else
    {
        const auto translation = event.contains("translation") ? event["translation"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto rotationDeg = event.contains("rotation_deg") ? event["rotation_deg"].get<std::array<float, 3>>() : std::array<float, 3>{0.f, 0.f, 0.f};
        const auto scale = event.contains("scale") ? event["scale"].get<std::array<float, 3>>() : std::array<float, 3>{1.f, 1.f, 1.f};
        entry.imguizmo = composeScriptedImguizmoTransform(translation, rotationDeg, scale);
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedInputCaptureFrame(out, frame, captureFrame);
}

inline int32_t parseScriptedActionIntValue(const nlohmann::json& event)
{
    if (event.contains("value"))
        return event["value"].get<int32_t>();
    if (event.contains("index"))
        return event["index"].get<int32_t>();
    return 0;
}

inline bool parseScriptedProjectionActionValue(
    const nlohmann::json& event,
    CCameraScriptedInputEvent::ActionData& action,
    CCameraScriptedInputParseResult& out)
{
    if (event.contains("value") && event["value"].is_string())
    {
        const auto valueStr = event["value"].get<std::string>();
        if (valueStr == "perspective")
            action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Perspective);
        else if (valueStr == "orthographic")
            action.value = static_cast<int32_t>(IPlanarProjection::CProjection::Orthographic);
        else
        {
            appendScriptedInputParseWarning(out, "Scripted action projection type has invalid value \"" + valueStr + "\".");
            return false;
        }
    }
    else
    {
        action.value = parseScriptedActionIntValue(event);
    }

    return true;
}

inline void parseScriptedActionEvent(
    const nlohmann::json& event,
    const uint64_t frame,
    const bool captureFrame,
    CCameraScriptedInputParseResult& out)
{
    if (!event.contains("action"))
    {
        appendScriptedInputParseWarning(out, "Scripted action event missing \"action\".");
        return;
    }

    const auto actionStr = event["action"].get<std::string>();
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Action;

    if (actionStr == "set_active_render_window")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionStr == "set_active_planar")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionStr == "set_projection_type")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType;
        if (!parseScriptedProjectionActionValue(event, entry.action, out))
            return;
    }
    else if (actionStr == "set_projection_index")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetProjectionIndex;
        entry.action.value = parseScriptedActionIntValue(event);
    }
    else if (actionStr == "set_use_window")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetUseWindow;
        entry.action.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionStr == "set_left_handed")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded;
        entry.action.value = event.value("value", false) ? 1 : 0;
    }
    else if (actionStr == "reset_active_camera")
    {
        entry.action.kind = CCameraScriptedInputEvent::ActionData::Kind::ResetActiveCamera;
        entry.action.value = 1;
    }
    else
    {
        appendScriptedInputParseWarning(out, "Scripted action event has invalid action \"" + actionStr + "\".");
        return;
    }

    out.timeline.events.emplace_back(std::move(entry));
    appendScriptedInputCaptureFrame(out, frame, captureFrame);
}

inline void parseScriptedInputEvent(
    const nlohmann::json& event,
    CCameraScriptedInputParseResult& out)
{
    if (!event.contains("frame") || !event.contains("type"))
    {
        appendScriptedInputParseWarning(out, "Scripted input event missing \"frame\" or \"type\".");
        return;
    }

    const auto frame = event["frame"].get<uint64_t>();
    const auto type = event["type"].get<std::string>();
    const bool captureFrame = event.value("capture", false);

    if (type == "keyboard")
        parseScriptedKeyboardEvent(event, frame, captureFrame, out);
    else if (type == "mouse")
        parseScriptedMouseEvent(event, frame, captureFrame, out);
    else if (type == "imguizmo")
        parseScriptedImguizmoEvent(event, frame, captureFrame, out);
    else if (type == "action")
        parseScriptedActionEvent(event, frame, captureFrame, out);
    else
        appendScriptedInputParseWarning(out, "Scripted input event has invalid type \"" + type + "\".");
}

inline void parseScriptedInputEvents(
    const nlohmann::json& script,
    CCameraScriptedInputParseResult& out)
{
    if (!script.contains("events"))
        return;

    for (const auto& event : script["events"])
        parseScriptedInputEvent(event, out);
}

inline bool parseScriptedImguizmoVirtualCheck(
    const nlohmann::json& check,
    CCameraScriptedInputCheck& outCheck,
    CCameraScriptedInputParseResult& out)
{
    outCheck.kind = CCameraScriptedInputCheck::Kind::ImguizmoVirtual;
    outCheck.tolerance = check.value("tolerance", outCheck.tolerance);

    if (!check.contains("events"))
    {
        appendScriptedInputParseWarning(out, "Imguizmo virtual check missing \"events\".");
        return false;
    }

    for (const auto& expectedEvent : check["events"])
    {
        if (!expectedEvent.contains("type") || !expectedEvent.contains("magnitude"))
        {
            appendScriptedInputParseWarning(out, "Imguizmo virtual check event missing \"type\" or \"magnitude\".");
            continue;
        }

        const auto typeStr = expectedEvent["type"].get<std::string>();
        const auto type = CVirtualGimbalEvent::stringToVirtualEvent(typeStr);
        if (type == CVirtualGimbalEvent::None)
        {
            appendScriptedInputParseWarning(out, "Imguizmo virtual check event has invalid type \"" + typeStr + "\".");
            continue;
        }

        CCameraScriptedInputCheck::ExpectedVirtualEvent expected;
        expected.type = type;
        expected.magnitude = expectedEvent["magnitude"].get<double>();
        outCheck.expectedVirtualEvents.emplace_back(expected);
    }

    return true;
}

inline bool parseScriptedCheck(
    const nlohmann::json& check,
    CCameraScriptedInputParseResult& out)
{
    if (!check.contains("frame") || !check.contains("kind"))
    {
        appendScriptedInputParseWarning(out, "Scripted check missing \"frame\" or \"kind\".");
        return false;
    }

    const auto frame = check["frame"].get<uint64_t>();
    const auto kind = check["kind"].get<std::string>();

    CCameraScriptedInputCheck entry;
    entry.frame = frame;

    if (kind == "baseline")
    {
        entry.kind = CCameraScriptedInputCheck::Kind::Baseline;
    }
    else if (kind == "imguizmo_virtual")
    {
        if (!parseScriptedImguizmoVirtualCheck(check, entry, out))
            return false;
    }
    else if (kind == "gimbal_near")
    {
        entry.kind = CCameraScriptedInputCheck::Kind::GimbalNear;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);

        if (check.contains("position"))
        {
            const auto pos = check["position"].get<std::array<float, 3>>();
            entry.expectedPos = float32_t3(pos[0], pos[1], pos[2]);
            entry.hasExpectedPos = true;
        }
        if (check.contains("euler_deg"))
        {
            const auto euler = check["euler_deg"].get<std::array<float, 3>>();
            entry.expectedEulerDeg = float32_t3(euler[0], euler[1], euler[2]);
            entry.hasExpectedEuler = true;
        }
    }
    else if (kind == "gimbal_delta")
    {
        entry.kind = CCameraScriptedInputCheck::Kind::GimbalDelta;
        entry.posTolerance = check.value("pos_tolerance", entry.posTolerance);
        entry.eulerToleranceDeg = check.value("euler_tolerance_deg", entry.eulerToleranceDeg);
    }
    else if (kind == "gimbal_step")
    {
        entry.kind = CCameraScriptedInputCheck::Kind::GimbalStep;

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
            appendScriptedInputParseWarning(out, "gimbal_step check requires at least one delta constraint.");
            return false;
        }
    }
    else
    {
        appendScriptedInputParseWarning(out, "Scripted check has invalid kind \"" + kind + "\".");
        return false;
    }

    out.timeline.checks.emplace_back(std::move(entry));
    return true;
}

inline void parseScriptedChecks(
    const nlohmann::json& script,
    CCameraScriptedInputParseResult& out)
{
    if (!script.contains("checks"))
        return;

    for (const auto& check : script["checks"])
        parseScriptedCheck(check, out);
}

inline bool deserializeCameraScriptedInput(
    const nlohmann::json& script,
    CCameraScriptedInputParseResult& out,
    std::string* error = nullptr)
{
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

    parseScriptedCaptureFrames(script, out);

    if (script.contains("camera_controls"))
        parseScriptedControlOverrides(script["camera_controls"], out.cameraControls);

    if (!parseScriptedSequenceIfPresent(script, out, error))
        return false;

    parseScriptedInputEvents(script, out);
    parseScriptedChecks(script, out);

    finalizeScriptedTimeline(out.timeline);
    return true;
}

} // namespace nbl::hlsl

#endif
