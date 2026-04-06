// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPT_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPT_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "CCameraKeyframeTrack.hpp"
#include "IPlanarProjection.hpp"
#include "glm/glm/gtc/quaternion.hpp"

namespace nbl::hlsl
{

/**
* Compact authored camera-sequence format shared by playback, scripting, and validation tooling.
*
* The authored file describes:
*
* - which camera kind a segment targets
* - which reusable projection presentations should be shown
* - which keyframed camera goals should be sampled over time
* - which tracked-target poses should be sampled over time
* - which continuity thresholds and capture points should be generated
*
* The format intentionally does not store:
*
* - per-frame low-level event dumps
* - `61_UI`-specific window actions as authored source data
* - ImGuizmo transforms as the primary authored primitive
*
* A consumer such as `61_UI` may expand the compact sequence into its own runtime event/check
* representation, but the authored source of truth stays camera-domain and reusable.
*/

//! Authored projection view request for camera-sequence playback.
struct CCameraSequencePresentation
{
    IPlanarProjection::CProjection::ProjectionType projection = IPlanarProjection::CProjection::Perspective;
    bool leftHanded = true;
};

//! Shared continuity thresholds authored once and reused per sequence segment.
//! Max bounds are enforced per-step, while minimum progress can be satisfied by either position or rotation change.
struct CCameraSequenceContinuitySettings
{
    bool baseline = true;
    bool step = true;
    bool hasPosDeltaConstraint = true;
    float minPosDelta = 0.00025f;
    float maxPosDelta = 2.f;
    bool hasEulerDeltaConstraint = false;
    float minEulerDeltaDeg = 0.f;
    float maxEulerDeltaDeg = 1.f;
};

//! Relative goal adjustment authored against an initial preset captured from the target camera.
//! Deltas stay camera-domain and avoid binding the authored file to any specific input device or example.
struct CCameraSequenceGoalDelta
{
    bool hasPositionOffset = false;
    float64_t3 positionOffset = float64_t3(0.0);

    bool hasRotationEulerDegOffset = false;
    float32_t3 rotationEulerDegOffset = float32_t3(0.f);

    bool hasTargetOffset = false;
    float64_t3 targetOffset = float64_t3(0.0);

    bool hasOrbitUDeltaDeg = false;
    double orbitUDeltaDeg = 0.0;

    bool hasOrbitVDeltaDeg = false;
    double orbitVDeltaDeg = 0.0;

    bool hasOrbitDistanceDelta = false;
    float orbitDistanceDelta = 0.f;

    bool hasPathAngleDeltaDeg = false;
    double pathAngleDeltaDeg = 0.0;

    bool hasPathRadiusDelta = false;
    double pathRadiusDelta = 0.0;

    bool hasPathHeightDelta = false;
    double pathHeightDelta = 0.0;

    bool hasDynamicBaseFovDelta = false;
    float dynamicBaseFovDelta = 0.f;

    bool hasDynamicReferenceDistanceDelta = false;
    float dynamicReferenceDistanceDelta = 0.f;
};

//! One authored keyframe inside a reusable camera-sequence segment.
//! A keyframe can be described either as an absolute preset or as a delta relative to the captured reference preset.
struct CCameraSequenceKeyframe
{
    float time = 0.f;
    bool hasAbsolutePreset = false;
    CCameraPreset absolutePreset = {};
    bool hasDelta = false;
    CCameraSequenceGoalDelta delta = {};
};

//! Concrete tracked-target pose sampled from a shared authored sequence.
struct CCameraSequenceTrackedTargetPose
{
    float64_t3 position = float64_t3(0.0);
    glm::quat orientation = glm::quat(1.0, 0.0, 0.0, 0.0);
};

//! Relative tracked-target adjustment authored against an initial tracked-target pose.
struct CCameraSequenceTrackedTargetDelta
{
    bool hasPositionOffset = false;
    float64_t3 positionOffset = float64_t3(0.0);

    bool hasRotationEulerDegOffset = false;
    float32_t3 rotationEulerDegOffset = float32_t3(0.f);
};

//! One authored tracked-target keyframe inside a reusable camera-sequence segment.
//! Target keyframes stay camera-domain and can drive follow behavior without example-specific object references.
struct CCameraSequenceTrackedTargetKeyframe
{
    float time = 0.f;
    bool hasAbsolutePosition = false;
    float64_t3 absolutePosition = float64_t3(0.0);
    bool hasAbsoluteRotationEulerDeg = false;
    float32_t3 absoluteRotationEulerDeg = float32_t3(0.f);
    bool hasDelta = false;
    CCameraSequenceTrackedTargetDelta delta = {};
};

//! Runtime sampled tracked-target track built from an authored segment plus a reference pose.
struct CCameraSequenceTrackedTargetTrack
{
    struct SKeyframe
    {
        float time = 0.f;
        CCameraSequenceTrackedTargetPose pose = {};
    };

    std::vector<SKeyframe> keyframes;
};

//! Defaults shared by all camera-sequence segments unless overridden locally.
struct CCameraSequenceSegmentDefaults
{
    float durationSeconds = 4.f;
    std::vector<CCameraSequencePresentation> presentations;
    CCameraSequenceContinuitySettings continuity = {};
    std::vector<float> captureFractions = { 1.f };
    bool resetCamera = true;
};

//! Authored reusable camera-sequence segment.
//! A segment is the main unit of authored playback and validation and usually maps to one camera showcase chunk.
struct CCameraSequenceSegment
{
    std::string name;
    ICamera::CameraKind cameraKind = ICamera::CameraKind::Unknown;
    std::string cameraIdentifier;

    bool hasDurationSeconds = false;
    float durationSeconds = 0.f;

    bool hasResetCamera = false;
    bool resetCamera = true;

    std::vector<CCameraSequencePresentation> presentations;

    bool hasContinuity = false;
    CCameraSequenceContinuitySettings continuity = {};

    bool hasCaptureFractions = false;
    std::vector<float> captureFractions;

    std::vector<CCameraSequenceKeyframe> keyframes;
    std::vector<CCameraSequenceTrackedTargetKeyframe> targetKeyframes;
};

//! Top-level reusable camera-sequence script.
//! Consumers are expected to expand this compact description into their own runtime playback/check pipeline.
struct CCameraSequenceScript
{
    bool enabled = true;
    bool log = false;
    bool exclusive = false;
    bool hardFail = false;
    bool visualDebug = false;
    float visualDebugTargetFps = 0.f;
    float visualDebugHoldSeconds = 0.f;
    bool hasEnableActiveCameraMovement = false;
    bool enableActiveCameraMovement = true;
    std::string capturePrefix = "script";
    float fps = 60.f;
    CCameraSequenceSegmentDefaults defaults = {};
    std::vector<CCameraSequenceSegment> segments;
};

inline bool tryParseCameraKind(std::string_view value, ICamera::CameraKind& outKind)
{
    if (value == "FPS")
        outKind = ICamera::CameraKind::FPS;
    else if (value == "Free")
        outKind = ICamera::CameraKind::Free;
    else if (value == "Orbit")
        outKind = ICamera::CameraKind::Orbit;
    else if (value == "Arcball")
        outKind = ICamera::CameraKind::Arcball;
    else if (value == "Turntable")
        outKind = ICamera::CameraKind::Turntable;
    else if (value == "TopDown")
        outKind = ICamera::CameraKind::TopDown;
    else if (value == "Isometric")
        outKind = ICamera::CameraKind::Isometric;
    else if (value == "Chase")
        outKind = ICamera::CameraKind::Chase;
    else if (value == "Dolly")
        outKind = ICamera::CameraKind::Dolly;
    else if (value == "DollyZoom" || value == "Dolly Zoom")
        outKind = ICamera::CameraKind::DollyZoom;
    else if (value == "Path")
        outKind = ICamera::CameraKind::Path;
    else
        return false;

    return true;
}

inline bool tryParseProjectionType(std::string_view value, IPlanarProjection::CProjection::ProjectionType& outType)
{
    if (value == "perspective" || value == "Perspective")
        outType = IPlanarProjection::CProjection::Perspective;
    else if (value == "orthographic" || value == "Orthographic")
        outType = IPlanarProjection::CProjection::Orthographic;
    else
        return false;

    return true;
}

inline void normalizeCaptureFractions(std::vector<float>& fractions)
{
    for (auto& fraction : fractions)
        fraction = std::clamp(fraction, 0.f, 1.f);

    std::sort(fractions.begin(), fractions.end());
    fractions.erase(std::unique(fractions.begin(), fractions.end(),
        [](const float lhs, const float rhs) { return std::abs(lhs - rhs) <= 1e-6f; }),
        fractions.end());
}

inline bool tryParseCaptureFraction(const nlohmann::json& entry, float& outFraction)
{
    if (entry.is_number())
    {
        outFraction = std::clamp(entry.get<float>(), 0.f, 1.f);
        return true;
    }

    if (!entry.is_string())
        return false;

    const auto tag = entry.get<std::string>();
    if (tag == "start")
        outFraction = 0.f;
    else if (tag == "mid" || tag == "middle")
        outFraction = 0.5f;
    else if (tag == "end")
        outFraction = 1.f;
    else
        return false;

    return true;
}

inline bool deserializeSequencePresentations(const nlohmann::json& root, std::vector<CCameraSequencePresentation>& out, std::string* error = nullptr)
{
    out.clear();
    if (!root.is_array())
    {
        if (error)
            *error = "Sequence presentations must be an array.";
        return false;
    }

    for (const auto& entry : root)
    {
        if (!entry.is_object() || !entry.contains("projection"))
        {
            if (error)
                *error = "Sequence presentation entry missing \"projection\".";
            return false;
        }

        CCameraSequencePresentation presentation;
        if (!tryParseProjectionType(entry["projection"].get<std::string>(), presentation.projection))
        {
            if (error)
                *error = "Sequence presentation has invalid projection type.";
            return false;
        }
        if (entry.contains("left_handed"))
            presentation.leftHanded = entry["left_handed"].get<bool>();
        out.emplace_back(presentation);
    }

    return true;
}

inline bool deserializeSequenceContinuity(const nlohmann::json& root, CCameraSequenceContinuitySettings& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence continuity settings must be an object.";
        return false;
    }

    out = {};
    if (root.contains("baseline"))
        out.baseline = root["baseline"].get<bool>();
    if (root.contains("step"))
        out.step = root["step"].get<bool>();

    if (root.contains("min_pos_delta"))
    {
        out.minPosDelta = root["min_pos_delta"].get<float>();
        out.hasPosDeltaConstraint = true;
    }
    if (root.contains("max_pos_delta"))
    {
        out.maxPosDelta = root["max_pos_delta"].get<float>();
        out.hasPosDeltaConstraint = true;
    }
    else if (root.contains("pos_tolerance"))
    {
        out.maxPosDelta = root["pos_tolerance"].get<float>();
        out.hasPosDeltaConstraint = true;
    }

    if (root.contains("min_euler_delta_deg"))
    {
        out.minEulerDeltaDeg = root["min_euler_delta_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }
    if (root.contains("max_euler_delta_deg"))
    {
        out.maxEulerDeltaDeg = root["max_euler_delta_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }
    else if (root.contains("euler_tolerance_deg"))
    {
        out.maxEulerDeltaDeg = root["euler_tolerance_deg"].get<float>();
        out.hasEulerDeltaConstraint = true;
    }

    if (root.contains("disable_pos_delta"))
        out.hasPosDeltaConstraint = !root["disable_pos_delta"].get<bool>();
    if (root.contains("disable_euler_delta"))
        out.hasEulerDeltaConstraint = !root["disable_euler_delta"].get<bool>();

    if (out.step && !(out.hasPosDeltaConstraint || out.hasEulerDeltaConstraint))
    {
        if (error)
            *error = "Sequence continuity step checks require at least one delta constraint.";
        return false;
    }

    return true;
}

inline bool deserializeSequenceGoalDelta(const nlohmann::json& root, CCameraSequenceGoalDelta& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence keyframe delta must be an object.";
        return false;
    }

    out = {};
    auto readFloat3 = [](const nlohmann::json& entry, auto& outValue) -> void
    {
        const auto arr = entry.get<std::array<float, 3>>();
        outValue = std::decay_t<decltype(outValue)>(arr[0], arr[1], arr[2]);
    };
    auto readDouble3 = [](const nlohmann::json& entry, auto& outValue) -> void
    {
        const auto arr = entry.get<std::array<double, 3>>();
        outValue = std::decay_t<decltype(outValue)>(arr[0], arr[1], arr[2]);
    };

    if (root.contains("position_offset"))
    {
        readDouble3(root["position_offset"], out.positionOffset);
        out.hasPositionOffset = true;
    }
    if (root.contains("rotation_euler_deg_offset"))
    {
        readFloat3(root["rotation_euler_deg_offset"], out.rotationEulerDegOffset);
        out.hasRotationEulerDegOffset = true;
    }
    if (root.contains("target_offset"))
    {
        readDouble3(root["target_offset"], out.targetOffset);
        out.hasTargetOffset = true;
    }
    if (root.contains("orbit_u_delta_deg"))
    {
        out.orbitUDeltaDeg = root["orbit_u_delta_deg"].get<double>();
        out.hasOrbitUDeltaDeg = true;
    }
    if (root.contains("orbit_v_delta_deg"))
    {
        out.orbitVDeltaDeg = root["orbit_v_delta_deg"].get<double>();
        out.hasOrbitVDeltaDeg = true;
    }
    if (root.contains("orbit_distance_delta"))
    {
        out.orbitDistanceDelta = root["orbit_distance_delta"].get<float>();
        out.hasOrbitDistanceDelta = true;
    }
    if (root.contains("path_angle_delta_deg"))
    {
        out.pathAngleDeltaDeg = root["path_angle_delta_deg"].get<double>();
        out.hasPathAngleDeltaDeg = true;
    }
    if (root.contains("path_radius_delta"))
    {
        out.pathRadiusDelta = root["path_radius_delta"].get<double>();
        out.hasPathRadiusDelta = true;
    }
    if (root.contains("path_height_delta"))
    {
        out.pathHeightDelta = root["path_height_delta"].get<double>();
        out.hasPathHeightDelta = true;
    }
    if (root.contains("dynamic_base_fov_delta"))
    {
        out.dynamicBaseFovDelta = root["dynamic_base_fov_delta"].get<float>();
        out.hasDynamicBaseFovDelta = true;
    }
    if (root.contains("dynamic_reference_distance_delta"))
    {
        out.dynamicReferenceDistanceDelta = root["dynamic_reference_distance_delta"].get<float>();
        out.hasDynamicReferenceDistanceDelta = true;
    }

    return true;
}

inline bool deserializeSequenceKeyframe(const nlohmann::json& root, CCameraSequenceKeyframe& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence keyframe must be an object.";
        return false;
    }

    out = {};
    if (root.contains("time"))
        out.time = std::max(0.f, root["time"].get<float>());

    if (root.contains("delta"))
    {
        if (!deserializeSequenceGoalDelta(root["delta"], out.delta, error))
            return false;
        out.hasDelta = true;
    }

    if (root.contains("preset"))
    {
        deserializePreset(root["preset"], out.absolutePreset);
        out.hasAbsolutePreset = true;
    }
    else if (root.contains("position") || root.contains("orientation") || root.contains("target_position") ||
        root.contains("distance") || root.contains("orbit_u") || root.contains("orbit_v") ||
        root.contains("orbit_distance") || root.contains("path_angle") || root.contains("path_radius") ||
        root.contains("path_height") || root.contains("dynamic_base_fov") || root.contains("dynamic_reference_distance"))
    {
        deserializePreset(root, out.absolutePreset);
        out.hasAbsolutePreset = true;
    }

    return true;
}

inline bool deserializeSequenceTrackedTargetDelta(const nlohmann::json& root, CCameraSequenceTrackedTargetDelta& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence target delta must be an object.";
        return false;
    }

    out = {};
    auto readFloat3 = [](const nlohmann::json& entry, auto& outValue) -> void
    {
        const auto arr = entry.get<std::array<float, 3>>();
        outValue = std::decay_t<decltype(outValue)>(arr[0], arr[1], arr[2]);
    };
    auto readDouble3 = [](const nlohmann::json& entry, auto& outValue) -> void
    {
        const auto arr = entry.get<std::array<double, 3>>();
        outValue = std::decay_t<decltype(outValue)>(arr[0], arr[1], arr[2]);
    };

    if (root.contains("position_offset"))
    {
        readDouble3(root["position_offset"], out.positionOffset);
        out.hasPositionOffset = true;
    }
    if (root.contains("rotation_euler_deg_offset"))
    {
        readFloat3(root["rotation_euler_deg_offset"], out.rotationEulerDegOffset);
        out.hasRotationEulerDegOffset = true;
    }

    return true;
}

inline bool deserializeSequenceTrackedTargetKeyframe(const nlohmann::json& root, CCameraSequenceTrackedTargetKeyframe& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence target keyframe must be an object.";
        return false;
    }

    out = {};
    if (root.contains("time"))
        out.time = std::max(0.f, root["time"].get<float>());

    if (root.contains("delta"))
    {
        if (!deserializeSequenceTrackedTargetDelta(root["delta"], out.delta, error))
            return false;
        out.hasDelta = true;
    }

    if (root.contains("position"))
    {
        const auto arr = root["position"].get<std::array<double, 3>>();
        out.absolutePosition = float64_t3(arr[0], arr[1], arr[2]);
        out.hasAbsolutePosition = true;
    }
    if (root.contains("rotation_euler_deg"))
    {
        const auto arr = root["rotation_euler_deg"].get<std::array<float, 3>>();
        out.absoluteRotationEulerDeg = float32_t3(arr[0], arr[1], arr[2]);
        out.hasAbsoluteRotationEulerDeg = true;
    }

    return true;
}

inline bool deserializeSequenceSegment(const nlohmann::json& root, CCameraSequenceSegment& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Sequence segment must be an object.";
        return false;
    }

    out = {};
    if (root.contains("name"))
        out.name = root["name"].get<std::string>();
    if (root.contains("camera_identifier"))
        out.cameraIdentifier = root["camera_identifier"].get<std::string>();
    if (root.contains("camera_kind"))
    {
        if (!tryParseCameraKind(root["camera_kind"].get<std::string>(), out.cameraKind))
        {
            if (error)
                *error = "Sequence segment has invalid camera_kind.";
            return false;
        }
    }
    if (root.contains("duration_seconds"))
    {
        out.durationSeconds = std::max(0.f, root["duration_seconds"].get<float>());
        out.hasDurationSeconds = true;
    }
    if (root.contains("reset_camera"))
    {
        out.resetCamera = root["reset_camera"].get<bool>();
        out.hasResetCamera = true;
    }
    if (root.contains("presentations"))
    {
        if (!deserializeSequencePresentations(root["presentations"], out.presentations, error))
            return false;
    }
    if (root.contains("continuity"))
    {
        if (!deserializeSequenceContinuity(root["continuity"], out.continuity, error))
            return false;
        out.hasContinuity = true;
    }
    if (root.contains("captures"))
    {
        if (!root["captures"].is_array())
        {
            if (error)
                *error = "Sequence segment captures must be an array.";
            return false;
        }

        out.captureFractions.clear();
        for (const auto& entry : root["captures"])
        {
            float fraction = 0.f;
            if (!tryParseCaptureFraction(entry, fraction))
            {
                if (error)
                    *error = "Sequence segment capture entry is invalid.";
                return false;
            }
            out.captureFractions.emplace_back(fraction);
        }
        normalizeCaptureFractions(out.captureFractions);
        out.hasCaptureFractions = true;
    }
    if (root.contains("keyframes"))
    {
        if (!root["keyframes"].is_array())
        {
            if (error)
                *error = "Sequence segment keyframes must be an array.";
            return false;
        }
        for (const auto& entry : root["keyframes"])
        {
            CCameraSequenceKeyframe keyframe;
            if (!deserializeSequenceKeyframe(entry, keyframe, error))
                return false;
            out.keyframes.emplace_back(std::move(keyframe));
        }
    }
    if (root.contains("target_keyframes"))
    {
        if (!root["target_keyframes"].is_array())
        {
            if (error)
                *error = "Sequence segment target_keyframes must be an array.";
            return false;
        }
        for (const auto& entry : root["target_keyframes"])
        {
            CCameraSequenceTrackedTargetKeyframe keyframe;
            if (!deserializeSequenceTrackedTargetKeyframe(entry, keyframe, error))
                return false;
            out.targetKeyframes.emplace_back(std::move(keyframe));
        }
    }

    if (out.keyframes.empty())
    {
        if (error)
            *error = "Sequence segment requires at least one keyframe.";
        return false;
    }
    if (out.cameraKind == ICamera::CameraKind::Unknown && out.cameraIdentifier.empty())
    {
        if (error)
            *error = "Sequence segment requires camera_kind or camera_identifier.";
        return false;
    }

    return true;
}

inline bool deserializeCameraSequenceScript(const nlohmann::json& root, CCameraSequenceScript& out, std::string* error = nullptr)
{
    if (!root.is_object())
    {
        if (error)
            *error = "Camera sequence script must be an object.";
        return false;
    }

    out = {};
    if (root.contains("enabled"))
        out.enabled = root["enabled"].get<bool>();
    if (root.contains("log"))
        out.log = root["log"].get<bool>();
    if (root.contains("exclusive"))
        out.exclusive = root["exclusive"].get<bool>();
    if (root.contains("exclusive_input"))
        out.exclusive = root["exclusive_input"].get<bool>() || out.exclusive;
    if (root.contains("hard_fail"))
        out.hardFail = root["hard_fail"].get<bool>();
    if (root.contains("visual_debug"))
        out.visualDebug = root["visual_debug"].get<bool>();
    if (root.contains("visual_debug_target_fps"))
        out.visualDebugTargetFps = root["visual_debug_target_fps"].get<float>();
    if (root.contains("visual_debug_hold_seconds"))
        out.visualDebugHoldSeconds = root["visual_debug_hold_seconds"].get<float>();
    if (root.contains("enableActiveCameraMovement"))
    {
        out.enableActiveCameraMovement = root["enableActiveCameraMovement"].get<bool>();
        out.hasEnableActiveCameraMovement = true;
    }
    if (root.contains("capture_prefix"))
        out.capturePrefix = root["capture_prefix"].get<std::string>();
    if (root.contains("fps"))
        out.fps = std::max(1.f, root["fps"].get<float>());

    if (root.contains("defaults"))
    {
        const auto& defaults = root["defaults"];
        if (!defaults.is_object())
        {
            if (error)
                *error = "Camera sequence defaults must be an object.";
            return false;
        }

        if (defaults.contains("duration_seconds"))
            out.defaults.durationSeconds = std::max(0.f, defaults["duration_seconds"].get<float>());
        if (defaults.contains("reset_camera"))
            out.defaults.resetCamera = defaults["reset_camera"].get<bool>();
        if (defaults.contains("presentations"))
        {
            if (!deserializeSequencePresentations(defaults["presentations"], out.defaults.presentations, error))
                return false;
        }
        if (defaults.contains("continuity"))
        {
            if (!deserializeSequenceContinuity(defaults["continuity"], out.defaults.continuity, error))
                return false;
        }
        if (defaults.contains("captures"))
        {
            if (!defaults["captures"].is_array())
            {
                if (error)
                    *error = "Camera sequence default captures must be an array.";
                return false;
            }

            out.defaults.captureFractions.clear();
            for (const auto& entry : defaults["captures"])
            {
                float fraction = 0.f;
                if (!tryParseCaptureFraction(entry, fraction))
                {
                    if (error)
                        *error = "Camera sequence default capture entry is invalid.";
                    return false;
                }
                out.defaults.captureFractions.emplace_back(fraction);
            }
            normalizeCaptureFractions(out.defaults.captureFractions);
        }
    }

    if (!root.contains("segments") || !root["segments"].is_array())
    {
        if (error)
            *error = "Camera sequence script requires a \"segments\" array.";
        return false;
    }

    for (const auto& entry : root["segments"])
    {
        CCameraSequenceSegment segment;
        if (!deserializeSequenceSegment(entry, segment, error))
            return false;
        out.segments.emplace_back(std::move(segment));
    }

    if (out.segments.empty())
    {
        if (error)
            *error = "Camera sequence script must contain at least one segment.";
        return false;
    }

    return true;
}

inline bool applyCanonicalSphericalGoal(CCameraGoal& goal)
{
    if (!(goal.hasTargetPosition && goal.hasOrbitState))
        return false;
    if (!std::isfinite(goal.orbitU) || !std::isfinite(goal.orbitV) || !std::isfinite(goal.orbitDistance))
        return false;

    const float appliedDistance = std::clamp(goal.orbitDistance, CSphericalTargetCamera::MinDistance, CSphericalTargetCamera::MaxDistance);
    const float64_t3 spherePosition(
        std::cos(goal.orbitV) * std::cos(goal.orbitU) * static_cast<double>(appliedDistance),
        std::cos(goal.orbitV) * std::sin(goal.orbitU) * static_cast<double>(appliedDistance),
        std::sin(goal.orbitV) * static_cast<double>(appliedDistance));
    goal.position = goal.targetPosition + spherePosition;
    goal.hasDistance = true;
    goal.distance = appliedDistance;
    goal.orbitDistance = appliedDistance;

    const auto forward = normalize(-spherePosition);
    const float64_t3 up = normalize(float64_t3(
        -std::sin(goal.orbitV) * std::cos(goal.orbitU),
        -std::sin(goal.orbitV) * std::sin(goal.orbitU),
        std::cos(goal.orbitV)));
    const float64_t3 right = normalize(cross(up, forward));
    goal.orientation = glm::quat_cast(glm::dmat3{ right, up, forward });
    return true;
}

inline bool buildSequenceKeyframePreset(const CCameraPreset& reference, const CCameraSequenceKeyframe& authored, CCameraPreset& outPreset, std::string* error = nullptr)
{
    if (authored.hasAbsolutePreset)
    {
        outPreset = authored.absolutePreset;
        if (outPreset.identifier.empty())
            outPreset.identifier = reference.identifier;
        if (outPreset.name.empty())
            outPreset.name = reference.name;
        return isGoalFinite(makeGoalFromPreset(outPreset));
    }

    outPreset = reference;
    if (!authored.hasDelta)
        return true;

    auto goal = makeGoalFromPreset(reference);
    const auto& delta = authored.delta;

    const bool hasPoseDelta = delta.hasPositionOffset || delta.hasRotationEulerDegOffset;
    const bool hasSphericalDelta = delta.hasTargetOffset || delta.hasOrbitUDeltaDeg || delta.hasOrbitVDeltaDeg || delta.hasOrbitDistanceDelta;
    const bool hasPathDelta = delta.hasPathAngleDeltaDeg || delta.hasPathRadiusDelta || delta.hasPathHeightDelta;

    if (hasPoseDelta && (hasSphericalDelta || hasPathDelta))
    {
        if (error)
            *error = "Sequence keyframe delta cannot mix pose offsets with spherical/path deltas.";
        return false;
    }

    if (delta.hasPositionOffset)
        goal.position += delta.positionOffset;

    if (delta.hasRotationEulerDegOffset)
    {
        const auto deltaRadians = glm::radians(delta.rotationEulerDegOffset);
        goal.orientation = glm::normalize(goal.orientation * glm::quat(deltaRadians));
    }

    if (delta.hasTargetOffset)
    {
        if (!goal.hasTargetPosition)
        {
            if (error)
                *error = "Sequence keyframe target_offset requires target state.";
            return false;
        }
        goal.targetPosition += delta.targetOffset;
    }

    if (delta.hasOrbitUDeltaDeg || delta.hasOrbitVDeltaDeg || delta.hasOrbitDistanceDelta)
    {
        if (!goal.hasOrbitState)
        {
            if (error)
                *error = "Sequence keyframe orbit deltas require spherical orbit state.";
            return false;
        }
        if (delta.hasOrbitUDeltaDeg)
            goal.orbitU = wrapAngleRad(goal.orbitU + glm::radians(delta.orbitUDeltaDeg));
        if (delta.hasOrbitVDeltaDeg)
            goal.orbitV = std::clamp(goal.orbitV + glm::radians(delta.orbitVDeltaDeg), -1.55334303427, 1.55334303427);
        if (delta.hasOrbitDistanceDelta)
            goal.orbitDistance += delta.orbitDistanceDelta;
    }

    if (delta.hasPathAngleDeltaDeg || delta.hasPathRadiusDelta || delta.hasPathHeightDelta)
    {
        if (!goal.hasPathState)
        {
            if (error)
                *error = "Sequence keyframe path deltas require path state.";
            return false;
        }
        if (delta.hasPathAngleDeltaDeg)
            goal.pathState.angle = wrapAngleRad(goal.pathState.angle + glm::radians(delta.pathAngleDeltaDeg));
        if (delta.hasPathRadiusDelta)
            goal.pathState.radius += delta.pathRadiusDelta;
        if (delta.hasPathHeightDelta)
            goal.pathState.height += delta.pathHeightDelta;
    }

    if (delta.hasDynamicBaseFovDelta || delta.hasDynamicReferenceDistanceDelta)
    {
        if (!goal.hasDynamicPerspectiveState)
        {
            if (error)
                *error = "Sequence keyframe dynamic perspective deltas require dynamic perspective state.";
            return false;
        }
        if (delta.hasDynamicBaseFovDelta)
            goal.dynamicPerspectiveState.baseFov = std::clamp(goal.dynamicPerspectiveState.baseFov + delta.dynamicBaseFovDelta, 1.f, 179.f);
        if (delta.hasDynamicReferenceDistanceDelta)
            goal.dynamicPerspectiveState.referenceDistance = std::max(0.001f, goal.dynamicPerspectiveState.referenceDistance + delta.dynamicReferenceDistanceDelta);
    }

    if (hasPathDelta)
    {
        if (!applyCanonicalPathGoal(goal))
        {
            if (error)
                *error = "Sequence keyframe failed to canonicalize path state.";
            return false;
        }
    }
    else if (hasSphericalDelta)
    {
        if (!applyCanonicalSphericalGoal(goal))
        {
            if (error)
                *error = "Sequence keyframe failed to canonicalize spherical state.";
            return false;
        }
    }

    if (!isGoalFinite(goal))
    {
        if (error)
            *error = "Sequence keyframe produced a non-finite goal.";
        return false;
    }

    assignGoalToPreset(outPreset, goal);
    return true;
}

inline bool buildSequenceTrackFromReference(const CCameraPreset& reference, const CCameraSequenceSegment& segment, CCameraKeyframeTrack& outTrack, std::string* error = nullptr)
{
    outTrack = {};
    outTrack.keyframes.reserve(segment.keyframes.size());

    for (const auto& entry : segment.keyframes)
    {
        CCameraKeyframe keyframe;
        keyframe.time = std::max(0.f, entry.time);
        if (!buildSequenceKeyframePreset(reference, entry, keyframe.preset, error))
            return false;
        outTrack.keyframes.emplace_back(std::move(keyframe));
    }

    sortKeyframeTrackByTime(outTrack);
    normalizeSelectedKeyframeTrack(outTrack);
    return !outTrack.keyframes.empty();
}

inline bool isSequenceTrackedTargetPoseFinite(const CCameraSequenceTrackedTargetPose& pose)
{
    return isFiniteVec3(pose.position) &&
        std::isfinite(pose.orientation.x) &&
        std::isfinite(pose.orientation.y) &&
        std::isfinite(pose.orientation.z) &&
        std::isfinite(pose.orientation.w);
}

inline bool buildSequenceTrackedTargetPoseFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceTrackedTargetKeyframe& authored,
    CCameraSequenceTrackedTargetPose& outPose,
    std::string* error = nullptr)
{
    outPose = reference;

    if (authored.hasAbsolutePosition)
        outPose.position = authored.absolutePosition;
    if (authored.hasAbsoluteRotationEulerDeg)
        outPose.orientation = glm::quat(glm::radians(authored.absoluteRotationEulerDeg));

    if (authored.hasDelta)
    {
        if (authored.delta.hasPositionOffset)
            outPose.position += authored.delta.positionOffset;
        if (authored.delta.hasRotationEulerDegOffset)
            outPose.orientation = glm::normalize(outPose.orientation * glm::quat(glm::radians(authored.delta.rotationEulerDegOffset)));
    }

    if (!isSequenceTrackedTargetPoseFinite(outPose))
    {
        if (error)
            *error = "Sequence target keyframe produced a non-finite pose.";
        return false;
    }

    return true;
}

inline bool buildSequenceTrackedTargetTrackFromReference(
    const CCameraSequenceTrackedTargetPose& reference,
    const CCameraSequenceSegment& segment,
    CCameraSequenceTrackedTargetTrack& outTrack,
    std::string* error = nullptr)
{
    outTrack = {};
    outTrack.keyframes.reserve(segment.targetKeyframes.size());

    for (const auto& entry : segment.targetKeyframes)
    {
        CCameraSequenceTrackedTargetTrack::SKeyframe keyframe;
        keyframe.time = std::max(0.f, entry.time);
        if (!buildSequenceTrackedTargetPoseFromReference(reference, entry, keyframe.pose, error))
            return false;
        outTrack.keyframes.emplace_back(std::move(keyframe));
    }

    std::sort(outTrack.keyframes.begin(), outTrack.keyframes.end(),
        [](const auto& lhs, const auto& rhs)
        {
            if (lhs.time == rhs.time)
                return false;
            return lhs.time < rhs.time;
        });

    return !outTrack.keyframes.empty();
}

inline bool tryBuildSequenceTrackedTargetPoseAtTime(
    const CCameraSequenceTrackedTargetTrack& track,
    const float time,
    CCameraSequenceTrackedTargetPose& outPose)
{
    if (track.keyframes.empty())
        return false;
    if (track.keyframes.size() == 1u || time <= track.keyframes.front().time)
    {
        outPose = track.keyframes.front().pose;
        return true;
    }
    if (time >= track.keyframes.back().time)
    {
        outPose = track.keyframes.back().pose;
        return true;
    }

    for (size_t ix = 1u; ix < track.keyframes.size(); ++ix)
    {
        const auto& lhs = track.keyframes[ix - 1u];
        const auto& rhs = track.keyframes[ix];
        if (time > rhs.time)
            continue;

        const auto span = std::max(1e-6f, rhs.time - lhs.time);
        const auto alpha = std::clamp((time - lhs.time) / span, 0.f, 1.f);
        outPose.position = lhs.pose.position + (rhs.pose.position - lhs.pose.position) * static_cast<double>(alpha);
        outPose.orientation = glm::normalize(glm::slerp(lhs.pose.orientation, rhs.pose.orientation, alpha));
        return true;
    }

    outPose = track.keyframes.back().pose;
    return true;
}

inline bool sequenceSegmentUsesTrackedTargetTrack(const CCameraSequenceSegment& segment)
{
    return !segment.targetKeyframes.empty();
}

inline float getSequenceSegmentDurationSeconds(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment, const CCameraKeyframeTrack* track = nullptr)
{
    if (segment.hasDurationSeconds)
        return std::max(0.f, segment.durationSeconds);
    if (script.defaults.durationSeconds > 0.f)
        return script.defaults.durationSeconds;
    if (track)
        return getPlaybackTrackDuration(*track);
    return 0.f;
}

inline const std::vector<CCameraSequencePresentation>& getSequenceSegmentPresentations(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.presentations.empty() ? script.defaults.presentations : segment.presentations;
}

inline CCameraSequenceContinuitySettings getSequenceSegmentContinuity(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.hasContinuity ? segment.continuity : script.defaults.continuity;
}

inline std::vector<float> getSequenceSegmentCaptureFractions(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    auto captures = segment.hasCaptureFractions ? segment.captureFractions : script.defaults.captureFractions;
    normalizeCaptureFractions(captures);
    return captures;
}

inline bool getSequenceSegmentResetCamera(const CCameraSequenceScript& script, const CCameraSequenceSegment& segment)
{
    return segment.hasResetCamera ? segment.resetCamera : script.defaults.resetCamera;
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_SEQUENCE_SCRIPT_HPP_
