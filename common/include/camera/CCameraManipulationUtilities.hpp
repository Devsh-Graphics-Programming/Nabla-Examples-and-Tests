// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_MANIPULATION_UTILITIES_HPP_
#define _C_CAMERA_MANIPULATION_UTILITIES_HPP_

#include <algorithm>
#include <vector>

#include "CCameraPresetFlow.hpp"

namespace nbl::hlsl
{

//! Reusable constraint settings for post-manipulation camera clamping.
struct SCameraConstraintSettings
{
    bool enabled = false;
    bool clampPitch = false;
    bool clampYaw = false;
    bool clampRoll = false;
    bool clampDistance = false;
    float pitchMinDeg = -80.f;
    float pitchMaxDeg = 80.f;
    float yawMinDeg = -180.f;
    float yawMaxDeg = 180.f;
    float rollMinDeg = -180.f;
    float rollMaxDeg = 180.f;
    float minDistance = 0.1f;
    float maxDistance = 1000.f;
};

//! Scale translation and rotation event magnitudes without touching unrelated event types.
inline void scaleVirtualEvents(std::vector<CVirtualGimbalEvent>& events, const uint32_t count, const float translationScale, const float rotationScale)
{
    for (uint32_t i = 0u; i < count; ++i)
    {
        auto& ev = events[i];
        const auto type = ev.type;

        if (type == CVirtualGimbalEvent::MoveForward || type == CVirtualGimbalEvent::MoveBackward ||
            type == CVirtualGimbalEvent::MoveLeft || type == CVirtualGimbalEvent::MoveRight ||
            type == CVirtualGimbalEvent::MoveUp || type == CVirtualGimbalEvent::MoveDown)
        {
            ev.magnitude *= translationScale;
        }
        else if (type == CVirtualGimbalEvent::TiltUp || type == CVirtualGimbalEvent::TiltDown ||
            type == CVirtualGimbalEvent::PanLeft || type == CVirtualGimbalEvent::PanRight ||
            type == CVirtualGimbalEvent::RollLeft || type == CVirtualGimbalEvent::RollRight)
        {
            ev.magnitude *= rotationScale;
        }
    }
}

//! Reinterpret world-space translation intents as local camera-space movement events.
inline void remapTranslationEventsFromWorldToCameraLocal(ICamera* camera, std::vector<CVirtualGimbalEvent>& events, uint32_t& count)
{
    if (!camera)
        return;

    float64_t3 worldDelta = float64_t3(0.0);
    std::vector<CVirtualGimbalEvent> filtered;
    filtered.reserve(events.size());

    for (uint32_t i = 0u; i < count; ++i)
    {
        const auto& ev = events[i];
        switch (ev.type)
        {
            case CVirtualGimbalEvent::MoveRight: worldDelta.x += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveLeft: worldDelta.x -= ev.magnitude; break;
            case CVirtualGimbalEvent::MoveUp: worldDelta.y += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveDown: worldDelta.y -= ev.magnitude; break;
            case CVirtualGimbalEvent::MoveForward: worldDelta.z += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveBackward: worldDelta.z -= ev.magnitude; break;
            default:
                filtered.emplace_back(ev);
                break;
        }
    }

    if (worldDelta.x == 0.0 && worldDelta.y == 0.0 && worldDelta.z == 0.0)
    {
        events = std::move(filtered);
        count = static_cast<uint32_t>(events.size());
        return;
    }

    const auto& gimbal = camera->getGimbal();
    const auto right = gimbal.getXAxis();
    const auto up = gimbal.getYAxis();
    const auto forward = gimbal.getZAxis();

    const float64_t3 localDelta = float64_t3(
        hlsl::dot(worldDelta, right),
        hlsl::dot(worldDelta, up),
        hlsl::dot(worldDelta, forward)
    );

    auto emitAxis = [&](double v, CVirtualGimbalEvent::VirtualEventType pos, CVirtualGimbalEvent::VirtualEventType neg)
    {
        if (v == 0.0)
            return;
        auto& ev = filtered.emplace_back();
        ev.type = (v > 0.0) ? pos : neg;
        ev.magnitude = std::abs(v);
    };

    emitAxis(localDelta.x, CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft);
    emitAxis(localDelta.y, CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown);
    emitAxis(localDelta.z, CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward);

    events = std::move(filtered);
    count = static_cast<uint32_t>(events.size());
}

//! Apply shared distance and Euler-angle constraints after manipulation.
inline bool applyCameraConstraints(const CCameraGoalSolver& solver, ICamera* camera, const SCameraConstraintSettings& constraints)
{
    if (!constraints.enabled || !camera)
        return false;

    if (camera->hasCapability(ICamera::SphericalTarget))
    {
        if (!constraints.clampDistance)
            return false;

        ICamera::SphericalTargetState sphericalState;
        if (!camera->tryGetSphericalTargetState(sphericalState))
            return false;

        const float clamped = std::clamp<float>(sphericalState.distance, constraints.minDistance, constraints.maxDistance);
        if (clamped == sphericalState.distance)
            return false;

        return camera->trySetSphericalDistance(clamped);
    }

    if (!(constraints.clampPitch || constraints.clampYaw || constraints.clampRoll))
        return false;

    const auto& gimbal = camera->getGimbal();
    const auto pos = gimbal.getPosition();
    const auto eulerDeg = glm::degrees(glm::eulerAngles(gimbal.getOrientation()));

    auto clamped = eulerDeg;
    if (constraints.clampPitch)
        clamped.x = std::clamp(clamped.x, constraints.pitchMinDeg, constraints.pitchMaxDeg);
    if (constraints.clampYaw)
        clamped.y = std::clamp(clamped.y, constraints.yawMinDeg, constraints.yawMaxDeg);
    if (constraints.clampRoll)
        clamped.z = std::clamp(clamped.z, constraints.rollMinDeg, constraints.rollMaxDeg);

    if (clamped.x == eulerDeg.x && clamped.y == eulerDeg.y && clamped.z == eulerDeg.z)
        return false;

    CCameraPreset preset;
    preset.goal.position = pos;
    preset.goal.orientation = glm::quat(hlsl::radians(clamped));
    return applyPreset(solver, camera, preset);
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_MANIPULATION_UTILITIES_HPP_
