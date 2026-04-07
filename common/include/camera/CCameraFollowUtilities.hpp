// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"

namespace nbl::core
{

/**
* Reusable tracked-target and follow helpers layered on top of the shared camera API.
*
* The tracked subject owns its own gimbal. Follow stays outside `ICamera` and maps
* a camera plus tracked target into a `CCameraGoal`.
*/
class CTrackedTarget
{
public:
    using gimbal_t = ICamera::CGimbal;

    CTrackedTarget(
        const hlsl::float64_t3& position = hlsl::float64_t3(0.0),
        const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>(),
        std::string identifier = "Follow Target")
        : m_identifier(std::move(identifier)),
        m_gimbal({ .position = position, .orientation = orientation })
    {
        m_gimbal.updateView();
    }

    inline const std::string& getIdentifier() const { return m_identifier; }
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    inline gimbal_t& getGimbal() { return m_gimbal; }

    inline void setPose(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
    {
        m_gimbal.begin();
        m_gimbal.setPosition(position);
        m_gimbal.setOrientation(orientation);
        m_gimbal.end();
        m_gimbal.updateView();
    }

    inline void setPosition(const hlsl::float64_t3& position)
    {
        setPose(position, m_gimbal.getOrientation());
    }

    inline void setOrientation(const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation)
    {
        setPose(m_gimbal.getPosition(), orientation);
    }

    inline bool trySetFromTransform(const hlsl::float64_t4x4& transform)
    {
        const auto right = hlsl::normalize(hlsl::float64_t3(transform[0]));
        const auto up = hlsl::normalize(hlsl::float64_t3(transform[1]));
        const auto forward = hlsl::normalize(hlsl::float64_t3(transform[2]));
        if (!hlsl::isOrthoBase(right, up, forward))
            return false;

        setPose(hlsl::float64_t3(transform[3]), hlsl::makeQuaternionFromBasis(right, up, forward));
        return true;
    }

private:
    std::string m_identifier;
    gimbal_t m_gimbal;
};

/**
* Follow policy layered on top of a tracked target gimbal.
*
* The modes are intentionally explicit because `follow` is not one behavior:
*
* - `OrbitTarget` keeps a target-relative orbit/path rig and re-centers the tracked target
* - `LookAtTarget` keeps the camera world position and only rotates the view toward the target
* - `KeepWorldOffset` keeps a world-space camera offset from the target and locks the view onto it
* - `KeepLocalOffset` keeps a target-local camera offset and locks the view onto it
*
* The tracked target remains the source of truth. The camera does not own the tracked subject.
*/
enum class ECameraFollowMode : uint8_t
{
    Disabled,
    OrbitTarget,
    LookAtTarget,
    KeepWorldOffset,
    KeepLocalOffset
};

//! Reusable follow configuration interpreted against a tracked target gimbal.
//! `worldOffset` and `localOffset` are only meaningful for their matching offset-based modes.
struct SCameraFollowConfig
{
    bool enabled = false;
    ECameraFollowMode mode = ECameraFollowMode::OrbitTarget;
    hlsl::float64_t3 worldOffset = hlsl::float64_t3(0.0);
    hlsl::float64_t3 localOffset = hlsl::float64_t3(0.0);
};

inline constexpr bool cameraFollowModeLocksViewToTarget(const ECameraFollowMode mode)
{
    switch (mode)
    {
        case ECameraFollowMode::OrbitTarget:
        case ECameraFollowMode::LookAtTarget:
        case ECameraFollowMode::KeepWorldOffset:
        case ECameraFollowMode::KeepLocalOffset:
            return true;
        default:
            return false;
    }
}

inline constexpr bool cameraFollowModeMovesCameraPosition(const ECameraFollowMode mode)
{
    switch (mode)
    {
        case ECameraFollowMode::OrbitTarget:
        case ECameraFollowMode::KeepWorldOffset:
        case ECameraFollowMode::KeepLocalOffset:
            return true;
        default:
            return false;
    }
}

inline constexpr bool cameraFollowModeKeepsCameraWorldPosition(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::LookAtTarget;
}

inline constexpr bool cameraFollowModeUsesWorldOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepWorldOffset;
}

inline constexpr bool cameraFollowModeUsesLocalOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepLocalOffset;
}

inline constexpr bool cameraFollowModeUsesTrackedTargetLocalFrame(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepLocalOffset;
}

inline constexpr bool cameraFollowModeUsesCapturedOffset(const ECameraFollowMode mode)
{
    return cameraFollowModeUsesWorldOffset(mode) || cameraFollowModeUsesLocalOffset(mode);
}

inline hlsl::float64_t3 transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& localOffset)
{
    return gimbal.getXAxis() * localOffset.x +
        gimbal.getYAxis() * localOffset.y +
        gimbal.getZAxis() * localOffset.z;
}

inline hlsl::float64_t3 projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& worldOffset)
{
    return hlsl::float64_t3(
        hlsl::dot(worldOffset, gimbal.getXAxis()),
        hlsl::dot(worldOffset, gimbal.getYAxis()),
        hlsl::dot(worldOffset, gimbal.getZAxis()));
}

inline bool buildFollowLookAtOrientation(
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& preferredUp,
    hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation)
{
    const auto toTarget = targetPosition - position;
    const double toTargetLength = hlsl::length(toTarget);
    if (!std::isfinite(toTargetLength) || toTargetLength <= 1e-9)
        return false;

    const auto forward = toTarget / toTargetLength;
    auto up = preferredUp;
    if (!isFiniteVec3(up) || hlsl::length(up) <= 1e-9)
        up = hlsl::float64_t3(0.0, 0.0, 1.0);
    else
        up = hlsl::normalize(up);

    auto right = hlsl::cross(up, forward);
    if (!isFiniteVec3(right) || hlsl::length(right) <= 1e-9)
    {
        const auto fallbackUp = std::abs(forward.z) < 0.99 ? hlsl::float64_t3(0.0, 0.0, 1.0) : hlsl::float64_t3(0.0, 1.0, 0.0);
        right = hlsl::cross(fallbackUp, forward);
        if (!isFiniteVec3(right) || hlsl::length(right) <= 1e-9)
            return false;
    }
    right = hlsl::normalize(right);
    up = hlsl::normalize(hlsl::cross(forward, right));
    if (!hlsl::isOrthoBase(right, up, forward))
        return false;

    outOrientation = hlsl::makeQuaternionFromBasis(right, up, forward);
    return true;
}

inline bool applyFollowSphericalPose(
    CCameraGoal& goal,
    const hlsl::float64_t3& targetPosition,
    const double orbitU,
    const double orbitV,
    const float distance)
{
    if (!std::isfinite(orbitU) || !std::isfinite(orbitV) || !std::isfinite(distance))
        return false;

    const float clampedDistance = std::clamp(distance, ICamera::SphericalMinDistance, ICamera::SphericalMaxDistance);
    const hlsl::float64_t3 spherePosition(
        std::cos(orbitV) * std::cos(orbitU) * static_cast<double>(clampedDistance),
        std::cos(orbitV) * std::sin(orbitU) * static_cast<double>(clampedDistance),
        std::sin(orbitV) * static_cast<double>(clampedDistance));

    const auto forward = hlsl::normalize(-spherePosition);
    const auto up = hlsl::normalize(hlsl::float64_t3(
        -std::sin(orbitV) * std::cos(orbitU),
        -std::sin(orbitV) * std::sin(orbitU),
        std::cos(orbitV)));
    const auto right = hlsl::normalize(hlsl::cross(up, forward));
    if (!hlsl::isOrthoBase(right, up, forward))
        return false;

    goal.hasTargetPosition = true;
    goal.targetPosition = targetPosition;
    goal.hasDistance = true;
    goal.distance = clampedDistance;
    goal.hasOrbitState = true;
    goal.orbitU = orbitU;
    goal.orbitV = orbitV;
    goal.orbitDistance = clampedDistance;
    goal.position = targetPosition + spherePosition;
    goal.orientation = hlsl::makeQuaternionFromBasis(right, up, forward);
    return true;
}

inline bool buildFollowSphericalGoalFromPose(CCameraGoal& goal, const hlsl::float64_t3& targetPosition, const hlsl::float64_t3& position)
{
    const auto offset = position - targetPosition;
    const double distance = hlsl::length(offset);
    if (!std::isfinite(distance) || distance <= 1e-9)
        return false;

    const float clampedDistance = std::clamp(static_cast<float>(distance), ICamera::SphericalMinDistance, ICamera::SphericalMaxDistance);
    const auto local = offset / static_cast<double>(clampedDistance);
    const double orbitU = std::atan2(local.y, local.x);
    const double orbitV = std::asin(std::clamp(local.z, -1.0, 1.0));

    return applyFollowSphericalPose(goal, targetPosition, orbitU, orbitV, clampedDistance);
}

inline bool captureFollowOffsetsFromCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    SCameraFollowConfig& ioConfig)
{
    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
        return false;

    const auto& targetGimbal = trackedTarget.getGimbal();
    ioConfig.worldOffset = capture.goal.position - targetGimbal.getPosition();
    ioConfig.localOffset = projectFollowWorldOffsetToLocal(targetGimbal, ioConfig.worldOffset);
    return true;
}

inline bool tryComputeFollowTargetLockMetrics(
    const ICamera::CGimbal& cameraGimbal,
    const CTrackedTarget& trackedTarget,
    float& outAngleDeg,
    double* outDistance = nullptr)
{
    const auto toTarget = trackedTarget.getGimbal().getPosition() - cameraGimbal.getPosition();
    const auto targetDistance = hlsl::length(toTarget);
    if (!std::isfinite(targetDistance) || targetDistance <= 1e-9)
        return false;

    const auto forward = hlsl::normalize(cameraGimbal.getZAxis());
    if (!isFiniteVec3(forward) || hlsl::length(forward) <= 1e-9)
        return false;

    const auto targetDir = toTarget / targetDistance;
    const auto dotForward = std::clamp(hlsl::dot(forward, targetDir), -1.0, 1.0);
    outAngleDeg = static_cast<float>(hlsl::degrees(std::acos(dotForward)));
    if (!std::isfinite(outAngleDeg))
        return false;

    if (outDistance)
        *outDistance = targetDistance;
    return true;
}

inline bool tryBuildFollowGoal(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal& outGoal)
{
    if (!camera || !config.enabled || config.mode == ECameraFollowMode::Disabled)
        return false;

    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
        return false;

    outGoal = capture.goal;

    const auto& targetGimbal = trackedTarget.getGimbal();
    const auto targetPosition = targetGimbal.getPosition();

    switch (config.mode)
    {
        case ECameraFollowMode::OrbitTarget:
        {
            if (!camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
                return false;

            if (outGoal.hasPathState)
            {
                outGoal.hasTargetPosition = true;
                outGoal.targetPosition = targetPosition;
                outGoal = canonicalizeGoal(outGoal);
                return isGoalFinite(outGoal);
            }

            const bool hasSphericalState = outGoal.hasOrbitState || outGoal.hasDistance;
            if (!hasSphericalState)
                return false;

            const auto orbitDistance = outGoal.hasOrbitState ? outGoal.orbitDistance : outGoal.distance;
            return applyFollowSphericalPose(outGoal, targetPosition, outGoal.orbitU, outGoal.orbitV, orbitDistance);
        }

        case ECameraFollowMode::LookAtTarget:
        {
            if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
                return buildFollowSphericalGoalFromPose(outGoal, targetPosition, capture.goal.position);

            outGoal.position = capture.goal.position;
            return buildFollowLookAtOrientation(outGoal.position, targetPosition, targetGimbal.getYAxis(), outGoal.orientation) && isGoalFinite(outGoal);
        }

        case ECameraFollowMode::KeepWorldOffset:
        {
            const auto position = targetPosition + config.worldOffset;
            if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
                return buildFollowSphericalGoalFromPose(outGoal, targetPosition, position);

            outGoal.position = position;
            return buildFollowLookAtOrientation(outGoal.position, targetPosition, targetGimbal.getYAxis(), outGoal.orientation) && isGoalFinite(outGoal);
        }

        case ECameraFollowMode::KeepLocalOffset:
        {
            const auto position = targetPosition + transformFollowLocalOffset(targetGimbal, config.localOffset);
            if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
                return buildFollowSphericalGoalFromPose(outGoal, targetPosition, position);

            outGoal.position = position;
            return buildFollowLookAtOrientation(outGoal.position, targetPosition, targetGimbal.getYAxis(), outGoal.orientation) && isGoalFinite(outGoal);
        }

        default:
            return false;
    }
}

inline CCameraGoalSolver::SApplyResult applyFollowToCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal* outGoal = nullptr)
{
    CCameraGoal goal = {};
    if (!tryBuildFollowGoal(solver, camera, trackedTarget, config, goal))
        return {};

    if (outGoal)
        *outGoal = goal;

    return solver.applyDetailed(camera, goal);
}

} // namespace nbl::core

#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_
