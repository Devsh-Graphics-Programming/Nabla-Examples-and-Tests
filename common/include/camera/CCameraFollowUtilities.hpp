// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"

namespace nbl::hlsl
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
        const float64_t3& position = float64_t3(0.0),
        const glm::quat& orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        std::string identifier = "Follow Target")
        : m_identifier(std::move(identifier)),
        m_gimbal({ .position = position, .orientation = orientation })
    {
        m_gimbal.updateView();
    }

    inline const std::string& getIdentifier() const { return m_identifier; }
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    inline gimbal_t& getGimbal() { return m_gimbal; }

    inline void setPose(const float64_t3& position, const glm::quat& orientation)
    {
        m_gimbal.begin();
        m_gimbal.setPosition(position);
        m_gimbal.setOrientation(orientation);
        m_gimbal.end();
        m_gimbal.updateView();
    }

    inline void setPosition(const float64_t3& position)
    {
        setPose(position, m_gimbal.getOrientation());
    }

    inline void setOrientation(const glm::quat& orientation)
    {
        setPose(m_gimbal.getPosition(), orientation);
    }

    inline bool trySetFromTransform(const float64_t4x4& transform)
    {
        const auto right = normalize(float64_t3(transform[0]));
        const auto up = normalize(float64_t3(transform[1]));
        const auto forward = normalize(float64_t3(transform[2]));
        if (!isOrthoBase(right, up, forward))
            return false;

        setPose(float64_t3(transform[3]), glm::quat_cast(glm::dmat3{ right, up, forward }));
        return true;
    }

private:
    std::string m_identifier;
    gimbal_t m_gimbal;
};

enum class ECameraFollowMode : uint8_t
{
    Disabled,
    OrbitTarget,
    LookAtTarget,
    KeepWorldOffset,
    KeepLocalOffset
};

struct SCameraFollowConfig
{
    bool enabled = false;
    ECameraFollowMode mode = ECameraFollowMode::OrbitTarget;
    float64_t3 worldOffset = float64_t3(0.0);
    float64_t3 localOffset = float64_t3(0.0);
};

inline constexpr const char* getCameraFollowModeLabel(const ECameraFollowMode mode)
{
    switch (mode)
    {
        case ECameraFollowMode::Disabled: return "Disabled";
        case ECameraFollowMode::OrbitTarget: return "Orbit target";
        case ECameraFollowMode::LookAtTarget: return "Look at target";
        case ECameraFollowMode::KeepWorldOffset: return "Keep world offset";
        case ECameraFollowMode::KeepLocalOffset: return "Keep local offset";
        default: return "Unknown";
    }
}

inline constexpr bool cameraFollowModeUsesWorldOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepWorldOffset;
}

inline constexpr bool cameraFollowModeUsesLocalOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepLocalOffset;
}

inline float64_t3 transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const float64_t3& localOffset)
{
    return gimbal.getXAxis() * localOffset.x +
        gimbal.getYAxis() * localOffset.y +
        gimbal.getZAxis() * localOffset.z;
}

inline float64_t3 projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const float64_t3& worldOffset)
{
    return float64_t3(
        hlsl::dot(worldOffset, gimbal.getXAxis()),
        hlsl::dot(worldOffset, gimbal.getYAxis()),
        hlsl::dot(worldOffset, gimbal.getZAxis()));
}

inline bool buildFollowLookAtOrientation(
    const float64_t3& position,
    const float64_t3& targetPosition,
    const float64_t3& preferredUp,
    glm::quat& outOrientation)
{
    const auto toTarget = targetPosition - position;
    const double toTargetLength = length(toTarget);
    if (!std::isfinite(toTargetLength) || toTargetLength <= 1e-9)
        return false;

    const auto forward = toTarget / toTargetLength;
    auto up = preferredUp;
    if (!isFiniteVec3(up) || length(up) <= 1e-9)
        up = float64_t3(0.0, 0.0, 1.0);
    else
        up = normalize(up);

    auto right = cross(up, forward);
    if (!isFiniteVec3(right) || length(right) <= 1e-9)
    {
        const auto fallbackUp = std::abs(forward.z) < 0.99 ? float64_t3(0.0, 0.0, 1.0) : float64_t3(0.0, 1.0, 0.0);
        right = cross(fallbackUp, forward);
        if (!isFiniteVec3(right) || length(right) <= 1e-9)
            return false;
    }
    right = normalize(right);
    up = normalize(cross(forward, right));
    if (!isOrthoBase(right, up, forward))
        return false;

    outOrientation = glm::quat_cast(glm::dmat3{ right, up, forward });
    return true;
}

inline bool applyFollowSphericalPose(
    CCameraGoal& goal,
    const float64_t3& targetPosition,
    const double orbitU,
    const double orbitV,
    const float distance)
{
    if (!std::isfinite(orbitU) || !std::isfinite(orbitV) || !std::isfinite(distance))
        return false;

    const float clampedDistance = std::clamp(distance, CSphericalTargetCamera::MinDistance, CSphericalTargetCamera::MaxDistance);
    const float64_t3 spherePosition(
        std::cos(orbitV) * std::cos(orbitU) * static_cast<double>(clampedDistance),
        std::cos(orbitV) * std::sin(orbitU) * static_cast<double>(clampedDistance),
        std::sin(orbitV) * static_cast<double>(clampedDistance));

    const auto forward = normalize(-spherePosition);
    const auto up = normalize(float64_t3(
        -std::sin(orbitV) * std::cos(orbitU),
        -std::sin(orbitV) * std::sin(orbitU),
        std::cos(orbitV)));
    const auto right = normalize(cross(up, forward));
    if (!isOrthoBase(right, up, forward))
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
    goal.orientation = glm::quat_cast(glm::dmat3{ right, up, forward });
    return true;
}

inline bool buildFollowSphericalGoalFromPose(CCameraGoal& goal, const float64_t3& targetPosition, const float64_t3& position)
{
    const auto offset = position - targetPosition;
    const double distance = length(offset);
    if (!std::isfinite(distance) || distance <= 1e-9)
        return false;

    const float clampedDistance = std::clamp(static_cast<float>(distance), CSphericalTargetCamera::MinDistance, CSphericalTargetCamera::MaxDistance);
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

} // namespace nbl::hlsl

#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_
