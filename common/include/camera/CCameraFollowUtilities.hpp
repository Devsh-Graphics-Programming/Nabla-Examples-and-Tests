// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraKindUtilities.hpp"

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
        hlsl::float64_t3 position = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
        if (!hlsl::tryExtractRigidPoseFromTransform(transform, position, orientation))
            return false;

        setPose(position, orientation);
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

inline constexpr ECameraFollowMode getDefaultCameraFollowMode(const ICamera::CameraKind kind)
{
    switch (kind)
    {
        case ICamera::CameraKind::Orbit:
        case ICamera::CameraKind::Arcball:
        case ICamera::CameraKind::Turntable:
        case ICamera::CameraKind::TopDown:
        case ICamera::CameraKind::Isometric:
        case ICamera::CameraKind::DollyZoom:
        case ICamera::CameraKind::Path:
            return ECameraFollowMode::OrbitTarget;
        case ICamera::CameraKind::Chase:
        case ICamera::CameraKind::Dolly:
            return ECameraFollowMode::KeepLocalOffset;
        default:
            return ECameraFollowMode::Disabled;
    }
}

inline constexpr SCameraFollowConfig makeDefaultFollowConfig(const ICamera::CameraKind kind)
{
    const auto mode = getDefaultCameraFollowMode(kind);
    return {
        .enabled = mode != ECameraFollowMode::Disabled,
        .mode = mode
    };
}

inline constexpr SCameraFollowConfig makeDefaultFollowConfig(const ICamera* const camera)
{
    return camera ? makeDefaultFollowConfig(camera->getKind()) : SCameraFollowConfig{};
}

inline hlsl::float64_t3 transformFollowLocalOffset(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& localOffset)
{
    return hlsl::rotateVectorByQuaternion(gimbal.getOrientation(), localOffset);
}

inline hlsl::float64_t3 projectFollowWorldOffsetToLocal(const ICamera::CGimbal& gimbal, const hlsl::float64_t3& worldOffset)
{
    return hlsl::projectWorldVectorToLocalQuaternionFrame(gimbal.getOrientation(), worldOffset);
}

inline bool buildFollowLookAtOrientation(
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& preferredUp,
    hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation)
{
    return hlsl::tryBuildLookAtOrientation(position, targetPosition, preferredUp, outOrientation);
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
    if (!hlsl::isFiniteScalar(targetDistance) || targetDistance <= ICamera::TinyScalarEpsilon)
        return false;

    const auto forward = cameraGimbal.getZAxis();
    const auto forwardLength = hlsl::length(forward);
    if (!hlsl::isFiniteVec3(forward) || !hlsl::isFiniteScalar(forwardLength) || forwardLength <= ICamera::TinyScalarEpsilon)
        return false;

    const auto forwardDirection = forward / forwardLength;
    const auto targetDir = toTarget / targetDistance;
    const auto dotForward = std::clamp(hlsl::dot(forwardDirection, targetDir), -1.0, 1.0);
    outAngleDeg = static_cast<float>(hlsl::degrees(hlsl::acos(dotForward)));
    if (!hlsl::isFiniteScalar(outAngleDeg))
        return false;

    if (outDistance)
        *outDistance = targetDistance;
    return true;
}

inline bool tryBuildFollowPositionGoal(
    ICamera* camera,
    CCameraGoal& outGoal,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& preferredUp)
{
    if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
        return buildCanonicalTargetRelativeGoalFromPosition(outGoal, targetPosition, position);

    outGoal.position = position;
    return buildFollowLookAtOrientation(outGoal.position, targetPosition, preferredUp, outGoal.orientation) && isGoalFinite(outGoal);
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
                return applyCanonicalPathGoalFields(outGoal, targetPosition, outGoal.pathState) && isGoalFinite(outGoal);
            }

            const bool hasSphericalState = outGoal.hasOrbitState || outGoal.hasDistance;
            if (!hasSphericalState)
                return false;

            const auto orbitDistance = outGoal.hasOrbitState ? outGoal.orbitDistance : outGoal.distance;
            return applyCanonicalTargetRelativeGoal(
                outGoal,
                {
                    .target = targetPosition,
                    .orbitU = outGoal.orbitU,
                    .orbitV = outGoal.orbitV,
                    .distance = orbitDistance
                });
        }

        case ECameraFollowMode::LookAtTarget:
        {
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, capture.goal.position, targetGimbal.getYAxis());
        }

        case ECameraFollowMode::KeepWorldOffset:
        {
            const auto position = targetPosition + config.worldOffset;
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getYAxis());
        }

        case ECameraFollowMode::KeepLocalOffset:
        {
            const auto position = targetPosition + transformFollowLocalOffset(targetGimbal, config.localOffset);
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getYAxis());
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
