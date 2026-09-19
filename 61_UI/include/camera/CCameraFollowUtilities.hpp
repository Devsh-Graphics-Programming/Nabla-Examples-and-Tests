// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_FOLLOW_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalSolver.hpp"
#include "SCameraToolingThresholds.hpp"
#include "nbl/ext/Cameras/CCameraKindUtilities.hpp"

using namespace nbl;
using namespace nbl::ext::cameras;

/// @brief Reusable tracked-target and follow helpers.
///
/// The tracked subject owns its own gimbal. Follow code reads that pose and
/// maps one camera plus one tracked target into a `CCameraGoal`.
class CTrackedTarget
{
public:
    using gimbal_t = IGimbal;

    /// @brief Construct a tracked target from an initial pose and optional identifier.
    CTrackedTarget(
        const hlsl::float64_t3& position = hlsl::float64_t3(0.0),
        const hlsl::math::quaternion<hlsl::float64_t>& orientation = hlsl::math::quaternion<hlsl::float64_t>::identity(),
        std::string identifier = "Follow Target");

    /// @brief Return the stable human-readable identifier of the tracked target.
    inline const std::string& getIdentifier() const { return m_identifier; }
    /// @brief Return read-only access to the tracked target gimbal.
    inline const gimbal_t& getGimbal() const { return m_gimbal; }
    /// @brief Return mutable access to the tracked target gimbal.
    inline gimbal_t& getGimbal() { return m_gimbal; }

    /// @brief Replace the tracked target pose in world space.
    void setPose(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation);

    /// @brief Replace only the tracked target position.
    void setPosition(const hlsl::float64_t3& position);

    /// @brief Replace only the tracked target orientation.
    void setOrientation(const hlsl::math::quaternion<hlsl::float64_t>& orientation);

    /// @brief Replace the tracked target pose from a rigid transform matrix when possible.
    bool trySetFromTransform(const hlsl::float64_t4x4& transform);

private:
    std::string m_identifier;
    gimbal_t m_gimbal;
};

/// @brief Follow policy layered on top of a tracked target gimbal.
///
/// Each mode defines how tracked-target motion updates the camera:
///
/// - `OrbitTarget` rewrites target-relative camera state so the tracked target becomes the camera target
/// - `LookAtTarget` preserves camera position and rebuilds orientation toward the tracked target
/// - `KeepWorldOffset` places the camera at `trackedTarget.position + offset` and looks at the target
/// - `KeepLocalOffset` transforms `offset` by the tracked-target local frame and looks at the target
///
/// The tracked target provides pose data. The camera reads that data and does
/// not own the tracked subject.
enum class ECameraFollowMode : uint8_t
{
    Unknown,
    OrbitTarget,
    LookAtTarget,
    KeepWorldOffset,
    KeepLocalOffset
};

/// @brief Reusable follow configuration interpreted against a tracked target gimbal.
struct SCameraFollowConfig
{
    /// @brief Whether follow should be applied at all.
    bool enabled = false;
    /// @brief Follow policy used when the configuration is enabled.
    ECameraFollowMode mode = ECameraFollowMode::OrbitTarget;
    /// @brief Camera-to-target offset in the frame the mode reads it in: world space under `KeepWorldOffset`,
    /// tracked-target local space under `KeepLocalOffset`, unused by the other modes.
    /// `captureFollowOffsetsFromCamera` writes it in whichever frame the current mode needs.
    hlsl::float64_t3 offset = hlsl::float64_t3(0.0);
};

/// @brief Shared policy helpers for tracked-target follow.
///
/// The helpers decide which follow modes lock the view, how offsets are captured,
/// and how a tracked target is translated into a `CCameraGoal` that can then be
/// applied through the shared goal solver.
struct CCameraFollowUtilities final
{
    /// @brief Return whether the follow mode rebuilds camera orientation toward the tracked target.
    static bool cameraFollowModeLocksViewToTarget(ECameraFollowMode mode);

    /// @brief Return whether the follow mode reads `SCameraFollowConfig::offset`, which has to be captured first.
    static bool cameraFollowModeUsesCapturedOffset(ECameraFollowMode mode);

    /// @brief Build the shared default follow configuration for one camera instance; a null camera gives a disabled one.
    static SCameraFollowConfig makeDefaultFollowConfig(const ICamera* camera);

    /// @brief Store the current camera-to-target offset into `ioConfig`, in the frame `ioConfig.mode` reads it in.
    static bool captureFollowOffsetsFromCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        SCameraFollowConfig& ioConfig);

    /// @brief Measure the angular lock error between a camera forward axis and a tracked target.
    /// @param outDistance optional (may be null); receives the camera-to-target distance on success.
    static bool tryComputeFollowTargetLockMetrics(
        const IGimbal& cameraGimbal,
        const CTrackedTarget& trackedTarget,
        hlsl::float64_t& outAngleDeg,
        hlsl::float64_t* outDistance = nullptr);

    static bool tryBuildFollowPositionGoal(
        ICamera* camera,
        CCameraGoal& outGoal,
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const hlsl::float64_t3& preferredUp);

    static bool tryBuildFollowGoal(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal& outGoal);

    static CCameraGoalSolver::SApplyResult applyFollowToCamera(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& config,
        CCameraGoal* outGoal = nullptr);
};


inline CTrackedTarget::CTrackedTarget(
    const hlsl::float64_t3& position,
    const hlsl::math::quaternion<hlsl::float64_t>& orientation,
    std::string identifier)
    : m_identifier(std::move(identifier)),
    m_gimbal(SCameraRigPose{ .position = position, .orientation = orientation })
{
}

inline void CTrackedTarget::setPose(const hlsl::float64_t3& position, const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    m_gimbal.setPose(SCameraRigPose{ .position = position, .orientation = orientation });
}

inline void CTrackedTarget::setPosition(const hlsl::float64_t3& position)
{
    setPose(position, m_gimbal.getOrientation());
}

inline void CTrackedTarget::setOrientation(const hlsl::math::quaternion<hlsl::float64_t>& orientation)
{
    setPose(m_gimbal.getPosition(), orientation);
}

inline bool CTrackedTarget::trySetFromTransform(const hlsl::float64_t4x4& transform)
{
    return m_gimbal.setPose(transform);
}

inline bool CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(const ECameraFollowMode mode)
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

inline bool CCameraFollowUtilities::cameraFollowModeUsesCapturedOffset(const ECameraFollowMode mode)
{
    return mode == ECameraFollowMode::KeepWorldOffset || mode == ECameraFollowMode::KeepLocalOffset;
}

inline SCameraFollowConfig CCameraFollowUtilities::makeDefaultFollowConfig(const ICamera* const camera)
{
    if (!camera)
        return {};

    auto mode = ECameraFollowMode::Unknown;
    switch (camera->getKind())
    {
        case ICamera::CameraKind::Orbit:
        case ICamera::CameraKind::Arcball:
        case ICamera::CameraKind::Turntable:
        case ICamera::CameraKind::TopDown:
        case ICamera::CameraKind::Isometric:
        case ICamera::CameraKind::DollyZoom:
        case ICamera::CameraKind::Path:
            mode = ECameraFollowMode::OrbitTarget;
            break;
        case ICamera::CameraKind::Chase:
        case ICamera::CameraKind::Dolly:
            mode = ECameraFollowMode::KeepLocalOffset;
            break;
        default:
            break;
    }

    return {
        .enabled = mode != ECameraFollowMode::Unknown,
        .mode = mode
    };
}

inline bool CCameraFollowUtilities::captureFollowOffsetsFromCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    SCameraFollowConfig& ioConfig)
{
    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
        return false;

    const auto& targetGimbal = trackedTarget.getGimbal();
    const auto worldOffset = capture.goal.position - targetGimbal.getPosition();

    // `KeepLocalOffset` replays the offset through the target's orientation, so it is stored in the target's
    // frame: rotating the world offset by the inverse of that orientation.
    ioConfig.offset = (ioConfig.mode == ECameraFollowMode::KeepLocalOffset)
        ? CCameraMathUtilities::projectWorldVectorToLocalQuaternionFrame(targetGimbal.getOrientation(), worldOffset)
        : worldOffset;
    return true;
}

inline bool CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(
    const IGimbal& cameraGimbal,
    const CTrackedTarget& trackedTarget,
    hlsl::float64_t& outAngleDeg,
    hlsl::float64_t* outDistance)
{
    const auto toTarget = trackedTarget.getGimbal().getPosition() - cameraGimbal.getPosition();
    const auto targetDistance = hlsl::length(toTarget);
    if (!CCameraMathUtilities::isFiniteScalar(targetDistance) || targetDistance <= SCameraToolingThresholds::TinyScalarEpsilon)
        return false;

    const auto forward = cameraGimbal.getForward();
    const auto forwardLength = hlsl::length(forward);
    if (!CCameraMathUtilities::isFiniteVec3(forward) || !CCameraMathUtilities::isFiniteScalar(forwardLength) || forwardLength <= SCameraToolingThresholds::TinyScalarEpsilon)
        return false;

    const auto forwardDirection = forward / forwardLength;
    const auto targetDir = toTarget / targetDistance;
    const auto dotForward = std::clamp(hlsl::dot(forwardDirection, targetDir), -1.0, 1.0);
    outAngleDeg = hlsl::degrees(hlsl::acos(dotForward));
    if (!CCameraMathUtilities::isFiniteScalar(outAngleDeg))
        return false;

    if (outDistance)
        *outDistance = targetDistance;
    return true;
}

inline bool CCameraFollowUtilities::tryBuildFollowPositionGoal(
    ICamera* camera,
    CCameraGoal& outGoal,
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const hlsl::float64_t3& preferredUp)
{
    if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
        return CCameraGoalUtilities::buildCanonicalTargetRelativeGoalFromPosition(outGoal, targetPosition, position);

    outGoal.position = position;
    return CCameraMathUtilities::tryBuildLookAtOrientation(outGoal.position, targetPosition, preferredUp, outGoal.orientation) &&
        CCameraGoalUtilities::isGoalFinite(outGoal);
}

inline bool CCameraFollowUtilities::tryBuildFollowGoal(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal& outGoal)
{
    if (!camera || !config.enabled || config.mode == ECameraFollowMode::Unknown)
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
                return CCameraGoalUtilities::applyCanonicalPathGoalFields(outGoal, targetPosition, outGoal.pathState) && CCameraGoalUtilities::isGoalFinite(outGoal);
            }

            const bool hasSphericalState = outGoal.hasOrbitState || outGoal.hasDistance;
            if (!hasSphericalState)
                return false;

            const auto orbitDistance = outGoal.hasOrbitState ? outGoal.orbitDistance : outGoal.distance;
            return CCameraGoalUtilities::applyCanonicalTargetRelativeGoal(
                outGoal,
                {
                    .target = targetPosition,
                    .angles = outGoal.orbitUv,
                    .distance = orbitDistance
                });
        }

        case ECameraFollowMode::LookAtTarget:
        {
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, capture.goal.position, targetGimbal.getUp());
        }

        case ECameraFollowMode::KeepWorldOffset:
        {
            const auto position = targetPosition + config.offset;
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, position, targetGimbal.getUp());
        }

        case ECameraFollowMode::KeepLocalOffset:
        {
            // the offset is stored in the target's frame, so it rotates with the target before it is applied
            const auto worldOffset = targetGimbal.getOrientation().transformVector(config.offset, true);
            return tryBuildFollowPositionGoal(camera, outGoal, targetPosition, targetPosition + worldOffset, targetGimbal.getUp());
        }

        default:
            return false;
    }
}

inline CCameraGoalSolver::SApplyResult CCameraFollowUtilities::applyFollowToCamera(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& config,
    CCameraGoal* outGoal)
{
    CCameraGoal goal = {};
    if (!tryBuildFollowGoal(solver, camera, trackedTarget, config, goal))
        return {};

    if (outGoal)
        *outGoal = goal;

    return solver.applyDetailed(camera, goal);
}


#endif // _C_CAMERA_FOLLOW_UTILITIES_HPP_

