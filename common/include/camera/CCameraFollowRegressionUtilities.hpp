// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

#include <string>

#include "CCameraFollowUtilities.hpp"

namespace nbl::system
{

/**
* Reusable follow-contract validation helpers.
*
* The checks stay camera-domain:
*
* - camera-to-target direction must match the camera forward axis for locking modes
* - target distance must be finite and internally consistent
* - spherical cameras must write the tracked target back into spherical target state
* - spherical distance must match the goal-derived distance when present
*/
struct SCameraFollowRegressionResult
{
    bool passed = false;
    bool hasLockMetrics = false;
    float lockAngleDeg = 0.0f;
    double targetDistance = 0.0;
    bool hasProjectedMetrics = false;
    float projectedNdcX = 0.0f;
    float projectedNdcY = 0.0f;
    float projectedNdcRadius = 0.0f;
    bool hasSphericalState = false;
    hlsl::float64_t3 sphericalTarget = hlsl::float64_t3(0.0);
    float sphericalDistance = 0.0f;
};

//! Reusable visual/debug metrics for one active follow configuration.
struct SCameraFollowVisualMetrics
{
    bool active = false;
    core::ECameraFollowMode mode = core::ECameraFollowMode::Disabled;
    bool lockValid = false;
    float lockAngleDeg = 0.0f;
    float targetDistance = 0.0f;
    bool projectedValid = false;
    float projectedNdcX = 0.0f;
    float projectedNdcY = 0.0f;
    float projectedNdcRadius = 0.0f;
};

//! Shared tolerances for follow target lock, writeback, and projected-center checks.
struct SCameraFollowRegressionThresholds
{
    static inline constexpr float DefaultClipWEpsilon = 1e-5f;
    static inline constexpr float DefaultProjectedNdcTolerance = 0.03f;
    static inline constexpr float DefaultLockAngleToleranceDeg = static_cast<float>(core::ICamera::DefaultAngularToleranceDeg);
    static inline constexpr double DefaultDistanceTolerance = core::ICamera::ScalarTolerance;
    static inline constexpr double DefaultTargetTolerance = core::ICamera::TinyScalarEpsilon;
    static inline constexpr double DefaultPositionTolerance = core::ICamera::DefaultPositionTolerance;
    static inline constexpr double DefaultRotationToleranceDeg = core::ICamera::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultScalarTolerance = core::ICamera::ScalarTolerance;

    float clipWEpsilon = DefaultClipWEpsilon;
    float projectedNdcTolerance = DefaultProjectedNdcTolerance;
    float lockAngleToleranceDeg = DefaultLockAngleToleranceDeg;
    double distanceTolerance = DefaultDistanceTolerance;
    double targetTolerance = DefaultTargetTolerance;
    double positionTolerance = DefaultPositionTolerance;
    double rotationToleranceDeg = DefaultRotationToleranceDeg;
    double scalarTolerance = DefaultScalarTolerance;
};

//! Bundled reusable follow regression flow.
//! The helper builds a follow goal, applies it, verifies the resulting camera state,
//! and then checks the lock/writeback follow contract.
struct SCameraFollowApplyValidationResult
{
    bool hasGoal = false;
    core::CCameraGoal goal = {};
    core::CCameraGoalSolver::SApplyResult applyResult = {};
    bool hasCapturedGoal = false;
    core::CCameraGoal capturedGoal = {};
    SCameraFollowRegressionResult regression = {};
};

inline bool tryComputeProjectedFollowTargetMetrics(
    const hlsl::float32_t4x4& viewProjMatrix,
    const core::CTrackedTarget& trackedTarget,
    float& outNdcX,
    float& outNdcY,
    float* outNdcRadius = nullptr,
    const float clipWEpsilon = SCameraFollowRegressionThresholds::DefaultClipWEpsilon)
{
    const auto target = hlsl::getCastedVector<hlsl::float32_t>(trackedTarget.getGimbal().getPosition());
    const auto clip = hlsl::mul(viewProjMatrix, hlsl::float32_t4(target.x, target.y, target.z, 1.0f));
    if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) || !std::isfinite(clip.w))
        return false;

    const auto absW = std::abs(clip.w);
    if (absW < clipWEpsilon)
        return false;

    const float invW = 1.0f / clip.w;
    outNdcX = clip.x * invW;
    outNdcY = clip.y * invW;
    if (!std::isfinite(outNdcX) || !std::isfinite(outNdcY))
        return false;

    if (outNdcRadius)
        *outNdcRadius = hlsl::length(hlsl::float32_t2(outNdcX, outNdcY));

    return true;
}

inline bool validateProjectedFollowTargetContract(
    const hlsl::float32_t4x4& viewProjMatrix,
    const core::CTrackedTarget& trackedTarget,
    float& outNdcRadius,
    std::string* error = nullptr,
    const SCameraFollowRegressionThresholds& thresholds = {})
{
    float ndcX = 0.0f;
    float ndcY = 0.0f;
    if (!tryComputeProjectedFollowTargetMetrics(viewProjMatrix, trackedTarget, ndcX, ndcY, &outNdcRadius, thresholds.clipWEpsilon))
    {
        if (error)
            *error = "failed to project follow target";
        return false;
    }

    if (outNdcRadius > thresholds.projectedNdcTolerance)
    {
        if (error)
        {
            *error = "projected target mismatch ndc=(" + std::to_string(ndcX) +
                "," + std::to_string(ndcY) + ") radius=" + std::to_string(outNdcRadius);
        }
        return false;
    }

    return true;
}

inline SCameraFollowVisualMetrics buildFollowVisualMetrics(
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig* followConfig,
    const hlsl::float32_t4x4* viewProjMatrix = nullptr)
{
    SCameraFollowVisualMetrics out = {};
    if (!camera || !followConfig || !followConfig->enabled || followConfig->mode == core::ECameraFollowMode::Disabled)
        return out;

    out.active = true;
    out.mode = followConfig->mode;

    double targetDistance = 0.0;
    out.lockValid = core::cameraFollowModeLocksViewToTarget(followConfig->mode) &&
        core::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &targetDistance);
    if (out.lockValid)
        out.targetDistance = static_cast<float>(targetDistance);

    if (out.lockValid && viewProjMatrix)
    {
        out.projectedValid = tryComputeProjectedFollowTargetMetrics(
            *viewProjMatrix,
            trackedTarget,
            out.projectedNdcX,
            out.projectedNdcY,
            &out.projectedNdcRadius);
    }

    return out;
}

inline bool validateFollowTargetContract(
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig& followConfig,
    const core::CCameraGoal& followGoal,
    SCameraFollowRegressionResult& out,
    std::string* error = nullptr,
    const hlsl::float32_t4x4* viewProjMatrix = nullptr,
    const SCameraFollowRegressionThresholds& thresholds = {})
{
    out = {};
    if (!camera)
    {
        if (error)
            *error = "missing camera";
        return false;
    }

    if (core::cameraFollowModeLocksViewToTarget(followConfig.mode))
    {
        out.hasLockMetrics = core::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &out.targetDistance);
        if (!out.hasLockMetrics)
        {
            if (error)
                *error = "failed to compute follow lock metrics";
            return false;
        }

        const auto expectedTargetDistance = hlsl::length(trackedTarget.getGimbal().getPosition() - camera->getGimbal().getPosition());
        if (!std::isfinite(expectedTargetDistance) || std::abs(expectedTargetDistance - out.targetDistance) > thresholds.distanceTolerance)
        {
            if (error)
            {
                *error = "target distance mismatch actual=" + std::to_string(out.targetDistance) +
                    " expected=" + std::to_string(expectedTargetDistance);
            }
            return false;
        }

        if (out.lockAngleDeg > thresholds.lockAngleToleranceDeg)
        {
            if (error)
                *error = "lock angle mismatch angle_deg=" + std::to_string(out.lockAngleDeg);
            return false;
        }

        if (viewProjMatrix)
        {
            out.hasProjectedMetrics = tryComputeProjectedFollowTargetMetrics(
                *viewProjMatrix,
                trackedTarget,
                out.projectedNdcX,
                out.projectedNdcY,
                &out.projectedNdcRadius,
                thresholds.clipWEpsilon);
            if (!out.hasProjectedMetrics)
            {
                if (error)
                    *error = "failed to compute projected follow target metrics";
                return false;
            }

            if (out.projectedNdcRadius > thresholds.projectedNdcTolerance)
            {
                if (error)
                {
                    *error = "projected target mismatch ndc=(" + std::to_string(out.projectedNdcX) +
                        "," + std::to_string(out.projectedNdcY) + ") radius=" + std::to_string(out.projectedNdcRadius);
                }
                return false;
            }
        }
    }

    if (camera->supportsGoalState(core::ICamera::GoalStateSphericalTarget))
    {
        core::ICamera::SphericalTargetState state;
        if (!camera->tryGetSphericalTargetState(state))
        {
            if (error)
                *error = "missing spherical target state";
            return false;
        }

        out.hasSphericalState = true;
        out.sphericalTarget = state.target;
        out.sphericalDistance = state.distance;

        const auto trackedTargetPosition = trackedTarget.getGimbal().getPosition();
        const auto targetDelta = state.target - trackedTargetPosition;
        const auto targetDeltaLen = hlsl::length(targetDelta);
        if (!std::isfinite(targetDeltaLen) || targetDeltaLen > thresholds.targetTolerance)
        {
            if (error)
                *error = "spherical target writeback mismatch";
            return false;
        }

        const auto actualDistance = hlsl::length(camera->getGimbal().getPosition() - trackedTargetPosition);
        const auto expectedDistance = followGoal.hasOrbitState ? static_cast<double>(followGoal.orbitDistance) :
            (followGoal.hasDistance ? static_cast<double>(followGoal.distance) : actualDistance);
        if (!std::isfinite(actualDistance) || !std::isfinite(expectedDistance) ||
            std::abs(actualDistance - expectedDistance) > thresholds.distanceTolerance ||
            std::abs(static_cast<double>(state.distance) - expectedDistance) > thresholds.distanceTolerance)
        {
            if (error)
            {
                *error = "spherical distance mismatch actual=" + std::to_string(actualDistance) +
                    " state=" + std::to_string(state.distance) +
                    " expected=" + std::to_string(expectedDistance);
            }
            return false;
        }
    }

    out.passed = true;
    return true;
}

inline bool buildApplyAndValidateFollowTargetContract(
    const core::CCameraGoalSolver& solver,
    core::ICamera* camera,
    const core::CTrackedTarget& trackedTarget,
    const core::SCameraFollowConfig& followConfig,
    SCameraFollowApplyValidationResult& out,
    std::string* error = nullptr,
    const hlsl::float32_t4x4* viewProjMatrix = nullptr,
    const SCameraFollowRegressionThresholds& thresholds = {})
{
    out = {};

    if (!core::tryBuildFollowGoal(solver, camera, trackedTarget, followConfig, out.goal))
    {
        if (error)
            *error = "failed to build follow goal";
        return false;
    }
    out.hasGoal = true;

    out.applyResult = core::applyFollowToCamera(solver, camera, trackedTarget, followConfig);
    if (!out.applyResult.succeeded())
    {
        if (error)
            *error = "failed to apply follow goal";
        return false;
    }

    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
    {
        if (error)
            *error = "failed to capture camera state after follow apply";
        return false;
    }

    out.hasCapturedGoal = true;
    out.capturedGoal = capture.goal;
    if (!core::compareGoals(out.capturedGoal, out.goal, thresholds.positionTolerance, thresholds.rotationToleranceDeg, thresholds.scalarTolerance))
    {
        if (error)
            *error = std::string("follow goal mismatch. ") + core::describeGoalMismatch(out.capturedGoal, out.goal);
        return false;
    }

    if (!validateFollowTargetContract(
        camera,
        trackedTarget,
        followConfig,
        out.goal,
        out.regression,
        error,
        viewProjMatrix,
        thresholds))
    {
        return false;
    }

    return true;
}

} // namespace nbl::system

#endif // _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
