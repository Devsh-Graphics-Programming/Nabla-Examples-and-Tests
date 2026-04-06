// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

#include <string>

#include "CCameraFollowUtilities.hpp"

namespace nbl::hlsl
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
    float64_t3 sphericalTarget = float64_t3(0.0);
    float sphericalDistance = 0.0f;
};

inline bool tryComputeProjectedFollowTargetMetrics(
    const float32_t4x4& viewProjMatrix,
    const CTrackedTarget& trackedTarget,
    float& outNdcX,
    float& outNdcY,
    float* outNdcRadius = nullptr)
{
    const auto target = getCastedVector<float32_t>(trackedTarget.getGimbal().getPosition());
    const auto clip = mul(viewProjMatrix, float32_t4(target.x, target.y, target.z, 1.0f));
    if (!std::isfinite(clip.x) || !std::isfinite(clip.y) || !std::isfinite(clip.z) || !std::isfinite(clip.w))
        return false;

    const auto absW = std::abs(clip.w);
    if (absW < 1e-5f)
        return false;

    const float invW = 1.0f / clip.w;
    outNdcX = clip.x * invW;
    outNdcY = clip.y * invW;
    if (!std::isfinite(outNdcX) || !std::isfinite(outNdcY))
        return false;

    if (outNdcRadius)
        *outNdcRadius = std::sqrt(outNdcX * outNdcX + outNdcY * outNdcY);

    return true;
}

inline bool validateProjectedFollowTargetContract(
    const float32_t4x4& viewProjMatrix,
    const CTrackedTarget& trackedTarget,
    float& outNdcRadius,
    std::string* error = nullptr,
    const float ndcRadiusTolerance = 0.03f)
{
    float ndcX = 0.0f;
    float ndcY = 0.0f;
    if (!tryComputeProjectedFollowTargetMetrics(viewProjMatrix, trackedTarget, ndcX, ndcY, &outNdcRadius))
    {
        if (error)
            *error = "failed to project follow target";
        return false;
    }

    if (outNdcRadius > ndcRadiusTolerance)
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

inline bool validateFollowTargetContract(
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& followConfig,
    const CCameraGoal& followGoal,
    SCameraFollowRegressionResult& out,
    std::string* error = nullptr,
    const float lockAngleToleranceDeg = 0.1f,
    const double distanceTolerance = 1e-6,
    const double targetTolerance = 1e-9,
    const float32_t4x4* viewProjMatrix = nullptr,
    const float projectedNdcTolerance = 0.03f)
{
    out = {};
    if (!camera)
    {
        if (error)
            *error = "missing camera";
        return false;
    }

    if (cameraFollowModeLocksViewToTarget(followConfig.mode))
    {
        out.hasLockMetrics = tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &out.targetDistance);
        if (!out.hasLockMetrics)
        {
            if (error)
                *error = "failed to compute follow lock metrics";
            return false;
        }

        const auto expectedTargetDistance = length(trackedTarget.getGimbal().getPosition() - camera->getGimbal().getPosition());
        if (!std::isfinite(expectedTargetDistance) || std::abs(expectedTargetDistance - out.targetDistance) > distanceTolerance)
        {
            if (error)
            {
                *error = "target distance mismatch actual=" + std::to_string(out.targetDistance) +
                    " expected=" + std::to_string(expectedTargetDistance);
            }
            return false;
        }

        if (out.lockAngleDeg > lockAngleToleranceDeg)
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
                &out.projectedNdcRadius);
            if (!out.hasProjectedMetrics)
            {
                if (error)
                    *error = "failed to compute projected follow target metrics";
                return false;
            }

            if (out.projectedNdcRadius > projectedNdcTolerance)
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

    if (camera->supportsGoalState(ICamera::GoalStateSphericalTarget))
    {
        ICamera::SphericalTargetState state;
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
        const auto targetDeltaLen = length(targetDelta);
        if (!std::isfinite(targetDeltaLen) || targetDeltaLen > targetTolerance)
        {
            if (error)
                *error = "spherical target writeback mismatch";
            return false;
        }

        const auto actualDistance = length(camera->getGimbal().getPosition() - trackedTargetPosition);
        const auto expectedDistance = followGoal.hasOrbitState ? static_cast<double>(followGoal.orbitDistance) :
            (followGoal.hasDistance ? static_cast<double>(followGoal.distance) : actualDistance);
        if (!std::isfinite(actualDistance) || !std::isfinite(expectedDistance) ||
            std::abs(actualDistance - expectedDistance) > distanceTolerance ||
            std::abs(static_cast<double>(state.distance) - expectedDistance) > distanceTolerance)
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

} // namespace nbl::hlsl

#endif // _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
