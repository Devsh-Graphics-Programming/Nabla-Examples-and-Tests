// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

#include <string>

#include "CCameraFollowUtilities.hpp"
#include "SCameraToolingThresholds.hpp"

using namespace nbl;
using namespace nbl::ext::cameras;

struct SCameraProjectedTargetMetrics final
{
    hlsl::float32_t2 ndc = hlsl::float32_t2(0.0f);
    float radius = 0.0f;
};

/// @brief Reusable follow validation helpers.
///
/// The checks stay camera-domain:
///
/// - camera-to-target direction must match the camera forward axis for locking modes
/// - target distance must be finite and internally consistent
/// - spherical cameras must write the tracked target back into spherical target state
/// - spherical distance must match the goal-derived distance when present
struct SCameraFollowRegressionResult
{
    bool passed = false;
    bool hasLockMetrics = false;
    hlsl::float64_t lockAngleDeg = 0.0;
    hlsl::float64_t targetDistance = 0.0;
    bool hasProjectedMetrics = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
    bool hasSphericalState = false;
    hlsl::float64_t3 sphericalTarget = hlsl::float64_t3(0.0);
    hlsl::float64_t sphericalDistance = 0.0;
};

/// @brief Reusable visual/debug metrics for one active follow configuration.
struct SCameraFollowVisualMetrics
{
    bool active = false;
    ECameraFollowMode mode = ECameraFollowMode::Unknown;
    bool lockValid = false;
    hlsl::float64_t lockAngleDeg = 0.0;
    hlsl::float64_t targetDistance = 0.0;
    bool projectedValid = false;
    SCameraProjectedTargetMetrics projectedTarget = {};
};

/// @brief Shared view/projection bundle for CPU-side projected target metrics.
struct SCameraProjectionContext
{
    hlsl::float32_t4x4 viewMatrix = hlsl::float32_t4x4(1.0f);
    hlsl::float32_t4x4 projectionMatrix = hlsl::float32_t4x4(1.0f);
};

/// @brief Shared tolerances for follow target lock, writeback, and projected-center checks.
struct SCameraFollowRegressionThresholds
{
    static inline constexpr float DefaultClipWEpsilon = 1e-5f;
    static inline constexpr float DefaultProjectedNdcTolerance = 0.03f;
    static inline constexpr hlsl::float64_t DefaultLockAngleToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultDistanceTolerance = SCameraToolingThresholds::ScalarTolerance;
    static inline constexpr double DefaultTargetTolerance = SCameraToolingThresholds::TinyScalarEpsilon;
    static inline constexpr double DefaultPositionTolerance = SCameraToolingThresholds::DefaultPositionTolerance;
    static inline constexpr double DefaultRotationToleranceDeg = SCameraToolingThresholds::DefaultAngularToleranceDeg;
    static inline constexpr double DefaultScalarTolerance = SCameraToolingThresholds::ScalarTolerance;

    float clipWEpsilon = DefaultClipWEpsilon;
    float projectedNdcTolerance = DefaultProjectedNdcTolerance;
    hlsl::float64_t lockAngleToleranceDeg = DefaultLockAngleToleranceDeg;
    double distanceTolerance = DefaultDistanceTolerance;
    double targetTolerance = DefaultTargetTolerance;
    double positionTolerance = DefaultPositionTolerance;
    double rotationToleranceDeg = DefaultRotationToleranceDeg;
    double scalarTolerance = DefaultScalarTolerance;
};

/// @brief Bundled reusable follow regression flow.
/// The helper builds a follow goal, applies it, verifies the resulting camera state,
/// and then checks lock/writeback follow consistency.
struct SCameraFollowApplyValidationResult
{
    bool hasGoal = false;
    CCameraGoal goal = {};
    CCameraGoalSolver::SApplyResult applyResult = {};
    bool hasCapturedGoal = false;
    CCameraGoal capturedGoal = {};
    SCameraFollowRegressionResult regression = {};
};

struct CCameraFollowRegressionUtilities final
{
public:
    static SCameraFollowRegressionThresholds makeFollowRegressionThresholds(
        float projectedNdcTolerance = SCameraFollowRegressionThresholds::DefaultProjectedNdcTolerance,
        hlsl::float64_t lockAngleToleranceDeg = SCameraFollowRegressionThresholds::DefaultLockAngleToleranceDeg);

    static bool tryComputeProjectedFollowTargetMetrics(
        const SCameraProjectionContext& projectionContext,
        const CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        float clipWEpsilon = SCameraFollowRegressionThresholds::DefaultClipWEpsilon);

    /// @brief Check that the tracked target projects close enough to the screen centre.
    /// @param error optional (may be null); receives a description of the first failure.
    static bool validateProjectedFollowTargetContract(
        const SCameraProjectionContext& projectionContext,
        const CTrackedTarget& trackedTarget,
        SCameraProjectedTargetMetrics& outMetrics,
        std::string* error = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static SCameraFollowVisualMetrics buildFollowVisualMetrics(
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig* followConfig,
        const SCameraProjectionContext* projectionContext = nullptr);

    static bool validateFollowTargetContract(
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& followConfig,
        const CCameraGoal& followGoal,
        SCameraFollowRegressionResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});

    static bool buildApplyAndValidateFollowTargetContract(
        const CCameraGoalSolver& solver,
        ICamera* camera,
        const CTrackedTarget& trackedTarget,
        const SCameraFollowConfig& followConfig,
        SCameraFollowApplyValidationResult& out,
        std::string* error = nullptr,
        const SCameraProjectionContext* projectionContext = nullptr,
        const SCameraFollowRegressionThresholds& thresholds = {});
};


inline SCameraFollowRegressionThresholds CCameraFollowRegressionUtilities::makeFollowRegressionThresholds(
    const float projectedNdcTolerance,
    const hlsl::float64_t lockAngleToleranceDeg)
{
    auto thresholds = SCameraFollowRegressionThresholds{};
    thresholds.projectedNdcTolerance = projectedNdcTolerance;
    thresholds.lockAngleToleranceDeg = lockAngleToleranceDeg;
    return thresholds;
}

inline bool CCameraFollowRegressionUtilities::tryComputeProjectedFollowTargetMetrics(
    const SCameraProjectionContext& projectionContext,
    const CTrackedTarget& trackedTarget,
    SCameraProjectedTargetMetrics& outMetrics,
    const float clipWEpsilon)
{
    outMetrics = {};
    const hlsl::float32_t3 target = hlsl::_static_cast<hlsl::float32_t3>(trackedTarget.getGimbal().getPosition());
    const auto viewSpace = hlsl::mul(projectionContext.viewMatrix, hlsl::float32_t4(target.x, target.y, target.z, 1.0f));
    const auto clipProjection = hlsl::transpose(projectionContext.projectionMatrix);
    const auto clip = hlsl::mul(clipProjection, viewSpace);
    if (!CCameraMathUtilities::isFiniteScalar(clip.x) || !CCameraMathUtilities::isFiniteScalar(clip.y) || !CCameraMathUtilities::isFiniteScalar(clip.z) || !CCameraMathUtilities::isFiniteScalar(clip.w))
        return false;

    const auto absW = hlsl::abs(clip.w);
    if (absW < clipWEpsilon)
        return false;

    const float invW = 1.0f / clip.w;
    outMetrics.ndc = hlsl::float32_t2(clip.x, clip.y) * invW;
    if (!CCameraMathUtilities::isFiniteScalar(outMetrics.ndc.x) || !CCameraMathUtilities::isFiniteScalar(outMetrics.ndc.y))
        return false;

    outMetrics.radius = hlsl::length(outMetrics.ndc);
    return true;
}

inline bool CCameraFollowRegressionUtilities::validateProjectedFollowTargetContract(
    const SCameraProjectionContext& projectionContext,
    const CTrackedTarget& trackedTarget,
    SCameraProjectedTargetMetrics& outMetrics,
    std::string* error,
    const SCameraFollowRegressionThresholds& thresholds)
{
    if (!tryComputeProjectedFollowTargetMetrics(projectionContext, trackedTarget, outMetrics, thresholds.clipWEpsilon))
    {
        if (error)
            *error = "failed to project follow target";
        return false;
    }

    if (outMetrics.radius > thresholds.projectedNdcTolerance)
    {
        if (error)
        {
            *error = "projected target mismatch ndc=(" + std::to_string(outMetrics.ndc.x) +
                "," + std::to_string(outMetrics.ndc.y) + ") radius=" + std::to_string(outMetrics.radius);
        }
        return false;
    }

    return true;
}

inline SCameraFollowVisualMetrics CCameraFollowRegressionUtilities::buildFollowVisualMetrics(
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig* followConfig,
    const SCameraProjectionContext* projectionContext)
{
    SCameraFollowVisualMetrics out = {};
    if (!camera || !followConfig || !followConfig->enabled || followConfig->mode == ECameraFollowMode::Unknown)
        return out;

    out.active = true;
    out.mode = followConfig->mode;

    out.lockValid = CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(followConfig->mode) &&
        CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &out.targetDistance);

    if (out.lockValid && projectionContext)
        out.projectedValid = tryComputeProjectedFollowTargetMetrics(*projectionContext, trackedTarget, out.projectedTarget);

    return out;
}

inline bool CCameraFollowRegressionUtilities::validateFollowTargetContract(
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& followConfig,
    const CCameraGoal& followGoal,
    SCameraFollowRegressionResult& out,
    std::string* error,
    const SCameraProjectionContext* projectionContext,
    const SCameraFollowRegressionThresholds& thresholds)
{
    out = {};
    if (!camera)
    {
        if (error)
            *error = "missing camera";
        return false;
    }

    if (CCameraFollowUtilities::cameraFollowModeLocksViewToTarget(followConfig.mode))
    {
        out.hasLockMetrics = CCameraFollowUtilities::tryComputeFollowTargetLockMetrics(camera->getGimbal(), trackedTarget, out.lockAngleDeg, &out.targetDistance);
        if (!out.hasLockMetrics)
        {
            if (error)
                *error = "failed to compute follow lock metrics";
            return false;
        }

        const auto& trackedTargetGimbal = trackedTarget.getGimbal();
        const auto& cameraGimbal = camera->getGimbal();
        const hlsl::float64_t3 trackedTargetPosition = trackedTargetGimbal.getPosition();
        const hlsl::float64_t3 cameraPosition = cameraGimbal.getPosition();
        // TODO: this looks like it compares `out.targetDistance` against a recomputation of the same
        // camera-to-target length, which would leave only non-finite input able to fail it. Not verified. If that
        // reading is right, comparing against the distance the follow goal asked for
        // (`followGoal.orbitDistance` / `followGoal.distance`) would make it a real check.
        const double expectedTargetDistance = hlsl::length(trackedTargetPosition - cameraPosition);
        if (!CCameraMathUtilities::isFiniteScalar(expectedTargetDistance) || hlsl::abs(expectedTargetDistance - out.targetDistance) > thresholds.distanceTolerance)
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

        if (projectionContext)
        {
            if (!validateProjectedFollowTargetContract(*projectionContext, trackedTarget, out.projectedTarget, error, thresholds))
                return false;

            out.hasProjectedMetrics = true;
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
        out.sphericalDistance = static_cast<hlsl::float64_t>(state.distance);

        const auto& trackedTargetGimbal = trackedTarget.getGimbal();
        const auto& cameraGimbal = camera->getGimbal();
        const hlsl::float64_t3 trackedTargetPosition = trackedTargetGimbal.getPosition();
        const hlsl::float64_t3 targetDelta = state.target - trackedTargetPosition;
        const double targetDeltaLen = hlsl::length(targetDelta);
        if (!CCameraMathUtilities::isFiniteScalar(targetDeltaLen) || targetDeltaLen > thresholds.targetTolerance)
        {
            if (error)
                *error = "spherical target writeback mismatch";
            return false;
        }

        const double actualDistance = hlsl::length(cameraGimbal.getPosition() - trackedTargetPosition);
        const auto expectedDistance = followGoal.hasOrbitState ? static_cast<double>(followGoal.orbitDistance) :
            (followGoal.hasDistance ? static_cast<double>(followGoal.distance) : actualDistance);
        if (!CCameraMathUtilities::isFiniteScalar(actualDistance) || !CCameraMathUtilities::isFiniteScalar(expectedDistance) ||
            hlsl::abs(actualDistance - expectedDistance) > thresholds.distanceTolerance ||
            hlsl::abs(static_cast<double>(state.distance) - expectedDistance) > thresholds.distanceTolerance)
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

inline bool CCameraFollowRegressionUtilities::buildApplyAndValidateFollowTargetContract(
    const CCameraGoalSolver& solver,
    ICamera* camera,
    const CTrackedTarget& trackedTarget,
    const SCameraFollowConfig& followConfig,
    SCameraFollowApplyValidationResult& out,
    std::string* error,
    const SCameraProjectionContext* projectionContext,
    const SCameraFollowRegressionThresholds& thresholds)
{
    out = {};

    if (!CCameraFollowUtilities::tryBuildFollowGoal(solver, camera, trackedTarget, followConfig, out.goal))
    {
        if (error)
            *error = "failed to build follow goal";
        return false;
    }
    out.hasGoal = true;

    out.applyResult = CCameraFollowUtilities::applyFollowToCamera(solver, camera, trackedTarget, followConfig);
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
    if (!CCameraGoalUtilities::compareGoals(out.capturedGoal, out.goal, thresholds.positionTolerance, thresholds.rotationToleranceDeg, thresholds.scalarTolerance))
    {
        if (error)
            *error = std::string("follow goal mismatch. ") + CCameraGoalUtilities::describeGoalMismatch(out.capturedGoal, out.goal);
        return false;
    }

    return validateFollowTargetContract(
        camera,
        trackedTarget,
        followConfig,
        out.goal,
        out.regression,
        error,
        projectionContext,
        thresholds);
}


#endif // _C_CAMERA_FOLLOW_REGRESSION_UTILITIES_HPP_

