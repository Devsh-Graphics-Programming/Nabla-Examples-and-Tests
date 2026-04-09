#ifndef _C_CAMERA_PATH_UTILITIES_HPP_
#define _C_CAMERA_PATH_UTILITIES_HPP_

#include <cmath>
#include <string_view>
#include <vector>

#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraVirtualEventUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

//! Shared helpers for the target-relative cylindrical path rig exposed as `PathRig`.
struct SCameraPathPose final : SCameraRigPose
{
    hlsl::float64_t appliedDistance = 0.0;
    hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
};

struct SCameraPathDelta final
{
    double radius = 0.0;
    double height = 0.0;
    double angle = 0.0;

    inline hlsl::float64_t3 asVector() const
    {
        return hlsl::float64_t3(radius, height, angle);
    }

    inline hlsl::float64_t3 translationVector() const
    {
        return hlsl::float64_t3(radius, height, 0.0);
    }

    static inline SCameraPathDelta fromVector(const hlsl::float64_t3& value)
    {
        return {
            .radius = value.x,
            .height = value.y,
            .angle = value.z
        };
    }
};

struct SCameraPathStateTransition final
{
    ICamera::PathState current = {};
    ICamera::PathState desired = {};
    SCameraPathDelta delta = {};
};

struct SCameraCanonicalPathState final
{
    SCameraPathPose pose = {};
    SCameraTargetRelativeState targetRelative = {};
};

struct SCameraPathComparisonThresholds final
{
    double angleToleranceDeg = ICamera::DefaultAngularToleranceDeg;
    double scalarTolerance = ICamera::ScalarTolerance;
};

struct SCameraPathDistanceUpdateResult final
{
    bool exact = false;
    hlsl::float64_t appliedDistance = 0.0;
};

struct SCameraPathDefaults final
{
    static constexpr double MinRadius = static_cast<double>(ICamera::SphericalMinDistance);
    static constexpr double ScalarTolerance = ICamera::ScalarTolerance;
    static constexpr double ExactStateTolerance = ICamera::TinyScalarEpsilon;
    static constexpr double ExactAngleToleranceDeg = ExactStateTolerance * 180.0 / hlsl::numbers::pi<double>;
    static constexpr double AngleToleranceDeg = ICamera::DefaultAngularToleranceDeg;
    static inline constexpr std::string_view Identifier = "Target-relative Cylindrical Path Rig";
    static inline constexpr std::string_view Description = "Adjust a target-relative cylindrical path rig around a target";
    struct SLimits final
    {
        double minRadius = SCameraPathDefaults::MinRadius;
        hlsl::float64_t minDistance = static_cast<hlsl::float64_t>(ICamera::SphericalMinDistance);
        hlsl::float64_t maxDistance = static_cast<hlsl::float64_t>(ICamera::SphericalMaxDistance);
    };

    static inline constexpr SLimits Limits = {};
    static inline constexpr SCameraPathComparisonThresholds ComparisonThresholds = {
        .angleToleranceDeg = AngleToleranceDeg,
        .scalarTolerance = ScalarTolerance
    };
    static inline constexpr SCameraPathComparisonThresholds ExactComparisonThresholds = {
        .angleToleranceDeg = ExactAngleToleranceDeg,
        .scalarTolerance = ExactStateTolerance
    };
};

using SCameraPathLimits = SCameraPathDefaults::SLimits;

struct CCameraPathUtilities final
{
    static inline ICamera::PathState makeDefaultPathState(const double minRadius = SCameraPathDefaults::MinRadius)
    {
        return {
            .angle = 0.0,
            .radius = minRadius,
            .height = 0.0
        };
    }

    static inline SCameraPathComparisonThresholds makePathComparisonThresholds(
        const double angleToleranceDeg = SCameraPathDefaults::AngleToleranceDeg,
        const double scalarTolerance = SCameraPathDefaults::ScalarTolerance)
    {
        return {
            .angleToleranceDeg = angleToleranceDeg,
            .scalarTolerance = scalarTolerance
        };
    }

    static inline constexpr SCameraPathLimits makeDefaultPathLimits()
    {
        return SCameraPathDefaults::Limits;
    }

    static inline bool isPathStateFinite(const ICamera::PathState& state)
    {
        return hlsl::CCameraMathUtilities::isFiniteScalar(state.angle) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.radius) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.height);
    }

    static inline bool sanitizePathState(ICamera::PathState& state, const double minRadius)
    {
        return hlsl::CCameraMathUtilities::sanitizePathState(state.angle, state.radius, state.height, minRadius);
    }

    static inline bool tryScalePathStateDistance(
        const double desiredDistance,
        const double minRadius,
        ICamera::PathState& ioState,
        double* outAppliedDistance = nullptr)
    {
        return hlsl::CCameraMathUtilities::tryScalePathStateDistance(
            desiredDistance,
            minRadius,
            ioState.radius,
            ioState.height,
            outAppliedDistance);
    }

    static inline bool tryUpdatePathStateDistance(
        const float desiredDistance,
        const SCameraPathLimits& limits,
        ICamera::PathState& ioState,
        SCameraPathDistanceUpdateResult* outResult = nullptr)
    {
        const auto clampedDistance = std::clamp<hlsl::float64_t>(desiredDistance, limits.minDistance, limits.maxDistance);
        double appliedDistance = 0.0;
        if (!tryScalePathStateDistance(static_cast<double>(clampedDistance), limits.minRadius, ioState, &appliedDistance))
            return false;

        if (outResult)
        {
            outResult->appliedDistance = appliedDistance;
            outResult->exact = (clampedDistance == desiredDistance) &&
                hlsl::CCameraMathUtilities::nearlyEqualScalar(appliedDistance, static_cast<double>(desiredDistance), SCameraPathDefaults::ScalarTolerance);
        }
        return true;
    }

    static inline bool tryBuildPathStateFromPosition(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const double minRadius,
        ICamera::PathState& outState)
    {
        return hlsl::CCameraMathUtilities::tryBuildPathStateFromPosition(
            targetPosition,
            position,
            minRadius,
            outState.angle,
            outState.radius,
            outState.height);
    }

    static inline bool tryResolvePathState(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const SCameraPathLimits& limits,
        const ICamera::PathState* requestedState,
        ICamera::PathState& outState)
    {
        if (requestedState)
        {
            outState = *requestedState;
            return sanitizePathState(outState, limits.minRadius);
        }

        if (tryBuildPathStateFromPosition(targetPosition, position, limits.minRadius, outState))
            return true;

        outState = makeDefaultPathState(limits.minRadius);
        return sanitizePathState(outState, limits.minRadius);
    }

    static inline bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraPathPose& outPose)
    {
        return hlsl::CCameraMathUtilities::tryBuildPathPoseFromState(
            targetPosition,
            state.angle,
            state.radius,
            state.height,
            limits.minRadius,
            limits.minDistance,
            limits.maxDistance,
            outPose.position,
            outPose.orientation,
            &outPose.appliedDistance,
            &outPose.orbitUv);
    }

    static inline bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        hlsl::float64_t3& outPosition,
        hlsl::camera_quaternion_t<hlsl::float64_t>& outOrientation,
        hlsl::float64_t* outAppliedDistance = nullptr,
        hlsl::float64_t2* outOrbitUv = nullptr)
    {
        SCameraPathPose pathPose = {};
        if (!tryBuildPathPoseFromState(targetPosition, state, limits, pathPose))
            return false;

        outPosition = pathPose.position;
        outOrientation = pathPose.orientation;
        if (outAppliedDistance)
            *outAppliedDistance = pathPose.appliedDistance;
        if (outOrbitUv)
            *outOrbitUv = pathPose.orbitUv;
        return true;
    }

    static inline bool pathStatesNearlyEqual(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        return hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.angle, rhs.angle) <= thresholds.angleToleranceDeg &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.radius, rhs.radius, thresholds.scalarTolerance) &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.height, rhs.height, thresholds.scalarTolerance);
    }

    static inline bool pathStatesChanged(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        return !pathStatesNearlyEqual(lhs, rhs, thresholds);
    }

    static inline hlsl::float64_t3 buildPathStateDeltaVector(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        auto deltaVector = desiredState.asVector() - currentState.asVector();
        deltaVector.z = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.z);
        return deltaVector;
    }

    static inline SCameraPathDelta buildPathStateDelta(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        return SCameraPathDelta::fromVector(buildPathStateDeltaVector(currentState, desiredState));
    }

    static inline SCameraPathDelta makePathDeltaFromVirtualPathTranslate(const hlsl::float64_t3& delta)
    {
        return SCameraPathDelta::fromVector(delta);
    }

    static inline void appendPathAdvanceEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        const double moveDenominator,
        const double angleToleranceDeg = SCameraPathDefaults::AngleToleranceDeg,
        const double scalarTolerance = SCameraPathDefaults::ScalarTolerance)
    {
        CCameraVirtualEventUtilities::appendLocalTranslationEvents(
            events,
            delta.translationVector(),
            hlsl::float64_t3(moveDenominator),
            hlsl::float64_t3(scalarTolerance));
        CCameraVirtualEventUtilities::appendAngularDeltaEvent(
            events,
            delta.angle,
            moveDenominator,
            angleToleranceDeg,
            CVirtualGimbalEvent::MoveForward,
            CVirtualGimbalEvent::MoveBackward);
    }

    static inline bool tryBuildCanonicalPathState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraCanonicalPathState& outState)
    {
        outState = {};
        if (!tryBuildPathPoseFromState(targetPosition, state, limits, outState.pose))
            return false;

        outState.targetRelative = {
            .target = targetPosition,
            .orbitUv = outState.pose.orbitUv,
            .distance = static_cast<float>(outState.pose.appliedDistance)
        };
        return true;
    }

    static inline bool tryApplyPathStateDelta(
        const ICamera::PathState& currentState,
        const SCameraPathDelta& delta,
        const SCameraPathLimits& limits,
        ICamera::PathState& outState)
    {
        auto stateVector = currentState.asVector() + delta.asVector();
        stateVector.z = hlsl::CCameraMathUtilities::wrapAngleRad(stateVector.z);
        outState = ICamera::PathState::fromVector(stateVector);
        return sanitizePathState(outState, limits.minRadius);
    }

    static inline ICamera::PathState blendPathStates(
        const ICamera::PathState& from,
        const ICamera::PathState& to,
        const double alpha)
    {
        const auto fromVector = from.asVector();
        const auto toVector = to.asVector();
        return {
            .angle = hlsl::CCameraMathUtilities::lerpWrappedAngleRad(fromVector.z, toVector.z, alpha),
            .radius = fromVector.x + (toVector.x - fromVector.x) * alpha,
            .height = fromVector.y + (toVector.y - fromVector.y) * alpha
        };
    }

    static inline bool tryBuildPathStateTransition(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& currentPosition,
        const hlsl::float64_t3& desiredPosition,
        const SCameraPathLimits& limits,
        const ICamera::PathState* currentStateOverride,
        const ICamera::PathState* desiredStateOverride,
        SCameraPathStateTransition& outTransition)
    {
        if (!tryResolvePathState(targetPosition, currentPosition, limits, currentStateOverride, outTransition.current))
            return false;
        if (!tryResolvePathState(targetPosition, desiredPosition, limits, desiredStateOverride, outTransition.desired))
            return false;

        outTransition.delta = buildPathStateDelta(outTransition.current, outTransition.desired);
        return true;
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_PATH_UTILITIES_HPP_

