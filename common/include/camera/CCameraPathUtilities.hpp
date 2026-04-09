#ifndef _C_CAMERA_PATH_UTILITIES_HPP_
#define _C_CAMERA_PATH_UTILITIES_HPP_

#include <algorithm>
#include <cmath>
#include <functional>
#include <string_view>
#include <vector>

#include "CCameraPathMetadata.hpp"
#include "CCameraTargetRelativeUtilities.hpp"
#include "CCameraVirtualEventUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

//! Shared helpers for the reusable `PathRig` camera kind.
struct SCameraPathPose final : SCameraRigPose
{
    hlsl::float64_t appliedDistance = 0.0;
    hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
};

struct SCameraPathDelta final : ICamera::PathState
{
    inline hlsl::float64_t4 asVector() const
    {
        return ICamera::PathState::asVector();
    }

    inline hlsl::float64_t3 translationVector() const
    {
        return ICamera::PathState::asTranslationVector();
    }

    static inline SCameraPathDelta fromVector(const hlsl::float64_t4& value)
    {
        SCameraPathDelta delta = {};
        delta.s = value.x;
        delta.u = value.y;
        delta.v = value.z;
        delta.roll = value.w;
        return delta;
    }

    static inline SCameraPathDelta fromMotion(const hlsl::float64_t3& translation, const double pathRoll = 0.0)
    {
        SCameraPathDelta delta = {};
        delta.s = translation.z;
        delta.u = translation.x;
        delta.v = translation.y;
        delta.roll = pathRoll;
        return delta;
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
    double sToleranceDeg = ICamera::DefaultAngularToleranceDeg;
    double rollToleranceDeg = ICamera::DefaultAngularToleranceDeg;
    double scalarTolerance = ICamera::ScalarTolerance;
};

struct SCameraPathDistanceUpdateResult final
{
    bool exact = false;
    hlsl::float64_t appliedDistance = 0.0;
};

struct SCameraPathDefaults final
{
    static constexpr double MinU = static_cast<double>(ICamera::SphericalMinDistance);
    static constexpr double ScalarTolerance = ICamera::ScalarTolerance;
    static constexpr double ExactStateTolerance = ICamera::TinyScalarEpsilon;
    static constexpr double ExactAngleToleranceDeg = ExactStateTolerance * 180.0 / hlsl::numbers::pi<double>;
    static constexpr double AngleToleranceDeg = ICamera::DefaultAngularToleranceDeg;
    static inline constexpr std::string_view Identifier = SCameraPathRigMetadata::Identifier;
    static inline constexpr std::string_view Description = SCameraPathRigMetadata::DefaultModelDescription;

    struct SLimits final
    {
        double minU = SCameraPathDefaults::MinU;
        hlsl::float64_t minDistance = static_cast<hlsl::float64_t>(ICamera::SphericalMinDistance);
        hlsl::float64_t maxDistance = static_cast<hlsl::float64_t>(ICamera::SphericalMaxDistance);
    };

    static inline constexpr SLimits Limits = {};
    static inline constexpr SCameraPathComparisonThresholds ComparisonThresholds = {
        .sToleranceDeg = AngleToleranceDeg,
        .rollToleranceDeg = AngleToleranceDeg,
        .scalarTolerance = ScalarTolerance
    };
    static inline constexpr SCameraPathComparisonThresholds ExactComparisonThresholds = {
        .sToleranceDeg = ExactAngleToleranceDeg,
        .rollToleranceDeg = ExactAngleToleranceDeg,
        .scalarTolerance = ExactStateTolerance
    };
};

using SCameraPathLimits = SCameraPathDefaults::SLimits;

struct SCameraPathControlContext final
{
    ICamera::PathState currentState = {};
    hlsl::float64_t3 translation = hlsl::float64_t3(0.0);
    hlsl::float64_t3 rotation = hlsl::float64_t3(0.0);
    hlsl::float64_t3 targetPosition = hlsl::float64_t3(0.0);
    const CReferenceTransform* reference = nullptr;
    SCameraPathLimits limits = SCameraPathDefaults::Limits;
};

struct SCameraPathModel final
{
    using resolve_state_t = std::function<bool(
        const hlsl::float64_t3& targetPosition,
        const hlsl::float64_t3& position,
        const SCameraPathLimits& limits,
        const ICamera::PathState* requestedState,
        ICamera::PathState& outState)>;
    using control_law_t = std::function<SCameraPathDelta(const SCameraPathControlContext&)>;
    using integrate_t = std::function<bool(
        const ICamera::PathState& currentState,
        const SCameraPathDelta& delta,
        const SCameraPathLimits& limits,
        ICamera::PathState& outState)>;
    using evaluate_t = std::function<bool(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraCanonicalPathState& outState)>;
    using update_distance_t = std::function<bool(
        const float desiredDistance,
        const SCameraPathLimits& limits,
        ICamera::PathState& ioState,
        SCameraPathDistanceUpdateResult* outResult)>;

    resolve_state_t resolveState;
    control_law_t controlLaw;
    integrate_t integrate;
    evaluate_t evaluate;
    update_distance_t updateDistance;
};

struct CCameraPathUtilities final
{
    static inline ICamera::PathState makeDefaultPathState(const double minU = SCameraPathDefaults::MinU)
    {
        return {
            .s = 0.0,
            .u = minU,
            .v = 0.0,
            .roll = 0.0
        };
    }

    static inline SCameraPathComparisonThresholds makePathComparisonThresholds(
        const double angularToleranceDeg = SCameraPathDefaults::AngleToleranceDeg,
        const double scalarTolerance = SCameraPathDefaults::ScalarTolerance)
    {
        return {
            .sToleranceDeg = angularToleranceDeg,
            .rollToleranceDeg = angularToleranceDeg,
            .scalarTolerance = scalarTolerance
        };
    }

    static inline constexpr SCameraPathLimits makeDefaultPathLimits()
    {
        return SCameraPathDefaults::Limits;
    }

    static inline bool isPathStateFinite(const ICamera::PathState& state)
    {
        return hlsl::CCameraMathUtilities::isFiniteScalar(state.s) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.u) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.v) &&
            hlsl::CCameraMathUtilities::isFiniteScalar(state.roll);
    }

    static inline bool sanitizePathState(ICamera::PathState& state, const double minU)
    {
        return hlsl::CCameraMathUtilities::sanitizePathState(state.s, state.u, state.v, state.roll, minU);
    }

    static inline bool tryScalePathStateDistance(
        const double desiredDistance,
        const double minU,
        ICamera::PathState& ioState,
        double* outAppliedDistance = nullptr)
    {
        return hlsl::CCameraMathUtilities::tryScalePathStateDistance(
            desiredDistance,
            minU,
            ioState.u,
            ioState.v,
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
        if (!tryScalePathStateDistance(static_cast<double>(clampedDistance), limits.minU, ioState, &appliedDistance))
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
        const double minU,
        ICamera::PathState& outState)
    {
        outState = {};
        if (!hlsl::CCameraMathUtilities::tryBuildPathStateFromPosition(
                targetPosition,
                position,
                minU,
                outState.s,
                outState.u,
                outState.v))
        {
            return false;
        }

        outState.roll = 0.0;
        return true;
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
            return sanitizePathState(outState, limits.minU);
        }

        if (tryBuildPathStateFromPosition(targetPosition, position, limits.minU, outState))
            return true;

        outState = makeDefaultPathState(limits.minU);
        return sanitizePathState(outState, limits.minU);
    }

    static inline bool tryBuildPathPoseFromState(
        const hlsl::float64_t3& targetPosition,
        const ICamera::PathState& state,
        const SCameraPathLimits& limits,
        SCameraPathPose& outPose)
    {
        return hlsl::CCameraMathUtilities::tryBuildPathPoseFromState(
            targetPosition,
            state.s,
            state.u,
            state.v,
            state.roll,
            limits.minU,
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
        return hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.s, rhs.s) <= thresholds.sToleranceDeg &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.u, rhs.u, thresholds.scalarTolerance) &&
            hlsl::CCameraMathUtilities::nearlyEqualScalar(lhs.v, rhs.v, thresholds.scalarTolerance) &&
            hlsl::CCameraMathUtilities::getWrappedAngleDistanceDegrees(lhs.roll, rhs.roll) <= thresholds.rollToleranceDeg;
    }

    static inline bool pathStatesChanged(
        const ICamera::PathState& lhs,
        const ICamera::PathState& rhs,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        return !pathStatesNearlyEqual(lhs, rhs, thresholds);
    }

    static inline hlsl::float64_t4 buildPathStateDeltaVector(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        auto deltaVector = desiredState.asVector() - currentState.asVector();
        deltaVector.x = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.x);
        deltaVector.w = hlsl::CCameraMathUtilities::wrapAngleRad(deltaVector.w);
        return deltaVector;
    }

    static inline SCameraPathDelta buildPathStateDelta(
        const ICamera::PathState& currentState,
        const ICamera::PathState& desiredState)
    {
        return SCameraPathDelta::fromVector(buildPathStateDeltaVector(currentState, desiredState));
    }

    static inline SCameraPathDelta makePathDeltaFromVirtualPathMotion(
        const hlsl::float64_t3& translation,
        const hlsl::float64_t3& rotation = hlsl::float64_t3(0.0))
    {
        return SCameraPathDelta::fromMotion(translation, rotation.z);
    }

    static inline SCameraPathDelta buildDefaultPathControlDelta(const SCameraPathControlContext& context)
    {
        return makePathDeltaFromVirtualPathMotion(context.translation, context.rotation);
    }

    static inline void appendPathDeltaEvents(
        std::vector<CVirtualGimbalEvent>& events,
        const SCameraPathDelta& delta,
        const double moveDenominator,
        const double rotationDenominator,
        const SCameraPathComparisonThresholds& thresholds = {})
    {
        CCameraVirtualEventUtilities::appendLocalTranslationEvents(
            events,
            delta.translationVector(),
            hlsl::float64_t3(moveDenominator),
            hlsl::float64_t3(thresholds.scalarTolerance));
        CCameraVirtualEventUtilities::appendAngularDeltaEvent(
            events,
            delta.roll,
            rotationDenominator,
            thresholds.rollToleranceDeg,
            CVirtualGimbalEvent::RollRight,
            CVirtualGimbalEvent::RollLeft);
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
        stateVector.x = hlsl::CCameraMathUtilities::wrapAngleRad(stateVector.x);
        stateVector.w = hlsl::CCameraMathUtilities::wrapAngleRad(stateVector.w);
        outState = ICamera::PathState::fromVector(stateVector);
        return sanitizePathState(outState, limits.minU);
    }

    static inline ICamera::PathState blendPathStates(
        const ICamera::PathState& from,
        const ICamera::PathState& to,
        const double alpha)
    {
        const auto fromVector = from.asVector();
        const auto toVector = to.asVector();
        return {
            .s = hlsl::CCameraMathUtilities::lerpWrappedAngleRad(fromVector.x, toVector.x, alpha),
            .u = fromVector.y + (toVector.y - fromVector.y) * alpha,
            .v = fromVector.z + (toVector.z - fromVector.z) * alpha,
            .roll = hlsl::CCameraMathUtilities::lerpWrappedAngleRad(fromVector.w, toVector.w, alpha)
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

    static inline SCameraPathModel makeDefaultPathModel()
    {
        return {
            .resolveState =
                [](const hlsl::float64_t3& targetPosition,
                    const hlsl::float64_t3& position,
                    const SCameraPathLimits& limits,
                    const ICamera::PathState* requestedState,
                    ICamera::PathState& outState) -> bool
                {
                    return tryResolvePathState(targetPosition, position, limits, requestedState, outState);
                },
            .controlLaw =
                [](const SCameraPathControlContext& context) -> SCameraPathDelta
                {
                    return buildDefaultPathControlDelta(context);
                },
            .integrate =
                [](const ICamera::PathState& currentState,
                    const SCameraPathDelta& delta,
                    const SCameraPathLimits& limits,
                    ICamera::PathState& outState) -> bool
                {
                    return tryApplyPathStateDelta(currentState, delta, limits, outState);
                },
            .evaluate =
                [](const hlsl::float64_t3& targetPosition,
                    const ICamera::PathState& state,
                    const SCameraPathLimits& limits,
                    SCameraCanonicalPathState& outState) -> bool
                {
                    return tryBuildCanonicalPathState(targetPosition, state, limits, outState);
                },
            .updateDistance =
                [](const float desiredDistance,
                    const SCameraPathLimits& limits,
                    ICamera::PathState& ioState,
                    SCameraPathDistanceUpdateResult* outResult) -> bool
                {
                    return tryUpdatePathStateDistance(desiredDistance, limits, ioState, outResult);
                }
        };
    }
};

} // namespace nbl::core

#endif // _C_CAMERA_PATH_UTILITIES_HPP_
