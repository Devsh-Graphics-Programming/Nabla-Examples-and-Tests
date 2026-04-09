#ifndef _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_
#define _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_

#include <limits>

#include "CCameraVirtualEventUtilities.hpp"

namespace nbl::core
{

//! Canonical target-relative orbit state shared by spherical cameras, follow, and goal solving.
struct SCameraTargetRelativeState final
{
    hlsl::float64_t3 target = hlsl::float64_t3(0.0);
    double orbitU = 0.0;
    double orbitV = 0.0;
    float distance = ICamera::SphericalMinDistance;
};

//! Pose reconstructed from a target-relative orbit state.
struct SCameraTargetRelativePose final
{
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
    hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(ICamera::SphericalMinDistance);
};

//! Derived basis for target-relative orbit rigs.
struct SCameraTargetRelativeBasis final
{
    hlsl::float64_t3 localOffset = hlsl::float64_t3(0.0);
    hlsl::float64_t3 right = hlsl::float64_t3(1.0, 0.0, 0.0);
    hlsl::float64_t3 up = hlsl::float64_t3(0.0, 0.0, 1.0);
    hlsl::float64_t3 forward = hlsl::float64_t3(0.0, 1.0, 0.0);
};

//! Delta between current spherical target state and canonical target-relative goal.
struct SCameraTargetRelativeDelta final
{
    double orbitU = 0.0;
    double orbitV = 0.0;
    double distance = 0.0;

    inline hlsl::float64_t3 orbitVector() const
    {
        return hlsl::float64_t3(orbitV, orbitU, 0.0);
    }
};

struct SCameraTargetRelativeEventPolicy final
{
    bool translateOrbit = false;
    bool allowYaw = true;
    bool allowPitch = true;
    SCameraVirtualEventAxisBinding distanceBinding = {
        CVirtualGimbalEvent::MoveForward,
        CVirtualGimbalEvent::MoveBackward
    };
};

//! Shared authored/default constants for target-relative rigs.
struct SCameraTargetRelativeRigDefaults final
{
    static constexpr float InitialDistance = 1.0f;
    static constexpr double ArcballPitchLimitRad = hlsl::SCameraViewRigDefaults::ArcballPitchLimitRad;
    static constexpr double TurntablePitchLimitRad = hlsl::SCameraViewRigDefaults::TurntablePitchLimitRad;
    static constexpr double ChaseMaxPitchRad = hlsl::SCameraViewRigDefaults::ChaseMaxPitchRad;
    static constexpr double ChaseMinPitchRad = hlsl::SCameraViewRigDefaults::ChaseMinPitchRad;
    static constexpr double DollyPitchLimitRad = hlsl::SCameraViewRigDefaults::DollyPitchLimitRad;
    static constexpr double TopDownPitchRad = hlsl::SCameraViewRigDefaults::TopDownPitchRad;
    static constexpr double IsometricYawRad = hlsl::SCameraViewRigDefaults::IsometricYawRad;
    static constexpr double IsometricPitchRad = hlsl::SCameraViewRigDefaults::IsometricPitchRad;

    static inline constexpr SCameraTargetRelativeEventPolicy OrbitTranslatePolicy = {
        .translateOrbit = true
    };
    static inline constexpr SCameraTargetRelativeEventPolicy RotateDistancePolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true
    };
    static inline constexpr SCameraTargetRelativeEventPolicy TopDownPolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = false
    };
    static inline constexpr SCameraTargetRelativeEventPolicy IsometricPolicy = {
        .translateOrbit = false,
        .allowYaw = false,
        .allowPitch = false
    };
    static inline constexpr SCameraTargetRelativeEventPolicy DollyPolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true,
        .distanceBinding = {
            CVirtualGimbalEvent::None,
            CVirtualGimbalEvent::None
        }
    };
    static inline constexpr SCameraTargetRelativeEventPolicy ChasePolicy = {
        .translateOrbit = false,
        .allowYaw = true,
        .allowPitch = true,
        .distanceBinding = {
            CVirtualGimbalEvent::MoveUp,
            CVirtualGimbalEvent::MoveDown
        }
    };
};

inline bool tryBuildTargetRelativeStateFromPosition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const float minDistance,
    const float maxDistance,
    SCameraTargetRelativeState& outState)
{
    outState = {};
    outState.target = targetPosition;

    hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(minDistance);
    if (!hlsl::tryBuildOrbitFromPosition(
            targetPosition,
            position,
            static_cast<hlsl::float64_t>(minDistance),
            static_cast<hlsl::float64_t>(maxDistance),
            outState.orbitU,
            outState.orbitV,
            appliedDistance))
    {
        return false;
    }

    outState.distance = static_cast<float>(appliedDistance);
    return true;
}

inline bool tryBuildTargetRelativePoseFromState(
    const SCameraTargetRelativeState& state,
    const float minDistance,
    const float maxDistance,
    SCameraTargetRelativePose& outPose)
{
    outPose = {};
    return hlsl::tryBuildSphericalPoseFromOrbit(
        state.target,
        state.orbitU,
        state.orbitV,
        static_cast<hlsl::float64_t>(state.distance),
        static_cast<hlsl::float64_t>(minDistance),
        static_cast<hlsl::float64_t>(maxDistance),
        outPose.position,
        outPose.orientation,
        &outPose.appliedDistance);
}

inline bool tryBuildTargetRelativeBasis(
    const SCameraTargetRelativeState& state,
    const float minDistance,
    const float maxDistance,
    SCameraTargetRelativeBasis& outBasis)
{
    SCameraTargetRelativePose pose = {};
    if (!tryBuildTargetRelativePoseFromState(state, minDistance, maxDistance, pose))
        return false;

    outBasis.localOffset = pose.position - state.target;
    const auto basis = hlsl::getQuaternionBasisMatrix(pose.orientation);
    outBasis.right = basis[0];
    outBasis.up = basis[1];
    outBasis.forward = basis[2];
    return true;
}

inline bool tryBuildTargetRelativePoseFromPosition(
    const hlsl::float64_t3& targetPosition,
    const hlsl::float64_t3& position,
    const float minDistance,
    const float maxDistance,
    SCameraTargetRelativePose& outPose,
    SCameraTargetRelativeState* outState = nullptr)
{
    SCameraTargetRelativeState state = {};
    if (!tryBuildTargetRelativeStateFromPosition(targetPosition, position, minDistance, maxDistance, state))
        return false;

    if (!tryBuildTargetRelativePoseFromState(state, minDistance, maxDistance, outPose))
        return false;

    if (outState)
        *outState = state;
    return true;
}

inline SCameraTargetRelativeDelta buildTargetRelativeDelta(
    const ICamera::SphericalTargetState& currentState,
    const SCameraTargetRelativeState& desiredState)
{
    return {
        .orbitU = hlsl::wrapAngleRad(desiredState.orbitU - currentState.u),
        .orbitV = hlsl::wrapAngleRad(desiredState.orbitV - currentState.v),
        .distance = static_cast<double>(desiredState.distance - currentState.distance)
    };
}

inline void appendTargetRelativeDeltaEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const SCameraTargetRelativeDelta& delta,
    const double angularDenominator,
    const double angularToleranceDeg,
    const double distanceDenominator,
    const double distanceTolerance,
    const SCameraTargetRelativeEventPolicy& policy)
{
    if (policy.translateOrbit)
    {
        appendAngularAxisEvents(
            events,
            delta.orbitVector(),
            hlsl::float64_t3(angularDenominator),
            hlsl::float64_t3(angularToleranceDeg, angularToleranceDeg, std::numeric_limits<hlsl::float64_t>::infinity()),
            {{
                { CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft },
                { CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown },
                { CVirtualGimbalEvent::None, CVirtualGimbalEvent::None }
            }});
    }
    else
    {
        if (policy.allowYaw)
        {
            appendAngularDeltaEvent(
                events,
                delta.orbitU,
                angularDenominator,
                angularToleranceDeg,
                CVirtualGimbalEvent::PanRight,
                CVirtualGimbalEvent::PanLeft);
        }
        if (policy.allowPitch)
        {
            appendAngularDeltaEvent(
                events,
                delta.orbitV,
                angularDenominator,
                angularToleranceDeg,
                CVirtualGimbalEvent::TiltUp,
                CVirtualGimbalEvent::TiltDown);
        }
    }

    if (policy.distanceBinding.positive != CVirtualGimbalEvent::None &&
        policy.distanceBinding.negative != CVirtualGimbalEvent::None)
    {
        appendScaledVirtualEvent(
            events,
            delta.distance,
            distanceDenominator,
            distanceTolerance,
            policy.distanceBinding.positive,
            policy.distanceBinding.negative);
    }
}

} // namespace nbl::core

#endif // _C_CAMERA_TARGET_RELATIVE_UTILITIES_HPP_
