// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_GOAL_HPP_
#define _C_CAMERA_GOAL_HPP_

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include "ICamera.hpp"

namespace nbl::core
{

/**
* Typed transport object for camera state used by capture, comparison, presets, and playback.
*/
struct CCameraGoal
{
    hlsl::float64_t3 position = hlsl::float64_t3(0.0);
    hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
    ICamera::CameraKind sourceKind = ICamera::CameraKind::Unknown;
    uint32_t sourceCapabilities = ICamera::None;
    uint32_t sourceGoalStateMask = ICamera::GoalStateNone;
    bool hasTargetPosition = false;
    hlsl::float64_t3 targetPosition = hlsl::float64_t3(0.0);
    bool hasDistance = false;
    float distance = 0.f;
    bool hasOrbitState = false;
    double orbitU = 0.0;
    double orbitV = 0.0;
    float orbitDistance = 0.f;
    bool hasPathState = false;
    ICamera::PathState pathState = {};
    bool hasDynamicPerspectiveState = false;
    ICamera::DynamicPerspectiveState dynamicPerspectiveState = {};
};

inline double lerpWrappedAngleRad(double a, double b, double alpha)
{
    return a + hlsl::wrapAngleRad(b - a) * alpha;
}

inline uint32_t getRequiredGoalStateMask(const CCameraGoal& target)
{
    uint32_t mask = ICamera::GoalStateNone;
    if (target.hasTargetPosition || target.hasDistance || target.hasOrbitState)
        mask |= ICamera::GoalStateSphericalTarget;
    if (target.hasDynamicPerspectiveState)
        mask |= ICamera::GoalStateDynamicPerspective;
    if (target.hasPathState)
        mask |= ICamera::GoalStatePath;
    return mask;
}

inline bool applyCanonicalPathGoal(CCameraGoal& goal)
{
    if (!(goal.hasPathState && goal.hasTargetPosition))
        return false;
    if (!std::isfinite(goal.pathState.angle) || !std::isfinite(goal.pathState.radius) || !std::isfinite(goal.pathState.height))
        return false;

    hlsl::float64_t appliedOrbitDistance = 0.0;
    if (!hlsl::tryBuildPathPoseFromState(
        goal.targetPosition,
        goal.pathState.angle,
        goal.pathState.radius,
        goal.pathState.height,
        static_cast<hlsl::float64_t>(ICamera::SphericalMinDistance),
        static_cast<hlsl::float64_t>(ICamera::SphericalMinDistance),
        static_cast<hlsl::float64_t>(ICamera::SphericalMaxDistance),
        goal.position,
        goal.orientation,
        &appliedOrbitDistance,
        &goal.orbitU,
        &goal.orbitV))
    {
        return false;
    }

    goal.hasDistance = true;
    goal.distance = static_cast<float>(appliedOrbitDistance);
    goal.hasOrbitState = true;
    goal.orbitDistance = static_cast<float>(appliedOrbitDistance);
    return true;
}

inline CCameraGoal canonicalizeGoal(CCameraGoal goal)
{
    applyCanonicalPathGoal(goal);
    return goal;
}

inline bool isGoalFinite(const CCameraGoal& goal)
{
    if (!hlsl::isFiniteVec3(goal.position) || !hlsl::isFiniteQuaternion(goal.orientation))
        return false;
    if (goal.hasTargetPosition && !hlsl::isFiniteVec3(goal.targetPosition))
        return false;
    if (goal.hasDistance && !std::isfinite(goal.distance))
        return false;
    if (goal.hasOrbitState && (!std::isfinite(goal.orbitU) || !std::isfinite(goal.orbitV) || !std::isfinite(goal.orbitDistance)))
        return false;
    if (goal.hasPathState && (!std::isfinite(goal.pathState.angle) || !std::isfinite(goal.pathState.radius) || !std::isfinite(goal.pathState.height)))
        return false;
    if (goal.hasDynamicPerspectiveState &&
        (!std::isfinite(goal.dynamicPerspectiveState.baseFov) || !std::isfinite(goal.dynamicPerspectiveState.referenceDistance)))
        return false;
    return true;
}

inline bool compareGoals(const CCameraGoal& actual, const CCameraGoal& expected,
    const double posEps, const double rotEpsDeg, const double scalarEps)
{
    const auto currentOrientation = hlsl::normalizeQuaternion(actual.orientation);
    const auto expectedOrientation = hlsl::normalizeQuaternion(expected.orientation);
    if (!hlsl::isFiniteVec3(actual.position) || !hlsl::isFiniteVec3(expected.position) || !hlsl::isFiniteQuaternion(currentOrientation) || !hlsl::isFiniteQuaternion(expectedOrientation))
        return false;

    const double posDelta = hlsl::length(actual.position - expected.position);
    const double rotDeltaDeg = hlsl::getQuaternionAngularDistanceDegrees(currentOrientation, expectedOrientation);
    if (posDelta > posEps || rotDeltaDeg > rotEpsDeg)
        return false;

    if (expected.hasTargetPosition)
    {
        if (!actual.hasTargetPosition || !hlsl::nearlyEqualVec3(actual.targetPosition, expected.targetPosition, scalarEps))
            return false;
    }
    if (expected.hasDistance)
    {
        if (!actual.hasDistance || !hlsl::nearlyEqualScalar(static_cast<double>(actual.distance), static_cast<double>(expected.distance), scalarEps))
            return false;
    }
    if (expected.hasOrbitState)
    {
        if (!actual.hasOrbitState)
            return false;
        if (hlsl::degrees(hlsl::getWrappedAngleDistanceRadians(expected.orbitU, actual.orbitU)) > rotEpsDeg)
            return false;
        if (hlsl::degrees(hlsl::getWrappedAngleDistanceRadians(expected.orbitV, actual.orbitV)) > rotEpsDeg)
            return false;
        if (!hlsl::nearlyEqualScalar(static_cast<double>(actual.orbitDistance), static_cast<double>(expected.orbitDistance), scalarEps))
            return false;
    }
    if (expected.hasPathState)
    {
        if (!actual.hasPathState)
            return false;
        if (hlsl::degrees(hlsl::getWrappedAngleDistanceRadians(expected.pathState.angle, actual.pathState.angle)) > rotEpsDeg)
            return false;
        if (!hlsl::nearlyEqualScalar(actual.pathState.radius, expected.pathState.radius, scalarEps))
            return false;
        if (!hlsl::nearlyEqualScalar(actual.pathState.height, expected.pathState.height, scalarEps))
            return false;
    }
    if (expected.hasDynamicPerspectiveState)
    {
        if (!actual.hasDynamicPerspectiveState)
            return false;
        if (!hlsl::nearlyEqualScalar(static_cast<double>(actual.dynamicPerspectiveState.baseFov), static_cast<double>(expected.dynamicPerspectiveState.baseFov), scalarEps))
            return false;
        if (!hlsl::nearlyEqualScalar(static_cast<double>(actual.dynamicPerspectiveState.referenceDistance), static_cast<double>(expected.dynamicPerspectiveState.referenceDistance), scalarEps))
            return false;
    }

    return true;
}

inline std::string describeGoalMismatch(const CCameraGoal& actual, const CCameraGoal& expected)
{
    std::ostringstream oss;
    const auto currentOrientation = hlsl::normalizeQuaternion(actual.orientation);
    const auto expectedOrientation = hlsl::normalizeQuaternion(expected.orientation);
    const double posDelta = hlsl::length(actual.position - expected.position);
    const double rotDeltaDeg = hlsl::getQuaternionAngularDistanceDegrees(currentOrientation, expectedOrientation);
    oss << "pos_delta=" << posDelta
        << " rot_delta_deg=" << rotDeltaDeg
        << " current_pos=(" << actual.position.x << "," << actual.position.y << "," << actual.position.z << ")"
        << " expected_pos=(" << expected.position.x << "," << expected.position.y << "," << expected.position.z << ")"
        << " current_quat=(" << currentOrientation.data.x << "," << currentOrientation.data.y << "," << currentOrientation.data.z << "," << currentOrientation.data.w << ")"
        << " expected_quat=(" << expectedOrientation.data.x << "," << expectedOrientation.data.y << "," << expectedOrientation.data.z << "," << expectedOrientation.data.w << ")";

    if (actual.hasTargetPosition)
    {
        oss << " target=(" << actual.targetPosition.x << "," << actual.targetPosition.y << "," << actual.targetPosition.z << ")";
        if (actual.hasDistance)
            oss << " distance=" << actual.distance;
        if (actual.hasOrbitState)
            oss << " orbit_u=" << actual.orbitU << " orbit_v=" << actual.orbitV;
    }
    else if (expected.hasTargetPosition || expected.hasDistance || expected.hasOrbitState)
    {
        oss << " spherical_state=unavailable";
    }
    if (actual.hasPathState)
    {
        oss << " path_angle=" << actual.pathState.angle
            << " path_radius=" << actual.pathState.radius
            << " path_height=" << actual.pathState.height;
    }
    else if (expected.hasPathState)
    {
        oss << " path_state=unavailable";
    }

    if (actual.hasDynamicPerspectiveState)
    {
        oss << " dynamic_base_fov=" << actual.dynamicPerspectiveState.baseFov
            << " dynamic_reference_distance=" << actual.dynamicPerspectiveState.referenceDistance;
    }
    else if (expected.hasDynamicPerspectiveState)
    {
        oss << " dynamic_perspective_state=unavailable";
    }

    return oss.str();
}

inline CCameraGoal blendGoals(const CCameraGoal& a, const CCameraGoal& b, double alpha)
{
    CCameraGoal blended;
    blended.position = a.position + (b.position - a.position) * alpha;
    blended.orientation = hlsl::slerpQuaternion(a.orientation, b.orientation, static_cast<hlsl::float64_t>(alpha));
    blended.sourceKind = (a.sourceKind == b.sourceKind) ? a.sourceKind : ICamera::CameraKind::Unknown;
    blended.sourceCapabilities = a.sourceCapabilities & b.sourceCapabilities;
    blended.sourceGoalStateMask = a.sourceGoalStateMask | b.sourceGoalStateMask;
    blended.hasTargetPosition = a.hasTargetPosition || b.hasTargetPosition;
    if (blended.hasTargetPosition)
    {
        const auto ta = a.hasTargetPosition ? a.targetPosition : b.targetPosition;
        const auto tb = b.hasTargetPosition ? b.targetPosition : a.targetPosition;
        blended.targetPosition = ta + (tb - ta) * alpha;
    }
    blended.hasDistance = a.hasDistance || b.hasDistance;
    if (blended.hasDistance)
    {
        const float da = a.hasDistance ? a.distance : b.distance;
        const float db = b.hasDistance ? b.distance : a.distance;
        blended.distance = da + (db - da) * static_cast<float>(alpha);
    }
    blended.hasOrbitState = a.hasOrbitState || b.hasOrbitState;
    if (blended.hasOrbitState)
    {
        const double ua = a.hasOrbitState ? a.orbitU : b.orbitU;
        const double ub = b.hasOrbitState ? b.orbitU : a.orbitU;
        const double va = a.hasOrbitState ? a.orbitV : b.orbitV;
        const double vb = b.hasOrbitState ? b.orbitV : a.orbitV;
        const float da = a.hasOrbitState ? a.orbitDistance : b.orbitDistance;
        const float db = b.hasOrbitState ? b.orbitDistance : a.orbitDistance;

        blended.orbitU = lerpWrappedAngleRad(ua, ub, alpha);
        blended.orbitV = lerpWrappedAngleRad(va, vb, alpha);
        blended.orbitDistance = da + (db - da) * static_cast<float>(alpha);
    }
    blended.hasDynamicPerspectiveState = a.hasDynamicPerspectiveState || b.hasDynamicPerspectiveState;
    if (blended.hasDynamicPerspectiveState)
    {
        const auto dynamicA = a.hasDynamicPerspectiveState ? a.dynamicPerspectiveState : b.dynamicPerspectiveState;
        const auto dynamicB = b.hasDynamicPerspectiveState ? b.dynamicPerspectiveState : a.dynamicPerspectiveState;
        blended.dynamicPerspectiveState.baseFov = dynamicA.baseFov + (dynamicB.baseFov - dynamicA.baseFov) * static_cast<float>(alpha);
        blended.dynamicPerspectiveState.referenceDistance =
            dynamicA.referenceDistance + (dynamicB.referenceDistance - dynamicA.referenceDistance) * static_cast<float>(alpha);
    }
    blended.hasPathState = a.hasPathState || b.hasPathState;
    if (blended.hasPathState)
    {
        const auto pathA = a.hasPathState ? a.pathState : b.pathState;
        const auto pathB = b.hasPathState ? b.pathState : a.pathState;
        blended.pathState.angle = lerpWrappedAngleRad(pathA.angle, pathB.angle, alpha);
        blended.pathState.radius = pathA.radius + (pathB.radius - pathA.radius) * alpha;
        blended.pathState.height = pathA.height + (pathB.height - pathA.height) * alpha;
    }
    return canonicalizeGoal(blended);
}

} // namespace nbl::core

#endif // _C_CAMERA_GOAL_HPP_
