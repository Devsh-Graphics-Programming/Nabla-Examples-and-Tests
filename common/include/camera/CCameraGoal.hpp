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
#include "CSphericalTargetCamera.hpp"
#include "glm/glm/gtc/quaternion.hpp"

namespace nbl::hlsl
{

struct CCameraGoal
{
    float64_t3 position = float64_t3(0.0);
    glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    ICamera::CameraKind sourceKind = ICamera::CameraKind::Unknown;
    uint32_t sourceCapabilities = ICamera::None;
    uint32_t sourceGoalStateMask = ICamera::GoalStateNone;
    bool hasTargetPosition = false;
    float64_t3 targetPosition = float64_t3(0.0);
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

inline double wrapAngleRad(double angle)
{
    constexpr double Pi = 3.14159265358979323846;
    while (angle > Pi)
        angle -= 2.0 * Pi;
    while (angle < -Pi)
        angle += 2.0 * Pi;
    return angle;
}

inline double lerpWrappedAngleRad(double a, double b, double alpha)
{
    return a + wrapAngleRad(b - a) * alpha;
}

inline bool nearlyEqualGoalScalar(double a, double b, double eps = 1e-6)
{
    return std::abs(a - b) <= eps;
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

    const float64_t3 offset(
        std::cos(goal.pathState.angle) * goal.pathState.radius,
        goal.pathState.height,
        std::sin(goal.pathState.angle) * goal.pathState.radius);
    const double distance = length(offset);
    if (!std::isfinite(distance) || distance <= 1e-9)
        return false;

    const float appliedDistance = std::clamp(
        static_cast<float>(distance),
        CSphericalTargetCamera::MinDistance,
        CSphericalTargetCamera::MaxDistance);
    const auto local = offset / static_cast<double>(appliedDistance);
    goal.orbitU = std::atan2(local.y, local.x);
    goal.orbitV = std::asin(std::clamp(local.z, -1.0, 1.0));

    const float64_t3 spherePosition(
        std::cos(goal.orbitV) * std::cos(goal.orbitU) * static_cast<double>(appliedDistance),
        std::cos(goal.orbitV) * std::sin(goal.orbitU) * static_cast<double>(appliedDistance),
        std::sin(goal.orbitV) * static_cast<double>(appliedDistance));

    goal.position = goal.targetPosition + spherePosition;
    goal.hasDistance = true;
    goal.distance = appliedDistance;
    goal.hasOrbitState = true;
    goal.orbitDistance = appliedDistance;

    const auto forward = normalize(-spherePosition);
    const float64_t3 up = normalize(float64_t3(
        -std::sin(goal.orbitV) * std::cos(goal.orbitU),
        -std::sin(goal.orbitV) * std::sin(goal.orbitU),
        std::cos(goal.orbitV)));
    const float64_t3 right = normalize(cross(up, forward));
    goal.orientation = glm::quat_cast(glm::dmat3{ right, up, forward });
    return true;
}

inline CCameraGoal canonicalizeGoal(CCameraGoal goal)
{
    applyCanonicalPathGoal(goal);
    return goal;
}

template<typename Vec>
inline bool isFiniteVec3(const Vec& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

template<typename VecA, typename VecB>
inline bool nearlyEqualVec3(const VecA& a, const VecB& b, const double epsilon)
{
    return std::abs(static_cast<double>(a.x - b.x)) <= epsilon &&
        std::abs(static_cast<double>(a.y - b.y)) <= epsilon &&
        std::abs(static_cast<double>(a.z - b.z)) <= epsilon;
}

inline bool isGoalFinite(const CCameraGoal& goal)
{
    auto isFiniteQuat = [](const glm::quat& q) -> bool
    {
        return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w);
    };

    if (!isFiniteVec3(goal.position) || !isFiniteQuat(goal.orientation))
        return false;
    if (goal.hasTargetPosition && !isFiniteVec3(goal.targetPosition))
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
    auto isFiniteQuat = [](const glm::quat& q) -> bool
    {
        return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w);
    };

    auto angleDiffRad = [](double a, double b) -> double
    {
        constexpr double Pi = 3.14159265358979323846;
        constexpr double TwoPi = 6.28318530717958647692;
        double d = std::fmod(a - b + Pi, TwoPi);
        if (d < 0.0)
            d += TwoPi;
        return std::abs(d - Pi);
    };

    const auto currentOrientation = glm::normalize(actual.orientation);
    const auto expectedOrientation = glm::normalize(expected.orientation);
    if (!isFiniteVec3(actual.position) || !isFiniteVec3(expected.position) || !isFiniteQuat(currentOrientation) || !isFiniteQuat(expectedOrientation))
        return false;

    const double dx = static_cast<double>(actual.position.x - expected.position.x);
    const double dy = static_cast<double>(actual.position.y - expected.position.y);
    const double dz = static_cast<double>(actual.position.z - expected.position.z);
    const double posDelta = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double orientationDot = std::clamp(static_cast<double>(std::abs(glm::dot(currentOrientation, expectedOrientation))), 0.0, 1.0);
    const double rotDeltaDeg = glm::degrees(2.0 * std::acos(orientationDot));
    if (posDelta > posEps || rotDeltaDeg > rotEpsDeg)
        return false;

    if (expected.hasTargetPosition)
    {
        if (!actual.hasTargetPosition || !nearlyEqualVec3(actual.targetPosition, expected.targetPosition, scalarEps))
            return false;
    }
    if (expected.hasDistance)
    {
        if (!actual.hasDistance || std::abs(static_cast<double>(actual.distance - expected.distance)) > scalarEps)
            return false;
    }
    if (expected.hasOrbitState)
    {
        if (!actual.hasOrbitState)
            return false;
        if (angleDiffRad(expected.orbitU, actual.orbitU) > rotEpsDeg * (3.14159265358979323846 / 180.0))
            return false;
        if (angleDiffRad(expected.orbitV, actual.orbitV) > rotEpsDeg * (3.14159265358979323846 / 180.0))
            return false;
        if (std::abs(static_cast<double>(actual.orbitDistance - expected.orbitDistance)) > scalarEps)
            return false;
    }
    if (expected.hasPathState)
    {
        if (!actual.hasPathState)
            return false;
        if (std::abs(wrapAngleRad(expected.pathState.angle - actual.pathState.angle)) > rotEpsDeg * (3.14159265358979323846 / 180.0))
            return false;
        if (std::abs(actual.pathState.radius - expected.pathState.radius) > scalarEps)
            return false;
        if (std::abs(actual.pathState.height - expected.pathState.height) > scalarEps)
            return false;
    }
    if (expected.hasDynamicPerspectiveState)
    {
        if (!actual.hasDynamicPerspectiveState)
            return false;
        if (std::abs(static_cast<double>(actual.dynamicPerspectiveState.baseFov - expected.dynamicPerspectiveState.baseFov)) > scalarEps)
            return false;
        if (std::abs(static_cast<double>(actual.dynamicPerspectiveState.referenceDistance - expected.dynamicPerspectiveState.referenceDistance)) > scalarEps)
            return false;
    }

    return true;
}

inline std::string describeGoalMismatch(const CCameraGoal& actual, const CCameraGoal& expected)
{
    std::ostringstream oss;
    const auto currentOrientation = glm::normalize(actual.orientation);
    const auto expectedOrientation = glm::normalize(expected.orientation);
    const double dx = static_cast<double>(actual.position.x - expected.position.x);
    const double dy = static_cast<double>(actual.position.y - expected.position.y);
    const double dz = static_cast<double>(actual.position.z - expected.position.z);
    const double posDelta = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double orientationDot = std::clamp(static_cast<double>(std::abs(glm::dot(currentOrientation, expectedOrientation))), 0.0, 1.0);
    const double rotDeltaDeg = glm::degrees(2.0 * std::acos(orientationDot));
    oss << "pos_delta=" << posDelta
        << " rot_delta_deg=" << rotDeltaDeg
        << " current_pos=(" << actual.position.x << "," << actual.position.y << "," << actual.position.z << ")"
        << " expected_pos=(" << expected.position.x << "," << expected.position.y << "," << expected.position.z << ")"
        << " current_quat=(" << currentOrientation.x << "," << currentOrientation.y << "," << currentOrientation.z << "," << currentOrientation.w << ")"
        << " expected_quat=(" << expectedOrientation.x << "," << expectedOrientation.y << "," << expectedOrientation.z << "," << expectedOrientation.w << ")";

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
    blended.orientation = glm::slerp(a.orientation, b.orientation, static_cast<float>(alpha));
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

} // namespace nbl::hlsl

#endif // _C_CAMERA_GOAL_HPP_
