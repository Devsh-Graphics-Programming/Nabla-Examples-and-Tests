// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_
#define _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_

#include <span>

#include "CCameraKeyframeTrack.hpp"
#include "CCameraMathUtilities.hpp"
#include "CCameraPresetFlow.hpp"
#include "ICamera.hpp"

namespace nbl::system
{

//! Small shared bundle for smoke/regression pose deltas measured around one camera manipulation.
struct SCameraManipulationDelta
{
    double position = 0.0;
    double rotationDeg = 0.0;
};

struct SCameraSmokeComparisonThresholds final
{
    static constexpr double TinyScalarEpsilon = core::ICamera::TinyScalarEpsilon;
    static constexpr double DefaultPositionTolerance = core::ICamera::DefaultPositionTolerance;
    static constexpr double DefaultAngularToleranceDeg = core::ICamera::DefaultAngularToleranceDeg;
    static constexpr double DefaultScalarTolerance = core::ICamera::ScalarTolerance;
    static constexpr double StrictPositionTolerance = core::ICamera::ScalarTolerance;
    static constexpr double StrictAngularToleranceDeg = core::ICamera::DefaultAngularToleranceDeg;
    static constexpr double StrictScalarTolerance = core::ICamera::ScalarTolerance;
    static constexpr double TrackTimeTolerance = core::ICamera::ScalarTolerance;
};

//! Measure one camera pose delta against an authored reference pose.
inline bool tryComputeCameraManipulationDelta(
    core::ICamera* camera,
    const hlsl::float64_t3& beforePosition,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& beforeOrientation,
    SCameraManipulationDelta& outDelta)
{
    outDelta = {};
    if (!camera)
        return false;

    const auto& gimbal = camera->getGimbal();
    const auto afterPosition = gimbal.getPosition();
    const auto afterOrientation = hlsl::normalizeQuaternion(gimbal.getOrientation());
    if (!hlsl::isFiniteVec3(afterPosition) || !hlsl::isFiniteQuaternion(beforeOrientation) || !hlsl::isFiniteQuaternion(afterOrientation))
        return false;

    outDelta.position = hlsl::length(afterPosition - beforePosition);
    outDelta.rotationDeg = hlsl::getQuaternionAngularDistanceDegrees(beforeOrientation, afterOrientation);
    return true;
}

//! Manipulate a camera and report how far its pose moved in position and Euler-angle terms.
inline bool tryManipulateCameraAndMeasureDelta(
    core::ICamera* camera,
    std::span<const core::CVirtualGimbalEvent> events,
    SCameraManipulationDelta& outDelta,
    const double tinyEpsilon = SCameraSmokeComparisonThresholds::TinyScalarEpsilon)
{
    outDelta = {};
    if (!camera || events.empty())
        return false;

    const auto& beforeGimbal = camera->getGimbal();
    const auto beforePosition = beforeGimbal.getPosition();
    const auto beforeOrientation = hlsl::normalizeQuaternion(beforeGimbal.getOrientation());
    if (!hlsl::isFiniteVec3(beforePosition) || !hlsl::isFiniteQuaternion(beforeOrientation))
        return false;

    if (!camera->manipulate(events))
        return false;

    if (!tryComputeCameraManipulationDelta(camera, beforePosition, beforeOrientation, outDelta))
        return false;

    return outDelta.position > tinyEpsilon || outDelta.rotationDeg > tinyEpsilon;
}

inline bool comparePresetToCameraStateWithDefaultThresholds(
    const core::CCameraGoalSolver& solver,
    core::ICamera* camera,
    const core::CCameraPreset& preset)
{
    return core::comparePresetToCameraState(
        solver,
        camera,
        preset,
        SCameraSmokeComparisonThresholds::DefaultPositionTolerance,
        SCameraSmokeComparisonThresholds::DefaultAngularToleranceDeg,
        SCameraSmokeComparisonThresholds::DefaultScalarTolerance);
}

inline bool comparePresetToCameraStateWithStrictThresholds(
    const core::CCameraGoalSolver& solver,
    core::ICamera* camera,
    const core::CCameraPreset& preset)
{
    return core::comparePresetToCameraState(
        solver,
        camera,
        preset,
        SCameraSmokeComparisonThresholds::StrictPositionTolerance,
        SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
        SCameraSmokeComparisonThresholds::StrictScalarTolerance);
}

inline bool compareKeyframeTrackContentWithStrictThresholds(
    const core::CCameraKeyframeTrack& lhs,
    const core::CCameraKeyframeTrack& rhs)
{
    return core::compareKeyframeTrackContent(
        lhs,
        rhs,
        SCameraSmokeComparisonThresholds::TrackTimeTolerance,
        SCameraSmokeComparisonThresholds::StrictPositionTolerance,
        SCameraSmokeComparisonThresholds::StrictAngularToleranceDeg,
        SCameraSmokeComparisonThresholds::StrictScalarTolerance);
}

} // namespace nbl::system

#endif // _C_CAMERA_SMOKE_REGRESSION_UTILITIES_HPP_
