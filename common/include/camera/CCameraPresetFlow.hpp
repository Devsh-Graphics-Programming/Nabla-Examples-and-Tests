// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_FLOW_HPP_
#define _C_CAMERA_PRESET_FLOW_HPP_

#include <string>
#include <string_view>

#include "CCameraGoalAnalysis.hpp"

namespace nbl::hlsl
{

//! Reusable preset capture, comparison, and best-effort apply helpers.
inline bool comparePresetToCameraState(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset,
    const double posEps, const double rotEpsDeg, const double scalarEps)
{
    const auto capture = solver.captureDetailed(camera);
    if (!capture.canUseGoal())
        return false;

    return compareGoals(capture.goal, makeGoalFromPreset(preset), posEps, rotEpsDeg, scalarEps);
}

inline std::string describePresetCameraMismatch(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
{
    const auto capture = solver.captureDetailed(camera);
    if (!capture.hasCamera)
        return "camera=null";
    if (!capture.captured)
        return "goal_state=unavailable";
    if (!capture.finiteGoal)
        return "goal_state=invalid";

    return describeGoalMismatch(capture.goal, makeGoalFromPreset(preset));
}

inline bool tryCapturePreset(const SCameraCaptureAnalysis& captureAnalysis, ICamera* camera, std::string_view name, CCameraPreset& preset)
{
    preset = {};
    preset.name = std::string(name);
    if (!captureAnalysis.canCapture || !camera)
        return false;

    preset.identifier = std::string(camera->getIdentifier());
    assignGoalToPreset(preset, captureAnalysis.goal);
    return true;
}

inline bool tryCapturePreset(const CCameraGoalSolver& solver, ICamera* camera, std::string_view name, CCameraPreset& preset)
{
    return tryCapturePreset(analyzeCameraCapture(solver, camera), camera, name, preset);
}

inline CCameraPreset capturePreset(const CCameraGoalSolver& solver, ICamera* camera, std::string_view name)
{
    CCameraPreset preset;
    tryCapturePreset(solver, camera, name, preset);
    return preset;
}

inline CCameraGoalSolver::SApplyResult applyPresetDetailed(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
{
    if (!camera)
        return {};

    return solver.applyDetailed(camera, makeGoalFromPreset(preset));
}

inline bool applyPreset(const CCameraGoalSolver& solver, ICamera* camera, const CCameraPreset& preset)
{
    return applyPresetDetailed(solver, camera, preset).succeeded();
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_PRESET_FLOW_HPP_
