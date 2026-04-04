// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_HPP_
#define _C_CAMERA_PRESET_HPP_

#include <string>

#include "CCameraGoal.hpp"
#include "nlohmann/json.hpp"

namespace nbl::hlsl
{

struct CCameraPreset
{
    std::string name;
    std::string identifier;
    CCameraGoal goal = {};
};

struct CCameraKeyframe
{
    CCameraPreset preset;
    float time = 0.f;
};

inline void assignGoalToPreset(CCameraPreset& preset, const CCameraGoal& goal)
{
    preset.goal = canonicalizeGoal(goal);
}

inline CCameraGoal makeGoalFromPreset(const CCameraPreset& preset)
{
    return canonicalizeGoal(preset.goal);
}

inline nlohmann::json serializeGoal(const CCameraGoal& goal)
{
    nlohmann::json j;
    j["position"] = { goal.position.x, goal.position.y, goal.position.z };
    j["orientation"] = { goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w };
    j["camera_kind"] = static_cast<uint32_t>(goal.sourceKind);
    j["camera_capabilities"] = goal.sourceCapabilities;
    j["camera_goal_state_mask"] = goal.sourceGoalStateMask;
    if (goal.hasTargetPosition)
        j["target_position"] = { goal.targetPosition.x, goal.targetPosition.y, goal.targetPosition.z };
    if (goal.hasDistance)
        j["distance"] = goal.distance;
    if (goal.hasOrbitState)
    {
        j["orbit_u"] = goal.orbitU;
        j["orbit_v"] = goal.orbitV;
        j["orbit_distance"] = goal.orbitDistance;
    }
    if (goal.hasPathState)
    {
        j["path_angle"] = goal.pathState.angle;
        j["path_radius"] = goal.pathState.radius;
        j["path_height"] = goal.pathState.height;
    }
    if (goal.hasDynamicPerspectiveState)
    {
        j["dynamic_base_fov"] = goal.dynamicPerspectiveState.baseFov;
        j["dynamic_reference_distance"] = goal.dynamicPerspectiveState.referenceDistance;
    }
    return j;
}

inline void deserializeGoal(const nlohmann::json& entry, CCameraGoal& goal)
{
    goal = {};
    if (entry.contains("camera_kind"))
        goal.sourceKind = static_cast<ICamera::CameraKind>(entry["camera_kind"].get<uint32_t>());
    if (entry.contains("camera_capabilities"))
        goal.sourceCapabilities = entry["camera_capabilities"].get<uint32_t>();
    if (entry.contains("camera_goal_state_mask"))
        goal.sourceGoalStateMask = entry["camera_goal_state_mask"].get<uint32_t>();
    if (entry.contains("position") && entry["position"].is_array())
    {
        const auto& arr = entry["position"];
        goal.position = float64_t3(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>());
    }
    if (entry.contains("orientation") && entry["orientation"].is_array())
    {
        const auto& arr = entry["orientation"];
        goal.orientation = glm::quat(
            arr[3].get<float>(),
            arr[0].get<float>(),
            arr[1].get<float>(),
            arr[2].get<float>()
        );
    }
    if (entry.contains("target_position") && entry["target_position"].is_array())
    {
        const auto& arr = entry["target_position"];
        goal.targetPosition = float64_t3(arr[0].get<double>(), arr[1].get<double>(), arr[2].get<double>());
        goal.hasTargetPosition = true;
    }
    if (entry.contains("distance"))
    {
        goal.distance = entry["distance"].get<float>();
        goal.hasDistance = true;
    }
    if (entry.contains("orbit_u"))
    {
        goal.orbitU = entry["orbit_u"].get<double>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("orbit_v"))
    {
        goal.orbitV = entry["orbit_v"].get<double>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("orbit_distance"))
    {
        goal.orbitDistance = entry["orbit_distance"].get<float>();
        goal.hasOrbitState = true;
    }
    if (entry.contains("path_angle") && entry.contains("path_radius") && entry.contains("path_height"))
    {
        goal.pathState.angle = entry["path_angle"].get<double>();
        goal.pathState.radius = entry["path_radius"].get<double>();
        goal.pathState.height = entry["path_height"].get<double>();
        goal.hasPathState = true;
    }
    if (entry.contains("dynamic_base_fov"))
    {
        goal.dynamicPerspectiveState.baseFov = entry["dynamic_base_fov"].get<float>();
        goal.hasDynamicPerspectiveState = true;
    }
    if (entry.contains("dynamic_reference_distance"))
    {
        goal.dynamicPerspectiveState.referenceDistance = entry["dynamic_reference_distance"].get<float>();
        goal.hasDynamicPerspectiveState = true;
    }
}

inline nlohmann::json serializePreset(const CCameraPreset& preset)
{
    auto j = serializeGoal(makeGoalFromPreset(preset));
    j["name"] = preset.name;
    j["identifier"] = preset.identifier;
    return j;
}

inline void deserializePreset(const nlohmann::json& entry, CCameraPreset& preset)
{
    preset = {};
    if (entry.contains("name"))
        preset.name = entry["name"].get<std::string>();
    if (entry.contains("identifier"))
        preset.identifier = entry["identifier"].get<std::string>();

    CCameraGoal goal;
    deserializeGoal(entry, goal);
    assignGoalToPreset(preset, goal);
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_PRESET_HPP_
