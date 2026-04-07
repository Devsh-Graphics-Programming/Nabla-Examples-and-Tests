// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "camera/CCameraPersistence.hpp"

#include <array>
#include <fstream>

#include "nlohmann/json.hpp"

namespace
{
using json_t = nlohmann::json;

json_t serializeGoalJson(const nbl::core::CCameraGoal& goal)
{
    json_t json;
    json["position"] = { goal.position.x, goal.position.y, goal.position.z };
    json["orientation"] = {
        goal.orientation.data.x,
        goal.orientation.data.y,
        goal.orientation.data.z,
        goal.orientation.data.w
    };
    json["camera_kind"] = static_cast<uint32_t>(goal.sourceKind);
    json["camera_capabilities"] = goal.sourceCapabilities;
    json["camera_goal_state_mask"] = goal.sourceGoalStateMask;

    if (goal.hasTargetPosition)
        json["target_position"] = { goal.targetPosition.x, goal.targetPosition.y, goal.targetPosition.z };
    if (goal.hasDistance)
        json["distance"] = goal.distance;
    if (goal.hasOrbitState)
    {
        json["orbit_u"] = goal.orbitU;
        json["orbit_v"] = goal.orbitV;
        json["orbit_distance"] = goal.orbitDistance;
    }
    if (goal.hasPathState)
    {
        json["path_angle"] = goal.pathState.angle;
        json["path_radius"] = goal.pathState.radius;
        json["path_height"] = goal.pathState.height;
    }
    if (goal.hasDynamicPerspectiveState)
    {
        json["dynamic_base_fov"] = goal.dynamicPerspectiveState.baseFov;
        json["dynamic_reference_distance"] = goal.dynamicPerspectiveState.referenceDistance;
    }

    return json;
}

void deserializeGoalJson(const json_t& entry, nbl::core::CCameraGoal& goal)
{
    goal = {};

    if (entry.contains("camera_kind"))
        goal.sourceKind = static_cast<nbl::core::ICamera::CameraKind>(entry["camera_kind"].get<uint32_t>());
    if (entry.contains("camera_capabilities"))
        goal.sourceCapabilities = entry["camera_capabilities"].get<uint32_t>();
    if (entry.contains("camera_goal_state_mask"))
        goal.sourceGoalStateMask = entry["camera_goal_state_mask"].get<uint32_t>();

    if (entry.contains("position") && entry["position"].is_array())
    {
        const auto values = entry["position"].get<std::array<double, 3>>();
        goal.position = nbl::hlsl::float64_t3(values[0], values[1], values[2]);
    }
    if (entry.contains("orientation") && entry["orientation"].is_array())
    {
        const auto values = entry["orientation"].get<std::array<nbl::hlsl::float64_t, 4>>();
        goal.orientation = nbl::hlsl::makeQuaternionFromComponents<nbl::hlsl::float64_t>(
            values[0],
            values[1],
            values[2],
            values[3]);
    }
    if (entry.contains("target_position") && entry["target_position"].is_array())
    {
        const auto values = entry["target_position"].get<std::array<double, 3>>();
        goal.targetPosition = nbl::hlsl::float64_t3(values[0], values[1], values[2]);
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

json_t serializePresetJson(const nbl::core::CCameraPreset& preset)
{
    auto json = serializeGoalJson(nbl::core::makeGoalFromPreset(preset));
    json["name"] = preset.name;
    json["identifier"] = preset.identifier;
    return json;
}

void deserializePresetJson(const json_t& entry, nbl::core::CCameraPreset& preset)
{
    preset = {};
    if (entry.contains("name"))
        preset.name = entry["name"].get<std::string>();
    if (entry.contains("identifier"))
        preset.identifier = entry["identifier"].get<std::string>();

    nbl::core::CCameraGoal goal;
    deserializeGoalJson(entry, goal);
    nbl::core::assignGoalToPreset(preset, goal);
}

json_t serializeKeyframeTrackJson(const nbl::core::CCameraKeyframeTrack& track)
{
    json_t root;
    root["keyframes"] = json_t::array();

    for (const auto& keyframe : track.keyframes)
    {
        auto json = serializePresetJson(keyframe.preset);
        json["time"] = keyframe.time;
        root["keyframes"].push_back(std::move(json));
    }

    return root;
}

bool deserializeKeyframeTrackJson(const json_t& root, nbl::core::CCameraKeyframeTrack& track)
{
    if (!root.contains("keyframes") || !root["keyframes"].is_array())
        return false;

    track = {};
    for (const auto& entry : root["keyframes"])
    {
        nbl::core::CCameraKeyframe keyframe;
        if (entry.contains("time"))
            keyframe.time = std::max(0.f, entry["time"].get<float>());
        deserializePresetJson(entry, keyframe.preset);
        track.keyframes.emplace_back(std::move(keyframe));
    }

    nbl::core::sortKeyframeTrackByTime(track);
    nbl::core::normalizeSelectedKeyframeTrack(track);
    return true;
}

json_t serializePresetCollectionJson(std::span<const nbl::core::CCameraPreset> presets)
{
    json_t root;
    root["presets"] = json_t::array();
    for (const auto& preset : presets)
        root["presets"].push_back(serializePresetJson(preset));
    return root;
}

bool deserializePresetCollectionJson(const json_t& root, std::vector<nbl::core::CCameraPreset>& presets)
{
    if (!root.contains("presets") || !root["presets"].is_array())
        return false;

    std::vector<nbl::core::CCameraPreset> loadedPresets;
    loadedPresets.reserve(root["presets"].size());
    for (const auto& entry : root["presets"])
    {
        nbl::core::CCameraPreset preset;
        deserializePresetJson(entry, preset);
        loadedPresets.emplace_back(std::move(preset));
    }

    presets = std::move(loadedPresets);
    return true;
}
} // anonymous namespace

namespace nbl::system
{

bool writeGoal(std::ostream& out, const core::CCameraGoal& goal, const int indent)
{
    if (!out)
        return false;

    out << serializeGoalJson(goal).dump(indent);
    return static_cast<bool>(out);
}

bool readGoal(std::istream& in, core::CCameraGoal& goal)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    deserializeGoalJson(root, goal);
    return true;
}

bool saveGoalToFile(const path& filePath, const core::CCameraGoal& goal, const int indent)
{
    std::ofstream out(filePath.string(), std::ios::binary);
    return writeGoal(out, goal, indent);
}

bool loadGoalFromFile(const path& filePath, core::CCameraGoal& goal)
{
    std::ifstream in(filePath.string(), std::ios::binary);
    return readGoal(in, goal);
}

bool writePreset(std::ostream& out, const core::CCameraPreset& preset, const int indent)
{
    if (!out)
        return false;

    out << serializePresetJson(preset).dump(indent);
    return static_cast<bool>(out);
}

bool readPreset(std::istream& in, core::CCameraPreset& preset)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    deserializePresetJson(root, preset);
    return true;
}

bool savePresetToFile(const path& filePath, const core::CCameraPreset& preset, const int indent)
{
    std::ofstream out(filePath.string(), std::ios::binary);
    return writePreset(out, preset, indent);
}

bool loadPresetFromFile(const path& filePath, core::CCameraPreset& preset)
{
    std::ifstream in(filePath.string(), std::ios::binary);
    return readPreset(in, preset);
}

bool writeKeyframeTrack(std::ostream& out, const core::CCameraKeyframeTrack& track, const int indent)
{
    if (!out)
        return false;

    out << serializeKeyframeTrackJson(track).dump(indent);
    return static_cast<bool>(out);
}

bool readKeyframeTrack(std::istream& in, core::CCameraKeyframeTrack& track)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    return deserializeKeyframeTrackJson(root, track);
}

bool saveKeyframeTrackToFile(const path& filePath, const core::CCameraKeyframeTrack& track, const int indent)
{
    std::ofstream out(filePath.string(), std::ios::binary);
    return writeKeyframeTrack(out, track, indent);
}

bool loadKeyframeTrackFromFile(const path& filePath, core::CCameraKeyframeTrack& track)
{
    std::ifstream in(filePath.string(), std::ios::binary);
    return readKeyframeTrack(in, track);
}

bool writePresetCollection(std::ostream& out, std::span<const core::CCameraPreset> presets, const int indent)
{
    if (!out)
        return false;

    out << serializePresetCollectionJson(presets).dump(indent);
    return static_cast<bool>(out);
}

bool readPresetCollection(std::istream& in, std::vector<core::CCameraPreset>& presets)
{
    if (!in)
        return false;

    json_t root;
    in >> root;
    return deserializePresetCollectionJson(root, presets);
}

bool savePresetCollectionToFile(const path& filePath, std::span<const core::CCameraPreset> presets, const int indent)
{
    std::ofstream out(filePath.string(), std::ios::binary);
    return writePresetCollection(out, presets, indent);
}

bool loadPresetCollectionFromFile(const path& filePath, std::vector<core::CCameraPreset>& presets)
{
    std::ifstream in(filePath.string(), std::ios::binary);
    return readPresetCollection(in, presets);
}

} // namespace nbl::system
