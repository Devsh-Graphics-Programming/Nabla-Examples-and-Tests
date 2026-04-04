// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PERSISTENCE_HPP_
#define _C_CAMERA_PERSISTENCE_HPP_

#include <fstream>
#include <span>

#include "CCameraKeyframeTrack.hpp"
#include "nbl/system/path.h"

namespace nbl::hlsl
{

//! JSON and file persistence helpers for reusable camera presets and playback tracks.
//! Stream-based helpers exist first so examples can choose between file IO and in-memory round-trips.
inline nlohmann::json serializePresetCollection(std::span<const CCameraPreset> presets)
{
    nlohmann::json root;
    root["presets"] = nlohmann::json::array();
    for (const auto& preset : presets)
        root["presets"].push_back(serializePreset(preset));
    return root;
}

//! Parse a preset collection from JSON and replace the destination vector on success.
inline bool deserializePresetCollection(const nlohmann::json& root, std::vector<CCameraPreset>& presets)
{
    if (!root.contains("presets") || !root["presets"].is_array())
        return false;

    std::vector<CCameraPreset> loadedPresets;
    loadedPresets.reserve(root["presets"].size());
    for (const auto& entry : root["presets"])
    {
        CCameraPreset preset;
        deserializePreset(entry, preset);
        loadedPresets.emplace_back(std::move(preset));
    }

    presets = std::move(loadedPresets);
    return true;
}

//! Serialize a preset collection to an arbitrary stream.
inline bool writePresetCollection(std::ostream& out, std::span<const CCameraPreset> presets, const int indent = 2)
{
    if (!out)
        return false;

    out << serializePresetCollection(presets).dump(indent);
    return static_cast<bool>(out);
}

//! Deserialize a preset collection from an arbitrary stream.
inline bool readPresetCollection(std::istream& in, std::vector<CCameraPreset>& presets)
{
    if (!in)
        return false;

    nlohmann::json root;
    in >> root;
    return deserializePresetCollection(root, presets);
}

//! Convenience file wrapper around `writePresetCollection`.
inline bool savePresetCollectionToFile(const system::path& path, std::span<const CCameraPreset> presets, const int indent = 2)
{
    std::ofstream out(path.string(), std::ios::binary);
    return writePresetCollection(out, presets, indent);
}

//! Convenience file wrapper around `readPresetCollection`.
inline bool loadPresetCollectionFromFile(const system::path& path, std::vector<CCameraPreset>& presets)
{
    std::ifstream in(path.string(), std::ios::binary);
    return readPresetCollection(in, presets);
}

//! Serialize a keyframe track to an arbitrary stream.
inline bool writeKeyframeTrack(std::ostream& out, const CCameraKeyframeTrack& track, const int indent = 2)
{
    if (!out)
        return false;

    out << serializeKeyframeTrack(track).dump(indent);
    return static_cast<bool>(out);
}

//! Deserialize a keyframe track from an arbitrary stream.
inline bool readKeyframeTrack(std::istream& in, CCameraKeyframeTrack& track)
{
    if (!in)
        return false;

    nlohmann::json root;
    in >> root;
    return deserializeKeyframeTrack(root, track);
}

//! Convenience file wrapper around `writeKeyframeTrack`.
inline bool saveKeyframeTrackToFile(const system::path& path, const CCameraKeyframeTrack& track, const int indent = 2)
{
    std::ofstream out(path.string(), std::ios::binary);
    return writeKeyframeTrack(out, track, indent);
}

//! Convenience file wrapper around `readKeyframeTrack`.
inline bool loadKeyframeTrackFromFile(const system::path& path, CCameraKeyframeTrack& track)
{
    std::ifstream in(path.string(), std::ios::binary);
    return readKeyframeTrack(in, track);
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_PERSISTENCE_HPP_
