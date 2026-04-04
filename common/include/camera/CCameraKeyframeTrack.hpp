// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_KEYFRAME_TRACK_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "CCameraPreset.hpp"

namespace nbl::hlsl
{

//! Reusable keyframe container plus selection state for playback tooling.
struct CCameraKeyframeTrack
{
    std::vector<CCameraKeyframe> keyframes;
    int selectedKeyframeIx = -1;
};

inline bool tryBuildKeyframeTrackPresetAtTime(const CCameraKeyframeTrack& track, const float time, CCameraPreset& preset)
{
    if (track.keyframes.empty())
        return false;

    if (track.keyframes.size() == 1u)
    {
        preset = track.keyframes.front().preset;
        return true;
    }

    const auto clampedTime = std::clamp(time, 0.f, track.keyframes.back().time);
    size_t idx = 0u;
    while (idx + 1u < track.keyframes.size() && track.keyframes[idx + 1u].time < clampedTime)
        ++idx;

    const auto& a = track.keyframes[idx];
    const auto& b = track.keyframes[std::min(idx + 1u, track.keyframes.size() - 1u)];
    if (b.time <= a.time)
    {
        preset = a.preset;
        return true;
    }

    const double alpha = static_cast<double>(clampedTime - a.time) / static_cast<double>(b.time - a.time);
    preset = a.preset;
    assignGoalToPreset(preset, blendGoals(makeGoalFromPreset(a.preset), makeGoalFromPreset(b.preset), alpha));
    return true;
}

inline void sortKeyframeTrackByTime(CCameraKeyframeTrack& track)
{
    std::sort(track.keyframes.begin(), track.keyframes.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
}

inline void clampTrackTimeToKeyframes(const CCameraKeyframeTrack& track, float& time)
{
    if (track.keyframes.empty())
    {
        time = 0.f;
        return;
    }

    time = std::clamp(time, 0.f, track.keyframes.back().time);
}

inline int selectKeyframeTrackNearestTime(CCameraKeyframeTrack& track, const float time)
{
    if (track.keyframes.empty())
    {
        track.selectedKeyframeIx = -1;
        return track.selectedKeyframeIx;
    }

    size_t bestIx = 0u;
    float bestDelta = std::abs(track.keyframes.front().time - time);
    for (size_t i = 1u; i < track.keyframes.size(); ++i)
    {
        const float delta = std::abs(track.keyframes[i].time - time);
        if (delta < bestDelta)
        {
            bestDelta = delta;
            bestIx = i;
        }
    }

    track.selectedKeyframeIx = static_cast<int>(bestIx);
    return track.selectedKeyframeIx;
}

inline void normalizeSelectedKeyframeTrack(CCameraKeyframeTrack& track)
{
    if (track.keyframes.empty())
    {
        track.selectedKeyframeIx = -1;
        return;
    }

    if (track.selectedKeyframeIx < 0)
        track.selectedKeyframeIx = 0;
    else if (track.selectedKeyframeIx >= static_cast<int>(track.keyframes.size()))
        track.selectedKeyframeIx = static_cast<int>(track.keyframes.size()) - 1;
}

inline CCameraKeyframe* getSelectedKeyframe(CCameraKeyframeTrack& track)
{
    normalizeSelectedKeyframeTrack(track);
    if (track.selectedKeyframeIx < 0)
        return nullptr;
    return &track.keyframes[static_cast<size_t>(track.selectedKeyframeIx)];
}

inline const CCameraKeyframe* getSelectedKeyframe(const CCameraKeyframeTrack& track)
{
    if (track.selectedKeyframeIx < 0 || track.selectedKeyframeIx >= static_cast<int>(track.keyframes.size()))
        return nullptr;
    return &track.keyframes[static_cast<size_t>(track.selectedKeyframeIx)];
}

inline bool replaceSelectedKeyframePreset(CCameraKeyframeTrack& track, CCameraPreset preset)
{
    auto* selected = getSelectedKeyframe(track);
    if (!selected)
        return false;

    selected->preset = std::move(preset);
    return true;
}

inline nlohmann::json serializeKeyframeTrack(const CCameraKeyframeTrack& track)
{
    nlohmann::json root;
    root["keyframes"] = nlohmann::json::array();

    for (const auto& keyframe : track.keyframes)
    {
        auto j = serializePreset(keyframe.preset);
        j["time"] = keyframe.time;
        root["keyframes"].push_back(std::move(j));
    }

    return root;
}

inline bool deserializeKeyframeTrack(const nlohmann::json& root, CCameraKeyframeTrack& track)
{
    if (!root.contains("keyframes"))
        return false;

    track = {};
    for (const auto& entry : root["keyframes"])
    {
        CCameraKeyframe keyframe;
        if (entry.contains("time"))
            keyframe.time = std::max(0.f, entry["time"].get<float>());
        deserializePreset(entry, keyframe.preset);
        track.keyframes.emplace_back(std::move(keyframe));
    }

    sortKeyframeTrackByTime(track);
    normalizeSelectedKeyframeTrack(track);
    return true;
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_KEYFRAME_TRACK_HPP_
