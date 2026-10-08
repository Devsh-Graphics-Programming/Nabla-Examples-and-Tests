// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_KEYFRAME_TRACK_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_HPP_

#include <algorithm>
#include <cmath>
#include <vector>

#include "CCameraPreset.hpp"

using namespace nbl;
using namespace nbl::ext::cameras;

/// @brief Reusable keyframe container plus selection state for playback tooling.
struct CCameraKeyframeTrack
{
    std::vector<CCameraKeyframe> keyframes;
    int selectedKeyframeIx = -1;
};

struct CCameraKeyframeTrackUtilities final
{
public:
    /// @brief Compare two keyframes by authored time and shared preset state.
    static bool compareKeyframes(const CCameraKeyframe& lhs, const CCameraKeyframe& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps);

    /// @brief Compare two authored keyframe tracks with optional selection-state checking.
    static bool compareKeyframeTracks(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps, bool compareSelection = true);

    /// @brief Compare only the serialized/authored content of two tracks and ignore transient UI selection state.
    static bool compareKeyframeTrackContent(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
        double timeEps, double posEps, double rotEpsDeg, double scalarEps);

    static bool tryBuildKeyframeTrackPresetAtTime(const CCameraKeyframeTrack& track, float time, CCameraPreset& preset);

    static void sortKeyframeTrackByTime(CCameraKeyframeTrack& track);

    static void clampTrackTimeToKeyframes(const CCameraKeyframeTrack& track, float& time);

    static int selectKeyframeTrackNearestTime(CCameraKeyframeTrack& track, float time);

    static void normalizeSelectedKeyframeTrack(CCameraKeyframeTrack& track);

    static CCameraKeyframe* getSelectedKeyframe(CCameraKeyframeTrack& track);

    static const CCameraKeyframe* getSelectedKeyframe(const CCameraKeyframeTrack& track);

    static bool replaceSelectedKeyframePreset(CCameraKeyframeTrack& track, CCameraPreset preset);
};


inline bool CCameraKeyframeTrackUtilities::compareKeyframes(const CCameraKeyframe& lhs, const CCameraKeyframe& rhs,
    const double timeEps, const double posEps, const double rotEpsDeg, const double scalarEps)
{
    return hlsl::abs(static_cast<double>(lhs.time - rhs.time)) <= timeEps &&
        CCameraPresetUtilities::comparePresets(lhs.preset, rhs.preset, posEps, rotEpsDeg, scalarEps);
}

inline bool CCameraKeyframeTrackUtilities::compareKeyframeTracks(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
    const double timeEps, const double posEps, const double rotEpsDeg, const double scalarEps, const bool compareSelection)
{
    if ((compareSelection && lhs.selectedKeyframeIx != rhs.selectedKeyframeIx) || lhs.keyframes.size() != rhs.keyframes.size())
        return false;

    for (size_t i = 0u; i < lhs.keyframes.size(); ++i)
    {
        if (!compareKeyframes(lhs.keyframes[i], rhs.keyframes[i], timeEps, posEps, rotEpsDeg, scalarEps))
            return false;
    }

    return true;
}

inline bool CCameraKeyframeTrackUtilities::compareKeyframeTrackContent(const CCameraKeyframeTrack& lhs, const CCameraKeyframeTrack& rhs,
    const double timeEps, const double posEps, const double rotEpsDeg, const double scalarEps)
{
    return compareKeyframeTracks(lhs, rhs, timeEps, posEps, rotEpsDeg, scalarEps, false);
}

inline bool CCameraKeyframeTrackUtilities::tryBuildKeyframeTrackPresetAtTime(const CCameraKeyframeTrack& track, const float time, CCameraPreset& preset)
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
    CCameraPresetUtilities::assignGoalToPreset(
        preset,
        CCameraGoalUtilities::blendGoals(
            CCameraPresetUtilities::makeGoalFromPreset(a.preset),
            CCameraPresetUtilities::makeGoalFromPreset(b.preset),
            alpha));
    return true;
}

inline void CCameraKeyframeTrackUtilities::sortKeyframeTrackByTime(CCameraKeyframeTrack& track)
{
    std::sort(track.keyframes.begin(), track.keyframes.end(), [](const auto& a, const auto& b) { return a.time < b.time; });
}

inline void CCameraKeyframeTrackUtilities::clampTrackTimeToKeyframes(const CCameraKeyframeTrack& track, float& time)
{
    if (track.keyframes.empty())
    {
        time = 0.f;
        return;
    }

    time = std::clamp(time, 0.f, track.keyframes.back().time);
}

inline int CCameraKeyframeTrackUtilities::selectKeyframeTrackNearestTime(CCameraKeyframeTrack& track, const float time)
{
    if (track.keyframes.empty())
    {
        track.selectedKeyframeIx = -1;
        return track.selectedKeyframeIx;
    }

    size_t bestIx = 0u;
    float bestDelta = hlsl::abs(track.keyframes.front().time - time);
    for (size_t i = 1u; i < track.keyframes.size(); ++i)
    {
        const float delta = hlsl::abs(track.keyframes[i].time - time);
        if (delta < bestDelta)
        {
            bestDelta = delta;
            bestIx = i;
        }
    }

    track.selectedKeyframeIx = static_cast<int>(bestIx);
    return track.selectedKeyframeIx;
}

inline void CCameraKeyframeTrackUtilities::normalizeSelectedKeyframeTrack(CCameraKeyframeTrack& track)
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

inline CCameraKeyframe* CCameraKeyframeTrackUtilities::getSelectedKeyframe(CCameraKeyframeTrack& track)
{
    normalizeSelectedKeyframeTrack(track);
    if (track.selectedKeyframeIx < 0)
        return nullptr;
    return &track.keyframes[static_cast<size_t>(track.selectedKeyframeIx)];
}

inline const CCameraKeyframe* CCameraKeyframeTrackUtilities::getSelectedKeyframe(const CCameraKeyframeTrack& track)
{
    if (track.selectedKeyframeIx < 0 || track.selectedKeyframeIx >= static_cast<int>(track.keyframes.size()))
        return nullptr;
    return &track.keyframes[static_cast<size_t>(track.selectedKeyframeIx)];
}

inline bool CCameraKeyframeTrackUtilities::replaceSelectedKeyframePreset(CCameraKeyframeTrack& track, CCameraPreset preset)
{
    auto* selected = getSelectedKeyframe(track);
    if (!selected)
        return false;

    selected->preset = std::move(preset);
    return true;
}


#endif // _C_CAMERA_KEYFRAME_TRACK_HPP_
