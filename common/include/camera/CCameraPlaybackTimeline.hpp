// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PLAYBACK_TIMELINE_HPP_
#define _C_CAMERA_PLAYBACK_TIMELINE_HPP_

#include "CCameraKeyframeTrack.hpp"

namespace nbl::hlsl
{

//! Shared playback cursor state for camera keyframe tracks.
struct CCameraPlaybackCursor
{
    bool playing = false;
    bool loop = true;
    float speed = 1.f;
    float time = 0.f;
};

//! Outcome of advancing a playback cursor against a keyframe track.
struct SCameraPlaybackAdvanceResult
{
    bool hasTrack = false;
    bool changedTime = false;
    bool wrapped = false;
    bool reachedEnd = false;
    bool stopped = false;
    float duration = 0.f;
    float time = 0.f;
};

inline float getPlaybackTrackDuration(const CCameraKeyframeTrack& track)
{
    if (track.keyframes.empty())
        return 0.f;

    return track.keyframes.back().time;
}

inline void resetPlaybackCursor(CCameraPlaybackCursor& cursor, const float time = 0.f)
{
    cursor.playing = false;
    cursor.time = std::max(0.f, time);
}

inline void clampPlaybackCursorToTrack(const CCameraKeyframeTrack& track, CCameraPlaybackCursor& cursor)
{
    clampTrackTimeToKeyframes(track, cursor.time);
}

inline SCameraPlaybackAdvanceResult advancePlaybackCursor(CCameraPlaybackCursor& cursor, const CCameraKeyframeTrack& track, const double dtSec)
{
    SCameraPlaybackAdvanceResult result;
    result.hasTrack = !track.keyframes.empty();
    result.duration = getPlaybackTrackDuration(track);
    result.time = cursor.time;

    if (!result.hasTrack || !cursor.playing)
        return result;

    const auto previousTime = cursor.time;
    cursor.time += static_cast<float>(dtSec * cursor.speed);
    result.changedTime = cursor.time != previousTime;
    result.time = cursor.time;

    if (result.duration <= 0.f)
        return result;

    if (cursor.loop)
    {
        while (cursor.time > result.duration)
        {
            cursor.time -= result.duration;
            result.wrapped = true;
        }
    }
    else if (cursor.time > result.duration)
    {
        cursor.time = result.duration;
        cursor.playing = false;
        result.reachedEnd = true;
        result.stopped = true;
    }

    result.time = cursor.time;
    return result;
}

} // namespace nbl::hlsl

#endif // _C_CAMERA_PLAYBACK_TIMELINE_HPP_
