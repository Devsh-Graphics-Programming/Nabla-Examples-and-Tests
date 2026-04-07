// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_

#include <string>

#include "CCameraScriptedRuntime.hpp"
#include "CCameraSequenceScript.hpp"
#include "ICamera.hpp"

namespace nbl::system
{

/**
* Build expanded scripted runtime data from a compiled camera-sequence segment.
*
* This keeps authored sequence semantics in shared camera helpers instead of re-encoding
* `Goal`, `TrackedTargetTransform`, `Baseline`, `GimbalStep`, and capture scheduling inside
* one consumer.
*/
struct CCameraSequenceScriptedSegmentBuildInfo
{
    //! Planar that should receive the compiled segment.
    uint32_t planarIx = 0u;
    //! Number of windows the consumer can actually route presentation actions to.
    size_t availableWindowCount = 1u;
    //! Whether secondary window presentation actions should be emitted.
    bool useWindow = false;
    //! Whether per-frame follow-lock checks should be generated for this segment.
    bool includeFollowTargetLock = false;
};

//! Append one compiled segment as expanded scripted runtime payloads.
inline bool appendCompiledSequenceSegmentToScriptedTimeline(
    CCameraScriptedTimeline& timeline,
    const uint64_t baseFrame,
    const core::CCameraSequenceCompiledSegment& compiledSegment,
    const CCameraSequenceScriptedSegmentBuildInfo& buildInfo,
    std::string* error = nullptr)
{
    std::vector<core::CCameraSequenceCompiledFramePolicy> framePolicies;
    if (!buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, buildInfo.includeFollowTargetLock))
    {
        if (error)
            *error = "Failed to build compiled frame policies.";
        return false;
    }

    appendScriptedSegmentLabelEvent(timeline, baseFrame, compiledSegment.name);
    appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, 0);
    appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
    if (!compiledSegment.presentations.empty())
    {
        appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[0].projection));
        appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded, compiledSegment.presentations[0].leftHanded ? 1 : 0);
    }
    if (compiledSegment.resetCamera)
        appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::ResetActiveCamera, 1);

    if (buildInfo.useWindow)
    {
        for (size_t windowIx = 1u; windowIx < std::min(compiledSegment.presentations.size(), buildInfo.availableWindowCount); ++windowIx)
        {
            appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, static_cast<int32_t>(windowIx));
            appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
            appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[windowIx].projection));
            appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetLeftHanded, compiledSegment.presentations[windowIx].leftHanded ? 1 : 0);
        }
        appendScriptedActionEvent(timeline, baseFrame, CCameraScriptedInputEvent::ActionData::Kind::SetActiveRenderWindow, 0);
    }

    for (const auto& policy : framePolicies)
    {
        core::CCameraPreset preset;
        if (!tryBuildKeyframeTrackPresetAtTime(compiledSegment.track, policy.sampleTime, preset))
        {
            if (error)
                *error = "Failed to sample compiled segment track.";
            return false;
        }
        appendScriptedGoalEvent(timeline, baseFrame + policy.frameOffset, makeGoalFromPreset(preset));

        if (compiledSegment.usesTrackedTargetTrack())
        {
            core::CCameraSequenceTrackedTargetPose trackedTargetPose;
            if (!tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, policy.sampleTime, trackedTargetPose))
            {
                if (error)
                    *error = "Failed to sample compiled tracked-target track.";
                return false;
            }

            core::ICamera::CGimbal gimbal({ .position = trackedTargetPose.position, .orientation = trackedTargetPose.orientation });
            appendScriptedTrackedTargetTransformEvent(timeline, baseFrame + policy.frameOffset, gimbal.operator()<hlsl::float64_t4x4>());
        }

        if (policy.baseline)
            appendScriptedBaselineCheck(timeline, baseFrame + policy.frameOffset);
        if (policy.continuityStep)
        {
            appendScriptedGimbalStepCheck(
                timeline,
                baseFrame + policy.frameOffset,
                compiledSegment.continuity.hasPosDeltaConstraint,
                compiledSegment.continuity.maxPosDelta,
                compiledSegment.continuity.minPosDelta,
                compiledSegment.continuity.hasEulerDeltaConstraint,
                compiledSegment.continuity.maxEulerDeltaDeg,
                compiledSegment.continuity.minEulerDeltaDeg);
        }
        if (policy.followTargetLock)
            appendScriptedFollowTargetLockCheck(timeline, baseFrame + policy.frameOffset);
        if (policy.capture)
            timeline.captureFrames.emplace_back(baseFrame + policy.frameOffset);
    }

    return true;
}

} // namespace nbl::system

#endif
