#ifndef _NBL_THIS_EXAMPLE_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_INCLUDED_

#include <string>

#include "camera/CCameraScriptedActionUtilities.hpp"
#include "CCameraScriptedRuntime.hpp"
#include "CCameraSequenceScript.hpp"
#include "nbl/ext/Cameras/ICamera.hpp"

namespace nbl::this_example
{

struct CCameraSequenceScriptedSegmentBuildInfo
{
    uint32_t planarIx = 0u;
    size_t availableWindowCount = 1u;
    bool useWindow = false;
    bool includeFollowTargetLock = false;
};

struct CCameraSequenceScriptedBuilderUtilities final
{
    static inline bool appendCompiledSequenceSegmentToScriptedTimeline(
        CCameraScriptedTimeline& timeline,
        std::vector<CCameraScriptedActionEvent>& actionEvents,
        const uint64_t baseFrame,
        const CCameraSequenceCompiledSegment& compiledSegment,
        const CCameraSequenceScriptedSegmentBuildInfo& buildInfo,
        std::string* error = nullptr)
    {
        std::vector<CCameraSequenceCompiledFramePolicy> framePolicies;
        if (!CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, buildInfo.includeFollowTargetLock))
        {
            if (error)
                *error = "Failed to build compiled frame policies.";
            return false;
        }

        CCameraScriptedRuntimeUtilities::appendScriptedSegmentLabelEvent(timeline, baseFrame, compiledSegment.name);
        CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetActiveRenderWindow, 0);
        CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
        if (!compiledSegment.presentations.empty())
        {
            CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[0].projection));
            CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetLeftHanded, compiledSegment.presentations[0].leftHanded ? 1 : 0);
        }
        if (compiledSegment.resetCamera)
            CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::ResetActiveCamera, 1);

        if (buildInfo.useWindow)
        {
            for (size_t windowIx = 1u; windowIx < std::min(compiledSegment.presentations.size(), buildInfo.availableWindowCount); ++windowIx)
            {
                CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetActiveRenderWindow, static_cast<int32_t>(windowIx));
                CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetActivePlanar, static_cast<int32_t>(buildInfo.planarIx));
                CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetProjectionType, static_cast<int32_t>(compiledSegment.presentations[windowIx].projection));
                CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetLeftHanded, compiledSegment.presentations[windowIx].leftHanded ? 1 : 0);
            }
            CCameraScriptedActionUtilities::appendActionEvent(actionEvents, baseFrame, ECameraScriptedActionCode::SetActiveRenderWindow, 0);
        }

        for (const auto& policy : framePolicies)
        {
            CCameraPreset preset;
            if (!CCameraKeyframeTrackUtilities::tryBuildKeyframeTrackPresetAtTime(compiledSegment.track, policy.sampleTime, preset))
            {
                if (error)
                    *error = "Failed to sample compiled segment track.";
                return false;
            }
            CCameraScriptedRuntimeUtilities::appendScriptedGoalEvent(
                timeline,
                baseFrame + policy.frameOffset,
                CCameraPresetUtilities::makeGoalFromPreset(preset));

            if (compiledSegment.usesTrackedTargetTrack())
            {
                CCameraSequenceTrackedTargetPose trackedTargetPose;
                if (!CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, policy.sampleTime, trackedTargetPose))
                {
                    if (error)
                        *error = "Failed to sample compiled tracked-target track.";
                    return false;
                }

                CCameraScriptedRuntimeUtilities::appendScriptedTrackedTargetTransformEvent(
                    timeline,
                    baseFrame + policy.frameOffset,
                    nbl::ext::cameras::CCameraMathUtilities::composeTransformMatrix(trackedTargetPose.position, trackedTargetPose.orientation));
            }

            if (policy.baseline)
                CCameraScriptedRuntimeUtilities::appendScriptedBaselineCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.continuityStep)
            {
                CCameraScriptedRuntimeUtilities::appendScriptedGimbalStepCheck(
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
                CCameraScriptedRuntimeUtilities::appendScriptedFollowTargetLockCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.capture)
                timeline.captureFrames.emplace_back(baseFrame + policy.frameOffset);
        }

        return true;
    }
};

} // namespace nbl::this_example

#endif
