#ifndef _NBL_THIS_EXAMPLE_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_SEQUENCE_SCRIPTED_BUILDER_HPP_INCLUDED_

#include <string>

#include "camera/CCameraScriptedActionUtilities.hpp"
#include "nbl/ext/Cameras/CCameraScriptedRuntime.hpp"
#include "nbl/ext/Cameras/CCameraSequenceScript.hpp"
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
        nbl::ext::cameras::CCameraScriptedTimeline& timeline,
        std::vector<CCameraScriptedActionEvent>& actionEvents,
        const uint64_t baseFrame,
        const nbl::ext::cameras::CCameraSequenceCompiledSegment& compiledSegment,
        const CCameraSequenceScriptedSegmentBuildInfo& buildInfo,
        std::string* error = nullptr)
    {
        std::vector<nbl::ext::cameras::CCameraSequenceCompiledFramePolicy> framePolicies;
        if (!nbl::ext::cameras::CCameraSequenceScriptUtilities::buildCompiledSegmentFramePolicies(compiledSegment, framePolicies, buildInfo.includeFollowTargetLock))
        {
            if (error)
                *error = "Failed to build compiled frame policies.";
            return false;
        }

        nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedSegmentLabelEvent(timeline, baseFrame, compiledSegment.name);
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
            nbl::ext::cameras::CCameraPreset preset;
            if (!nbl::ext::cameras::CCameraKeyframeTrackUtilities::tryBuildKeyframeTrackPresetAtTime(compiledSegment.track, policy.sampleTime, preset))
            {
                if (error)
                    *error = "Failed to sample compiled segment track.";
                return false;
            }
            nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedGoalEvent(
                timeline,
                baseFrame + policy.frameOffset,
                nbl::ext::cameras::CCameraPresetUtilities::makeGoalFromPreset(preset));

            if (compiledSegment.usesTrackedTargetTrack())
            {
                nbl::ext::cameras::CCameraSequenceTrackedTargetPose trackedTargetPose;
                if (!nbl::ext::cameras::CCameraSequenceScriptUtilities::tryBuildSequenceTrackedTargetPoseAtTime(compiledSegment.trackedTargetTrack, policy.sampleTime, trackedTargetPose))
                {
                    if (error)
                        *error = "Failed to sample compiled tracked-target track.";
                    return false;
                }

                nbl::ext::cameras::ICamera::CGimbal gimbal({ .position = trackedTargetPose.position, .orientation = trackedTargetPose.orientation });
                nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedTrackedTargetTransformEvent(timeline, baseFrame + policy.frameOffset, gimbal.operator()<nbl::hlsl::float64_t4x4>());
            }

            if (policy.baseline)
                nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedBaselineCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.continuityStep)
            {
                nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedGimbalStepCheck(
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
                nbl::ext::cameras::CCameraScriptedRuntimeUtilities::appendScriptedFollowTargetLockCheck(timeline, baseFrame + policy.frameOffset);
            if (policy.capture)
                timeline.captureFrames.emplace_back(baseFrame + policy.frameOffset);
        }

        return true;
    }
};

} // namespace nbl::this_example

#endif
