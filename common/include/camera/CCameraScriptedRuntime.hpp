// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPTED_RUNTIME_HPP_
#define _C_CAMERA_SCRIPTED_RUNTIME_HPP_

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "CCameraGoal.hpp"
#include "IGimbal.hpp"
#include "nbl/ui/SInputEvent.h"

namespace nbl::hlsl
{

/**
* Shared scripted runtime contract used by camera-sequence consumers.
*
* The compact authored sequence remains camera-domain. A concrete runtime may still expand it
* into low-level per-frame events and checks, but those expanded payloads live in this shared
* header rather than inside one example.
*/
struct CCameraScriptedInputEvent
{
    enum class Type : uint8_t
    {
        Keyboard,
        Mouse,
        Imguizmo,
        Action,
        Goal,
        TrackedTargetTransform,
        SegmentLabel
    };

    struct KeyboardData
    {
        ui::E_KEY_CODE key = ui::EKC_NONE;
        ui::SKeyboardEvent::E_KEY_ACTION action = ui::SKeyboardEvent::ECA_UNITIALIZED;
    };

    struct MouseData
    {
        ui::SMouseEvent::E_EVENT_TYPE type = ui::SMouseEvent::EET_UNITIALIZED;
        ui::E_MOUSE_BUTTON button = ui::EMB_LEFT_BUTTON;
        ui::SMouseEvent::SClickEvent::E_ACTION action = ui::SMouseEvent::SClickEvent::EA_UNITIALIZED;
        int16_t x = 0;
        int16_t y = 0;
        int16_t dx = 0;
        int16_t dy = 0;
        int16_t v = 0;
        int16_t h = 0;
    };

    struct ActionData
    {
        enum class Kind : uint8_t
        {
            SetActiveRenderWindow,
            SetActivePlanar,
            SetProjectionType,
            SetProjectionIndex,
            SetUseWindow,
            SetLeftHanded,
            ResetActiveCamera
        };

        Kind kind = Kind::SetActiveRenderWindow;
        int32_t value = 0;
    };

    struct GoalData
    {
        CCameraGoal goal = {};
        bool requireExact = true;
    };

    struct TrackedTargetTransformData
    {
        float64_t4x4 transform = float64_t4x4(1.0);
    };

    struct SegmentLabelData
    {
        std::string label;
    };

    uint64_t frame = 0;
    Type type = Type::Keyboard;
    KeyboardData keyboard;
    MouseData mouse;
    float32_t4x4 imguizmo = float32_t4x4(1.f);
    ActionData action;
    GoalData goal;
    TrackedTargetTransformData trackedTargetTransform;
    SegmentLabelData segmentLabel;
};

struct CCameraScriptedInputCheck
{
    enum class Kind : uint8_t
    {
        Baseline,
        ImguizmoVirtual,
        GimbalNear,
        GimbalDelta,
        GimbalStep,
        FollowTargetLock
    };

    struct ExpectedVirtualEvent
    {
        CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::None;
        float64_t magnitude = 0.0;
    };

    uint64_t frame = 0;
    Kind kind = Kind::Baseline;
    float tolerance = 1e-3f;
    std::vector<ExpectedVirtualEvent> expectedVirtualEvents;

    float32_t3 expectedPos = float32_t3(0.f);
    float32_t3 expectedEulerDeg = float32_t3(0.f);
    bool hasExpectedPos = false;
    bool hasExpectedEuler = false;
    float posTolerance = 0.05f;
    float eulerToleranceDeg = 1.0f;
    float minPosDelta = 0.0f;
    float minEulerDeltaDeg = 0.0f;
    bool hasPosDeltaConstraint = false;
    bool hasEulerDeltaConstraint = false;
};

//! Fully expanded scripted timeline shared between authored parsers and runtime consumers.
struct CCameraScriptedTimeline
{
    std::vector<CCameraScriptedInputEvent> events;
    std::vector<CCameraScriptedInputCheck> checks;
    std::vector<uint64_t> captureFrames;

    inline void clear()
    {
        events.clear();
        checks.clear();
        captureFrames.clear();
    }

    inline bool empty() const
    {
        return events.empty() && checks.empty() && captureFrames.empty();
    }
};

inline void finalizeScriptedTimeline(
    std::vector<CCameraScriptedInputEvent>& events,
    std::vector<CCameraScriptedInputCheck>& checks,
    std::vector<uint64_t>& captureFrames,
    const bool disableCaptureFrames = false)
{
    std::stable_sort(events.begin(), events.end(),
        [](const CCameraScriptedInputEvent& a, const CCameraScriptedInputEvent& b) { return a.frame < b.frame; });
    std::stable_sort(checks.begin(), checks.end(),
        [](const CCameraScriptedInputCheck& a, const CCameraScriptedInputCheck& b) { return a.frame < b.frame; });
    if (!captureFrames.empty())
    {
        std::sort(captureFrames.begin(), captureFrames.end());
        captureFrames.erase(std::unique(captureFrames.begin(), captureFrames.end()), captureFrames.end());
    }
    if (disableCaptureFrames)
        captureFrames.clear();
}

inline void finalizeScriptedTimeline(CCameraScriptedTimeline& timeline, const bool disableCaptureFrames = false)
{
    finalizeScriptedTimeline(timeline.events, timeline.checks, timeline.captureFrames, disableCaptureFrames);
}

inline void appendScriptedActionEvent(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    const CCameraScriptedInputEvent::ActionData::Kind kind,
    const int32_t value)
{
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Action;
    entry.action.kind = kind;
    entry.action.value = value;
    timeline.events.emplace_back(std::move(entry));
}

inline void appendScriptedGoalEvent(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    const CCameraGoal& goal,
    const bool requireExact = true)
{
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::Goal;
    entry.goal.goal = goal;
    entry.goal.requireExact = requireExact;
    timeline.events.emplace_back(std::move(entry));
}

inline void appendScriptedTrackedTargetTransformEvent(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    const float64_t4x4& transform)
{
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::TrackedTargetTransform;
    entry.trackedTargetTransform.transform = transform;
    timeline.events.emplace_back(std::move(entry));
}

inline void appendScriptedSegmentLabelEvent(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    std::string label)
{
    CCameraScriptedInputEvent entry;
    entry.frame = frame;
    entry.type = CCameraScriptedInputEvent::Type::SegmentLabel;
    entry.segmentLabel.label = std::move(label);
    timeline.events.emplace_back(std::move(entry));
}

inline void appendScriptedBaselineCheck(CCameraScriptedTimeline& timeline, const uint64_t frame)
{
    CCameraScriptedInputCheck entry;
    entry.frame = frame;
    entry.kind = CCameraScriptedInputCheck::Kind::Baseline;
    timeline.checks.emplace_back(std::move(entry));
}

inline void appendScriptedGimbalStepCheck(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    const bool hasPosDeltaConstraint,
    const float posTolerance,
    const float minPosDelta,
    const bool hasEulerDeltaConstraint,
    const float eulerToleranceDeg,
    const float minEulerDeltaDeg)
{
    CCameraScriptedInputCheck entry;
    entry.frame = frame;
    entry.kind = CCameraScriptedInputCheck::Kind::GimbalStep;
    if (hasPosDeltaConstraint)
    {
        entry.hasPosDeltaConstraint = true;
        entry.posTolerance = posTolerance;
        entry.minPosDelta = minPosDelta;
    }
    if (hasEulerDeltaConstraint)
    {
        entry.hasEulerDeltaConstraint = true;
        entry.eulerToleranceDeg = eulerToleranceDeg;
        entry.minEulerDeltaDeg = minEulerDeltaDeg;
    }
    timeline.checks.emplace_back(std::move(entry));
}

inline void appendScriptedFollowTargetLockCheck(
    CCameraScriptedTimeline& timeline,
    const uint64_t frame,
    const float toleranceDeg = 1.0f,
    const float screenToleranceNdc = 0.03f)
{
    CCameraScriptedInputCheck entry;
    entry.frame = frame;
    entry.kind = CCameraScriptedInputCheck::Kind::FollowTargetLock;
    entry.eulerToleranceDeg = toleranceDeg;
    entry.posTolerance = screenToleranceNdc;
    timeline.checks.emplace_back(std::move(entry));
}

/**
* Per-frame scripted runtime batch already partitioned by payload kind.
*
* Consumers can dequeue authored events for one frame and then adapt only the buckets they care
* about, without repeatedly switching on `CCameraScriptedInputEvent::Type` in local glue.
*/
struct CCameraScriptedFrameEvents
{
    std::vector<CCameraScriptedInputEvent::KeyboardData> keyboard;
    std::vector<CCameraScriptedInputEvent::MouseData> mouse;
    std::vector<float32_t4x4> imguizmo;
    std::vector<CCameraScriptedInputEvent::ActionData> actions;
    std::vector<CCameraScriptedInputEvent::GoalData> goals;
    std::vector<CCameraScriptedInputEvent::TrackedTargetTransformData> trackedTargetTransforms;
    std::vector<std::string> segmentLabels;

    inline void clear()
    {
        keyboard.clear();
        mouse.clear();
        imguizmo.clear();
        actions.clear();
        goals.clear();
        trackedTargetTransforms.clear();
        segmentLabels.clear();
    }

    inline bool empty() const
    {
        return keyboard.empty() && mouse.empty() && imguizmo.empty() && actions.empty() &&
            goals.empty() && trackedTargetTransforms.empty() && segmentLabels.empty();
    }
};

//! Dequeue all authored scripted events scheduled for one frame.
inline void dequeueScriptedFrameEvents(
    const std::vector<CCameraScriptedInputEvent>& events,
    size_t& nextEventIndex,
    const uint64_t frame,
    CCameraScriptedFrameEvents& out)
{
    out.clear();
    while (nextEventIndex < events.size() && events[nextEventIndex].frame == frame)
    {
        const auto& ev = events[nextEventIndex];
        switch (ev.type)
        {
            case CCameraScriptedInputEvent::Type::Keyboard:
                out.keyboard.emplace_back(ev.keyboard);
                break;
            case CCameraScriptedInputEvent::Type::Mouse:
                out.mouse.emplace_back(ev.mouse);
                break;
            case CCameraScriptedInputEvent::Type::Imguizmo:
                out.imguizmo.emplace_back(ev.imguizmo);
                break;
            case CCameraScriptedInputEvent::Type::Action:
                out.actions.emplace_back(ev.action);
                break;
            case CCameraScriptedInputEvent::Type::Goal:
                out.goals.emplace_back(ev.goal);
                break;
            case CCameraScriptedInputEvent::Type::TrackedTargetTransform:
                out.trackedTargetTransforms.emplace_back(ev.trackedTargetTransform);
                break;
            case CCameraScriptedInputEvent::Type::SegmentLabel:
                out.segmentLabels.emplace_back(ev.segmentLabel.label);
                break;
        }

        ++nextEventIndex;
    }
}

} // namespace nbl::hlsl

#endif
