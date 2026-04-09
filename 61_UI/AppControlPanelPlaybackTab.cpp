#include "app/App.hpp"

#include <algorithm>

#include "app/AppControlPanelAuthoringUtilities.hpp"

void App::drawControlPanelPlaybackTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
    using checkbox_spec_t = nbl::ui::SCameraControlPanelCheckboxSpec;
    using slider_spec_t = nbl::ui::SCameraControlPanelSliderSpec;

    auto& playbackAuthoring = m_playbackAuthoring;

    if (!nbl::ui::beginControlPanelTabChild("PlaybackPanel", panelStyle))
    {
        nbl::ui::endControlPanelTabChild();
        return;
    }

    ImGui::PushItemWidth(-1.0f);
    auto* activeCamera = getActiveCamera();
    nbl::ui::drawSectionHeader("PlaybackHeader", "Playback", panelStyle.AccentColor, panelStyle);
    for (const auto& spec : {
        checkbox_spec_t{ .label = "Loop", .value = &playbackAuthoring.playback.loop, .hint = "Loop playback when it reaches the end" },
        checkbox_spec_t{ .label = "Override input", .value = &playbackAuthoring.playback.overrideInput, .hint = "Ignore manual input during playback" },
        checkbox_spec_t{ .label = "Affect all cameras", .value = &playbackAuthoring.affectsAll, .hint = "Apply playback to all cameras" }
    })
    {
        nbl::ui::drawCheckboxWithHint(spec);
    }
    nbl::ui::drawSliderFloatWithHint({
        .label = "Speed",
        .value = &playbackAuthoring.playback.speed,
        .minValue = SCameraAppAuthoringDefaults::PlaybackSpeedMin,
        .maxValue = SCameraAppAuthoringDefaults::PlaybackSpeedMax,
        .format = "%.2f",
        .hint = "Playback speed multiplier"
    });

    if (nbl::ui::drawActionButtonWithHint(playbackAuthoring.playback.playing ? "Pause" : "Play", "Start or pause playback"))
        playbackAuthoring.playback.playing = !playbackAuthoring.playback.playing;
    ImGui::SameLine();
    if (nbl::ui::drawActionButtonWithHint("Stop", "Stop playback and reset time"))
    {
        nbl::core::CCameraPlaybackTimelineUtilities::resetPlaybackCursor(playbackAuthoring.playback);
        applyPlaybackAtTime(playbackAuthoring.playback.time);
    }

    if (!playbackAuthoring.keyframeTrack.keyframes.empty())
    {
        const float duration = nbl::core::CCameraPlaybackTimelineUtilities::getPlaybackTrackDuration(playbackAuthoring.keyframeTrack);
        if (ImGui::SliderFloat("Time", &playbackAuthoring.playback.time, 0.f, duration, "%.3f"))
            applyPlaybackAtTime(playbackAuthoring.playback.time);
    }
    nbl::ui::drawApplyStatusBanner(
        playbackAuthoring.applyBanner.summary,
        playbackAuthoring.applyBanner.succeeded,
        playbackAuthoring.applyBanner.approximate,
        panelStyle);
    if (!playbackAuthoring.keyframeTrack.keyframes.empty())
    {
        CameraPreset playbackPreviewPreset;
        if (tryBuildPlaybackPresetAtTime(playbackAuthoring.playback.time, playbackPreviewPreset))
        {
            const auto playbackPreviewUi = analyzePresetForUi(activeCamera, playbackPreviewPreset);
            nbl::ui::drawPolicyStatus({
                .label = "Preview",
                .value = playbackPreviewUi.policyLabel,
                .active = playbackPreviewUi.canApply
            }, panelStyle);
        }
    }

    nbl::ui::drawSectionHeader("KeyframesHeader", "Keyframes", panelStyle.AccentColor, panelStyle);
    ImGui::InputFloat("New keyframe time", &playbackAuthoring.newKeyframeTime, SCameraAppAuthoringDefaults::KeyframeTimeStep, SCameraAppAuthoringDefaults::KeyframeTimeFastStep, "%.3f");
    nbl::ui::drawHoverHint("Time value for new keyframe");
    ImGui::SameLine();
    if (nbl::ui::drawActionButtonWithHint("Use playback time", "Set new keyframe time from current playback position"))
        playbackAuthoring.newKeyframeTime = playbackAuthoring.playback.time;
    const auto keyframeCaptureUi = analyzeCameraCaptureForUi(activeCamera);
    if (!keyframeCaptureUi.canCapture)
        ImGui::BeginDisabled();
    if (nbl::ui::drawActionButtonWithHint("Add keyframe", keyframeCaptureUi.canCapture ? "Add keyframe from current camera" : "Keyframe capture is blocked because there is no active camera or the current goal state is invalid"))
    {
        CameraKeyframe keyframe;
        const float authoredTime = std::max(0.f, playbackAuthoring.newKeyframeTime);
        keyframe.time = authoredTime;
        playbackAuthoring.newKeyframeTime = authoredTime;
        if (nbl::core::CCameraPresetFlowUtilities::tryCapturePreset(m_cameraGoalSolver, activeCamera, "Keyframe", keyframe.preset))
        {
            playbackAuthoring.keyframeTrack.keyframes.emplace_back(std::move(keyframe));
            sortKeyframesByTime();
            selectKeyframeNearestTime(authoredTime);
        }
    }
    if (!keyframeCaptureUi.canCapture)
        ImGui::EndDisabled();
    nbl::ui::drawPolicyStatus({
        .label = "Capture",
        .value = keyframeCaptureUi.policyLabel,
        .active = keyframeCaptureUi.canCapture
    }, panelStyle);
    ImGui::SameLine();
    if (nbl::ui::drawActionButtonWithHint("Clear keyframes", "Remove all keyframes"))
    {
        playbackAuthoring.keyframeTrack = {};
        nbl::core::CCameraPlaybackTimelineUtilities::resetPlaybackCursor(playbackAuthoring.playback);
        clearApplyStatusBanner(playbackAuthoring.applyBanner);
    }

    if (!playbackAuthoring.keyframeTrack.keyframes.empty())
    {
        normalizeSelectedKeyframe();
        if (ImGui::BeginChild("KeyframeList", ImVec2(0.0f, panelStyle.KeyframeListHeight), true))
        {
            for (size_t i = 0; i < playbackAuthoring.keyframeTrack.keyframes.size(); ++i)
            {
                const auto label = nbl::ui::buildKeyframeLabel(i, playbackAuthoring.keyframeTrack.keyframes[i]);
                if (ImGui::Selectable(label.c_str(), playbackAuthoring.keyframeTrack.selectedKeyframeIx == static_cast<int>(i)))
                    playbackAuthoring.keyframeTrack.selectedKeyframeIx = static_cast<int>(i);
            }
        }
        ImGui::EndChild();

        if (auto* selectedKeyframe = getSelectedKeyframe())
        {
            const auto keyframeUi = analyzePresetForUi(activeCamera, selectedKeyframe->preset);
            float selectedTime = selectedKeyframe->time;
            if (ImGui::InputFloat("Selected time", &selectedTime, SCameraAppAuthoringDefaults::KeyframeTimeStep, SCameraAppAuthoringDefaults::KeyframeTimeFastStep, "%.3f"))
            {
                selectedTime = std::max(0.f, selectedTime);
                selectedKeyframe->time = selectedTime;
                sortKeyframesByTime();
                selectKeyframeNearestTime(selectedTime);
                clampPlaybackTimeToKeyframes();
            }
            nbl::ui::drawHoverHint("Edit selected keyframe time");

            nbl::ui::drawGoalApplyPresentationSummary(keyframeUi, panelStyle);

            if (!keyframeUi.canApply)
                ImGui::BeginDisabled();
            if (nbl::ui::drawActionButtonWithHint("Apply selected", keyframeUi.canApply ? "Apply selected keyframe to the active camera" : "Apply is blocked because there is no active camera or the keyframe goal is invalid"))
                applyPresetFromUi(activeCamera, selectedKeyframe->preset);
            if (!keyframeUi.canApply)
                ImGui::EndDisabled();
            ImGui::SameLine();
            if (!keyframeCaptureUi.canCapture)
                ImGui::BeginDisabled();
            if (nbl::ui::drawActionButtonWithHint("Replace from camera", keyframeCaptureUi.canCapture ? "Overwrite selected keyframe from the current active camera" : "Replace is blocked because there is no active camera or the current goal state is invalid"))
                replaceSelectedKeyframeFromCamera(activeCamera);
            if (!keyframeCaptureUi.canCapture)
                ImGui::EndDisabled();
            ImGui::SameLine();
            if (nbl::ui::drawActionButtonWithHint("Jump to selected", "Set playback time to selected keyframe and preview it"))
            {
                playbackAuthoring.playback.time = selectedKeyframe->time;
                applyPlaybackAtTime(playbackAuthoring.playback.time);
            }
            ImGui::SameLine();
            if (nbl::ui::drawActionButtonWithHint("Remove selected", "Remove selected keyframe"))
            {
                playbackAuthoring.keyframeTrack.keyframes.erase(playbackAuthoring.keyframeTrack.keyframes.begin() + playbackAuthoring.keyframeTrack.selectedKeyframeIx);
                normalizeSelectedKeyframe();
                clampPlaybackTimeToKeyframes();
                if (playbackAuthoring.keyframeTrack.keyframes.empty())
                    clearApplyStatusBanner(playbackAuthoring.applyBanner);
            }
        }

        nbl::ui::drawSectionHeader("KeyframesStorageHeader", "Keyframe Storage", panelStyle.AccentColor, panelStyle);
        nbl::ui::inputTextString("Keyframe file", playbackAuthoring.keyframePath);
        if (nbl::ui::drawActionButtonWithHint("Save keyframes", "Save keyframes to JSON file"))
        {
            if (!saveKeyframesToFile(nbl::system::path(playbackAuthoring.keyframePath)))
                m_logger->log("Failed to save keyframes to \"%s\".", ILogger::ELL_ERROR, playbackAuthoring.keyframePath.c_str());
        }
        ImGui::SameLine();
        if (nbl::ui::drawActionButtonWithHint("Load keyframes", "Load keyframes from JSON file"))
        {
            if (!loadKeyframesFromFile(nbl::system::path(playbackAuthoring.keyframePath)))
                m_logger->log("Failed to load keyframes from \"%s\".", ILogger::ELL_ERROR, playbackAuthoring.keyframePath.c_str());
        }
    }

    ImGui::PopItemWidth();
    nbl::ui::endControlPanelTabChild();
}
