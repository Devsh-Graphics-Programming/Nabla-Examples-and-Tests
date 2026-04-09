#include "app/App.hpp"

#include <unordered_set>

bool App::tryCaptureGoal(ICamera* camera, CCameraGoal& out) const
{
	const auto capture = m_cameraGoalSolver.captureDetailed(camera);
	out = capture.goal;
	return capture.captured;
}

App::PresetUiAnalysis App::analyzePresetForUi(ICamera* camera, const CameraPreset& preset) const
{
	return nbl::ui::analyzePresetPresentation(m_cameraGoalSolver, camera, preset);
}

App::CaptureUiAnalysis App::analyzeCameraCaptureForUi(ICamera* camera) const
{
	return nbl::ui::analyzeCapturePresentation(m_cameraGoalSolver, camera);
}

CCameraGoalSolver::SCompatibilityResult App::analyzePresetCompatibility(ICamera* camera, const CameraPreset& preset) const
{
    return nbl::core::CCameraGoalAnalysisUtilities::analyzePresetApply(m_cameraGoalSolver, camera, preset).compatibility;
}

bool App::presetMatchesFilter(ICamera* camera, const CameraPreset& preset) const
{
	return analyzePresetForUi(camera, preset).matchesFilter(m_presetAuthoring.filterMode);
}

CCameraGoalSolver::SApplyResult App::applyPresetFromUi(ICamera* camera, const CameraPreset& preset)
{
	const auto result = nbl::core::applyPresetDetailed(m_cameraGoalSolver, camera, preset);
	if (result.succeeded())
		refreshFollowOffsetConfigsForCamera(camera);

	const auto presetUi = analyzePresetForUi(camera, preset);
	storeApplyStatusBanner(
		m_presetAuthoring.applyBanner,
        CCameraTextUtilities::describeApplyResult(result) + " | " + presetUi.compatibilityLabel,
		result.succeeded(),
		result.approximate());
	return result;
}

void App::storeApplyStatusBanner(ApplyStatusBanner& banner, std::string summary, const bool succeeded, const bool approximate)
{
	banner.summary = std::move(summary);
	banner.succeeded = succeeded;
	banner.approximate = approximate;
}

void App::clearApplyStatusBanner(ApplyStatusBanner& banner)
{
	banner.summary.clear();
	banner.succeeded = false;
	banner.approximate = false;
}

void App::storePlaybackApplySummary(const SCameraPresetApplySummary& summary)
{
	const auto& playbackAuthoring = m_playbackAuthoring;
	storeApplyStatusBanner(
		m_playbackAuthoring.applyBanner,
        nbl::ui::CCameraTextUtilities::describePresetApplySummary(
			summary,
			playbackAuthoring.affectsAll ? "Playback apply | no cameras available" : "Playback apply | no active camera"),
		summary.succeeded(),
		summary.approximate());
}

void App::appendVirtualEventLog(
	std::string_view source,
	std::string_view inputSource,
	const uint32_t planarIx,
	ICamera* camera,
	const CVirtualGimbalEvent* events,
	const uint32_t count)
{
	m_uiMetrics.virtualEventsThisFrame += count;
	const std::string sourceStr(source);
	const std::string inputSourceStr(inputSource);
	const std::string cameraName = camera ? std::string(camera->getIdentifier()) : std::string("None");
	for (uint32_t i = 0u; i < count; ++i)
	{
		const auto* eventName = CVirtualGimbalEvent::virtualEventToString(events[i].type).data();
		auto line = m_logFormatter->format(
			ILogger::ELL_INFO,
			"virtual frame=%llu src=%s input=%s cam=%s planar=%u event=%s mag=%.6f",
			static_cast<unsigned long long>(m_realFrameIx),
			sourceStr.c_str(),
			inputSourceStr.c_str(),
			cameraName.c_str(),
			planarIx,
			eventName,
			events[i].magnitude);
		m_eventLog.entries.push_back({
			m_realFrameIx,
			events[i].type,
			events[i].magnitude,
			sourceStr,
			inputSourceStr,
			cameraName,
			planarIx,
			std::move(line)
		});
	}

	while (m_eventLog.entries.size() > SCameraAppRuntimeDefaults::VirtualEventLogMax)
		m_eventLog.entries.pop_front();
}

SCameraPresetApplySummary App::applyPresetToTargets(const CameraPreset& preset)
{
	const auto& playbackAuthoring = m_playbackAuthoring;
	SCameraPresetApplySummary summary = {};
	if (!playbackAuthoring.affectsAll)
	{
		ICamera* activeCamera = getActiveCamera();
		summary = nbl::core::applyPresetToCameraRange(
			m_cameraGoalSolver,
			std::span<ICamera* const>(&activeCamera, activeCamera ? 1u : 0u),
			preset);
		if (summary.succeeded())
			refreshFollowOffsetConfigsForCamera(activeCamera);
		return summary;
	}

	std::vector<ICamera*> cameras;
	cameras.reserve(m_viewports.windowBindings.size());
	std::unordered_set<const ICamera*> visited;
	for (auto& binding : m_viewports.windowBindings)
	{
		auto& planar = m_planarProjections[binding.activePlanarIx];
		if (!planar)
			continue;

		auto* camera = planar->getCamera();
		if (!camera)
			continue;

		if (visited.insert(camera).second)
			cameras.push_back(camera);
	}

	summary = nbl::core::applyPresetToCameraRange(
		m_cameraGoalSolver,
		std::span<ICamera* const>(cameras.data(), cameras.size()),
		preset);
	if (summary.succeeded())
		refreshAllFollowOffsetConfigs();
	return summary;
}

bool App::tryBuildPlaybackPresetAtTime(const float time, CameraPreset& preset)
{
	return nbl::core::CCameraKeyframeTrackUtilities::tryBuildKeyframeTrackPresetAtTime(m_playbackAuthoring.keyframeTrack, time, preset);
}

bool App::applyPlaybackAtTime(const float time)
{
	CameraPreset preset;
	if (!tryBuildPlaybackPresetAtTime(time, preset))
	{
		clearApplyStatusBanner(m_playbackAuthoring.applyBanner);
		return false;
	}

	storePlaybackApplySummary(applyPresetToTargets(preset));
	return true;
}

void App::sortKeyframesByTime()
{
	nbl::core::CCameraKeyframeTrackUtilities::sortKeyframeTrackByTime(m_playbackAuthoring.keyframeTrack);
}

void App::clampPlaybackTimeToKeyframes()
{
	nbl::core::CCameraPlaybackTimelineUtilities::clampPlaybackCursorToTrack(m_playbackAuthoring.keyframeTrack, m_playbackAuthoring.playback);
}

int App::selectKeyframeNearestTime(const float time)
{
	return nbl::core::CCameraKeyframeTrackUtilities::selectKeyframeTrackNearestTime(m_playbackAuthoring.keyframeTrack, time);
}

void App::normalizeSelectedKeyframe()
{
	nbl::core::CCameraKeyframeTrackUtilities::normalizeSelectedKeyframeTrack(m_playbackAuthoring.keyframeTrack);
}

App::CameraKeyframe* App::getSelectedKeyframe()
{
	return nbl::core::CCameraKeyframeTrackUtilities::getSelectedKeyframe(m_playbackAuthoring.keyframeTrack);
}

const App::CameraKeyframe* App::getSelectedKeyframe() const
{
	return nbl::core::CCameraKeyframeTrackUtilities::getSelectedKeyframe(m_playbackAuthoring.keyframeTrack);
}

bool App::replaceSelectedKeyframeFromCamera(ICamera* camera)
{
	auto* selected = getSelectedKeyframe();
	if (!selected)
		return false;

	CameraPreset updatedPreset;
	const auto keyframeName = selected->preset.name.empty() ? std::string("Keyframe") : selected->preset.name;
	if (!nbl::core::tryCapturePreset(m_cameraGoalSolver, camera, keyframeName, updatedPreset))
		return false;

	return nbl::core::CCameraKeyframeTrackUtilities::replaceSelectedKeyframePreset(m_playbackAuthoring.keyframeTrack, std::move(updatedPreset));
}

void App::updatePlayback(const double dtSec)
{
	const auto advance = nbl::core::CCameraPlaybackTimelineUtilities::advancePlaybackCursor(m_playbackAuthoring.playback, m_playbackAuthoring.keyframeTrack, dtSec);
	if (!advance.hasTrack || !advance.changedTime)
		return;

	applyPlaybackAtTime(m_playbackAuthoring.playback.time);
}

