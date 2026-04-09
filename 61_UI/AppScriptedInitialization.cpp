#include "app/App.hpp"

#include "app/AppCameraConfigUtilities.hpp"
#include "app/AppResourceUtilities.hpp"
#include "camera/CCameraScriptedRuntimePersistence.hpp"

void App::resetScriptedInputRuntimeState()
{
	m_scriptedInput.nextEventIndex = 0u;
	m_scriptedInput.checkRuntime = {};
	m_scriptedInput.nextCaptureIndex = 0u;
	m_scriptedInput.failed = false;
	m_scriptedInput.summaryReported = false;
}

void App::finalizeScriptedInputRuntimeState()
{
	nbl::system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(m_scriptedInput.timeline, m_cliRuntime.disableScreenshotsCli);
}

void App::applyParsedScriptedInput(
	nbl::system::CCameraScriptedInputParseResult parsed,
	std::optional<CCameraSequenceScript>& pendingScriptedSequence)
{
	pendingScriptedSequence.reset();
	m_scriptedInput.timeline.clear();
	resetScriptedInputRuntimeState();
	m_scriptedInput.exclusive = false;
	m_scriptedInput.hardFail = false;
	m_scriptedInput.visualDebug = false;
	m_scriptedInput.visualTargetFps = 0.f;
	m_scriptedInput.visualCameraHoldSeconds = 0.f;
	m_scriptedInput.visualPlanar = {};
	m_scriptedInput.visualFollow = {};
	m_scriptedInput.scriptedMouseButtons = {};
	m_scriptedInput.framePacer = {};
	m_scriptedInput.capturePrefix = std::string(SCameraAppScriptedVisualDefaults::DefaultCapturePrefix);
	m_scriptedInput.captureOutputDir = localOutputCWD;

	m_scriptedInput.enabled = parsed.enabled;
	if (parsed.hasLog)
		m_scriptedInput.log = parsed.log || m_scriptedInput.log;
	m_scriptedInput.hardFail = parsed.hardFail;
	m_scriptedInput.visualDebug = parsed.visualDebug;
	m_scriptedInput.visualTargetFps = parsed.visualTargetFps;
	m_scriptedInput.visualCameraHoldSeconds = parsed.visualCameraHoldSeconds;
	if (m_cliRuntime.scriptVisualDebugCli)
		m_scriptedInput.visualDebug = true;
	if (m_scriptedInput.visualDebug)
	{
		if (m_scriptedInput.visualTargetFps <= 0.f)
			m_scriptedInput.visualTargetFps = SCameraAppScriptedVisualDefaults::TargetFps;
		if (m_scriptedInput.visualCameraHoldSeconds <= 0.f)
			m_scriptedInput.visualCameraHoldSeconds = SCameraAppScriptedVisualDefaults::HoldSeconds;
	}

	if (parsed.hasEnableActiveCameraMovement)
		m_viewports.enableActiveCameraMovement = parsed.enableActiveCameraMovement;
	else if (m_scriptedInput.enabled)
		m_viewports.enableActiveCameraMovement = true;

	m_scriptedInput.exclusive = parsed.exclusive;
	m_scriptedInput.capturePrefix = parsed.capturePrefix.empty() ? std::string(SCameraAppScriptedVisualDefaults::DefaultCapturePrefix) : parsed.capturePrefix;

	if (parsed.cameraControls.hasKeyboardScale)
		m_cameraControls.keyboardScale = parsed.cameraControls.keyboardScale;
	if (parsed.cameraControls.hasMouseMoveScale)
		m_cameraControls.mouseMoveScale = parsed.cameraControls.mouseMoveScale;
	if (parsed.cameraControls.hasMouseScrollScale)
		m_cameraControls.mouseScrollScale = parsed.cameraControls.mouseScrollScale;
	if (parsed.cameraControls.hasTranslationScale)
		m_cameraControls.translationScale = parsed.cameraControls.translationScale;
	if (parsed.cameraControls.hasRotationScale)
		m_cameraControls.rotationScale = parsed.cameraControls.rotationScale;

	for (const auto& warning : parsed.warnings)
		m_logger->log("%s", ILogger::ELL_WARNING, warning.c_str());

	pendingScriptedSequence = std::move(parsed.sequence);
	m_scriptedInput.timeline = std::move(parsed.timeline);
	finalizeScriptedInputRuntimeState();
}

bool App::tryLoadConfiguredScriptedInput(
	const argparse::ArgumentParser& program,
	const nbl::system::SCameraConfigCollections& cameraCollections,
	std::optional<CCameraSequenceScript>& outPendingScriptedSequence)
{
	outPendingScriptedSequence = std::nullopt;

	const auto tryApplyScriptedText = [&](const std::string_view scriptedText) -> bool
	{
		if (scriptedText.empty())
			return true;

		nbl::system::CCameraScriptedInputParseResult parsed = {};
		std::string scriptedInputParseError;
		if (!nbl::system::readCameraScriptedInput(scriptedText, parsed, &scriptedInputParseError))
			return logFail("Camera sequence script parse failed: %s", scriptedInputParseError.c_str());

		applyParsedScriptedInput(std::move(parsed), outPendingScriptedSequence);
		return true;
	};

	if (program.is_used("--script"))
	{
		nbl::system::SCameraScriptTextLoadResult scriptResource = {};
		std::string scriptedInputLoadError;
		if (!nbl::system::tryLoadCameraScriptText(
				getCameraAppResourceContext(),
				nbl::system::path(program.get<std::string>("--script")),
				scriptResource,
				&scriptedInputLoadError))
		{
			return logFail("Camera sequence script parse failed: %s", scriptedInputLoadError.c_str());
		}

		return tryApplyScriptedText(scriptResource.text);
	}

	std::string embeddedScriptedInput = {};
	if (!nbl::system::tryGetEmbeddedCameraScriptedInputText(cameraCollections, embeddedScriptedInput))
		return true;

	return tryApplyScriptedText(embeddedScriptedInput);
}

std::optional<uint32_t> App::resolveSequenceSegmentPlanarIx(const CCameraSequenceSegment& segment) const
{
	std::optional<uint32_t> match;
	for (uint32_t planarIx = 0u; planarIx < m_planarProjections.size(); ++planarIx)
	{
		auto* camera = m_planarProjections[planarIx]->getCamera();
		if (!camera)
			continue;

		const bool kindMatch = segment.cameraKind == ICamera::CameraKind::Unknown || camera->getKind() == segment.cameraKind;
		const bool identifierMatch = segment.cameraIdentifier.empty() || camera->getIdentifier() == segment.cameraIdentifier;
		if (!(kindMatch && identifierMatch))
			continue;

		if (match.has_value())
			return std::nullopt;
		match = planarIx;
	}

	return match;
}

bool App::expandPendingScriptedSequence(const CCameraSequenceScript& sequence)
{
	CCameraScriptedTimeline timeline;
	resetScriptedInputRuntimeState();

	const bool useWindowMode = nbl::core::CCameraSequenceScriptUtilities::sequenceScriptUsesMultiplePresentations(sequence);
	nbl::system::CCameraScriptedRuntimeUtilities::appendScriptedActionEvent(
		timeline,
		0u,
		CCameraScriptedInputEvent::ActionData::Kind::SetUseWindow,
		useWindowMode ? 1 : 0);

	const CCameraSequenceTrackedTargetPose referenceTrackedTargetPose = {
		.position = getDefaultFollowTargetPosition(),
		.orientation = getDefaultFollowTargetOrientation()
	};

	uint64_t frameCursor = 0u;
	for (const auto& segment : sequence.segments)
	{
		const auto planarIx = resolveSequenceSegmentPlanarIx(segment);
		if (!planarIx.has_value())
		{
            const auto kindLabel = segment.cameraKind != ICamera::CameraKind::Unknown ? std::string(CCameraTextUtilities::getCameraTypeLabel(segment.cameraKind)) : std::string("Unknown");
			return logFail(
				"Sequence segment \"%s\" has ambiguous or missing camera match for kind \"%s\" identifier \"%s\".",
				segment.name.c_str(),
				kindLabel.c_str(),
				segment.cameraIdentifier.c_str());
		}

		const bool useTrackedTargetFollow =
			nbl::core::CCameraSequenceScriptUtilities::sequenceSegmentUsesTrackedTargetTrack(segment) &&
			planarIx.value() < m_sceneInteraction.planarFollowConfigs.size() &&
			m_sceneInteraction.planarFollowConfigs[planarIx.value()].enabled &&
			m_sceneInteraction.planarFollowConfigs[planarIx.value()].mode != ECameraFollowMode::Disabled;

		nbl::core::CCameraSequenceCompiledSegment compiledSegment;
		std::string trackError;
		if (!nbl::core::CCameraSequenceScriptUtilities::compileSequenceSegmentFromReference(
				sequence,
				segment,
				m_presetAuthoring.initialPlanarPresets[planarIx.value()],
				referenceTrackedTargetPose,
				compiledSegment,
				&trackError))
		{
			return logFail("Sequence segment \"%s\" failed to compile: %s", segment.name.c_str(), trackError.c_str());
		}

		if (compiledSegment.presentations.size() > m_viewports.windowBindings.size())
		{
			m_logger->log(
				"Sequence segment \"%s\" requests %zu presentations, only %zu windows are available. Extra presentations will be ignored.",
				ILogger::ELL_WARNING,
				segment.name.c_str(),
				compiledSegment.presentations.size(),
				m_viewports.windowBindings.size());
		}

		std::string buildError;
		if (!nbl::system::CCameraSequenceScriptedBuilderUtilities::appendCompiledSequenceSegmentToScriptedTimeline(
				timeline,
				frameCursor,
				compiledSegment,
				{
					.planarIx = planarIx.value(),
					.availableWindowCount = m_viewports.windowBindings.size(),
					.useWindow = useWindowMode,
					.includeFollowTargetLock = useTrackedTargetFollow
				},
				&buildError))
		{
			return logFail(
				"Sequence segment \"%s\" failed to build scripted runtime data: %s",
				segment.name.c_str(),
				buildError.c_str());
		}

		frameCursor += compiledSegment.durationFrames;
	}

	nbl::system::CCameraScriptedRuntimeUtilities::finalizeScriptedTimeline(timeline, m_cliRuntime.disableScreenshotsCli);
	m_scriptedInput.timeline = std::move(timeline);
	return true;
}

