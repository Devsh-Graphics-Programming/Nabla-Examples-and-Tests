#include "app/App.hpp"

inline void logScriptedFramePayload(
	ILogger& logger,
	const uint64_t frameIx,
	const SScriptedFrameInputState& scriptedFrame)
{
	logger.log(
		"[script] frame %llu input kb=%zu mouse=%zu imguizmo=%zu goals=%zu target=%zu",
		ILogger::ELL_INFO,
		static_cast<unsigned long long>(frameIx),
		scriptedFrame.keyboard.size(),
		scriptedFrame.mouse.size(),
		scriptedFrame.frameEvents.imguizmo.size(),
		scriptedFrame.frameEvents.goals.size(),
		scriptedFrame.frameEvents.trackedTargetTransforms.size());
}

SAppFrameUpdateState App::buildFrameUpdateState()
{
	SAppFrameUpdateState frameState = {};
	prepareScriptedFrameState(frameState.scripted);
	prepareCameraAndUiInput(frameState.scripted, frameState.cameraInput, frameState.ui);
	return frameState;
}

void App::prepareScriptedFrameState(SAppFrameUpdateState::SPreparedScriptedFrame& outState)
{
	outState = {};
	outState.skipCameraInput = m_playbackAuthoring.playback.playing && m_playbackAuthoring.playback.overrideInput;
	dequeueScriptedFrameInput(outState.frame);
	applyScriptedFrameActions(outState.frame.frameEvents);
	ensureScriptedVisualPlanarState();
}

void App::prepareCameraAndUiInput(
	const SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState,
	SAppFrameUpdateState::SPreparedCapturedInput& outCameraInput,
	SAppFrameUpdateState::SUiRuntimeState& outUiState)
{
	prepareCapturedCameraInput(scriptedState, outCameraInput);
	prepareUiRuntimeState(outCameraInput, outUiState);
}

void App::prepareCapturedCameraInput(
	const SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState,
	SAppFrameUpdateState::SPreparedCapturedInput& outCameraInput)
{
	outCameraInput = {};
	outCameraInput.capturedEvents = captureUiInputEvents();
	if (m_scriptedInput.enabled && m_scriptedInput.exclusive)
		outCameraInput.capturedEvents.clear();

	appendScriptedInputEvents(scriptedState.frame, outCameraInput.capturedEvents);
	m_uiMetrics.inputEventsThisFrame = outCameraInput.capturedEvents.getEventCount();
	buildCameraInputEvents(
		outCameraInput.capturedEvents,
		outCameraInput.keyboardEvents,
		outCameraInput.mouseEvents);
}

void App::prepareUiRuntimeState(
	const SAppFrameUpdateState::SPreparedCapturedInput& cameraInput,
	SAppFrameUpdateState::SUiRuntimeState& outUiState)
{
	outUiState = {};
	outUiState.updateParams = buildUiUpdateParameters(cameraInput.capturedEvents);
}

void App::runCameraFramePasses(SAppFrameUpdateState& frameState)
{
	applyPreparedCameraInput(frameState.cameraInput, frameState.scripted.skipCameraInput);
	runPreparedScriptedFrame(frameState.scripted);
}

void App::applyPreparedCameraInput(
	const SAppFrameUpdateState::SPreparedCapturedInput& cameraInput,
	const bool skipCameraInput)
{
	applyActiveCameraInput(cameraInput.keyboardEvents, cameraInput.mouseEvents, skipCameraInput);
}

void App::runPreparedScriptedFrame(SAppFrameUpdateState::SPreparedScriptedFrame& scriptedState)
{
	if (m_scriptedInput.log && scriptedState.frame.hasRuntimePayload())
		logScriptedFramePayload(*m_logger, m_realFrameIx, scriptedState.frame);

	applyScriptedImguizmoInput(scriptedState.frame, scriptedState.skipCameraInput);
	applyScriptedGoals(scriptedState.frame.frameEvents, scriptedState.skipCameraInput);
	updateScriptedFollowVisualState(scriptedState.frame.frameEvents);
	runActiveFrameScriptedChecks(scriptedState.frame);
}

void App::updateUiFrame(const SAppFrameUpdateState::SUiRuntimeState& uiState)
{
	UpdateUiMetrics();
	m_ui.manager->update(uiState.updateParams);
}

void App::applyFrameRuntimeState(SAppFrameUpdateState& frameState)
{
	runCameraFramePasses(frameState);
	updateUiFrame(frameState.ui);
}

void App::update()
{
	updatePresentationTiming();
	updatePlayback(m_presentationTiming.frameDeltaSec);

	auto frameState = buildFrameUpdateState();
	applyFrameRuntimeState(frameState);
}

