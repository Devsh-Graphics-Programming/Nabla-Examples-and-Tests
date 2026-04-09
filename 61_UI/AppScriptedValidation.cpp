#include "app/App.hpp"
void App::updateScriptedFollowVisualState(const CCameraScriptedFrameEvents& scriptedFrameEvents)
{
	if (!scriptedFrameEvents.trackedTargetTransforms.empty())
	{
		setFollowTargetTransform(scriptedFrameEvents.trackedTargetTransforms.back().transform);
		applyFollowToConfiguredCameras(true);
		SCameraFollowVisualMetrics followMetrics = {};
		SActiveScriptedCameraContext runtimeContext = {};
		if (tryBuildActiveScriptedCameraContext(runtimeContext) && runtimeContext.followConfig)
		{
            followMetrics = nbl::system::CCameraFollowRegressionUtilities::buildFollowVisualMetrics(
                runtimeContext.viewport.camera,
                m_sceneInteraction.followTarget,
                runtimeContext.followConfig,
				runtimeContext.getProjectionContext());
		}
		m_scriptedInput.visualFollow = followMetrics;
		return;
	}

	applyFollowToConfiguredCameras();
	m_scriptedInput.visualFollow = {};
}

void App::runActiveFrameScriptedChecks(const SScriptedFrameInputState& scriptedFrame)
{
	if (!(m_scriptedInput.enabled && m_scriptedInput.checkRuntime.nextCheckIndex < m_scriptedInput.timeline.checks.size()))
		return;

	auto logFail = [&](const char* fmt, auto&&... args) -> void
	{
		m_scriptedInput.failed = true;
		m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
	};

	auto logPass = [&](const char* fmt, auto&&... args) -> void
	{
		if (!m_scriptedInput.log)
			return;
		m_logger->log(fmt, ILogger::ELL_INFO, std::forward<decltype(args)>(args)...);
	};

	SActiveScriptedCameraContext runtimeContext = {};
	const bool hasRuntimeContext = tryBuildActiveScriptedCameraContext(runtimeContext);

	const auto checkResult = nbl::system::evaluateScriptedChecksForFrame(
		m_scriptedInput.timeline.checks,
		m_scriptedInput.checkRuntime,
		{
			.frame = m_realFrameIx,
			.camera = hasRuntimeContext ? runtimeContext.viewport.camera : nullptr,
			.imguizmoVirtual = scriptedFrame.imguizmoVirtualEvents.data(),
			.imguizmoVirtualCount = static_cast<uint32_t>(scriptedFrame.imguizmoVirtualEvents.size()),
			.trackedTarget = &m_sceneInteraction.followTarget,
			.followConfig = hasRuntimeContext ? runtimeContext.followConfig : nullptr,
			.followProjectionContext = hasRuntimeContext ? runtimeContext.getProjectionContext() : nullptr,
			.goalSolver = &m_cameraGoalSolver
		});

	for (const auto& entry : checkResult.logs)
	{
		if (entry.failure)
			logFail("%s", entry.text.c_str());
		else
			logPass("%s", entry.text.c_str());
	}

	if (!m_scriptedInput.summaryReported && m_scriptedInput.checkRuntime.nextCheckIndex >= m_scriptedInput.timeline.checks.size())
	{
		m_scriptedInput.summaryReported = true;
		if (m_scriptedInput.failed)
			m_logger->log("[script] checks result: FAIL", ILogger::ELL_ERROR);
		else
			m_logger->log("[script] checks result: PASS", ILogger::ELL_INFO);
	}
}
