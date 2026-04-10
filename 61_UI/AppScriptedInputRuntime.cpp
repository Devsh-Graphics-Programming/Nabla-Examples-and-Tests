#include "app/App.hpp"

void App::logScriptedCameraPose(const char* label, ICamera* camera) const
{
	if (!(m_scriptedInput.log && camera))
		return;

	const auto& gimbal = camera->getGimbal();
	const auto position = gimbal.getPosition();
	const auto euler = hlsl::CCameraMathUtilities::getCameraOrientationEulerDegrees(gimbal.getOrientation());
	m_logger->log(
		"[script] %s gimbal pos=(%.3f, %.3f, %.3f) euler_deg=(%.3f, %.3f, %.3f)",
		ILogger::ELL_INFO,
		label,
		position.x,
		position.y,
		position.z,
		euler.x,
		euler.y,
		euler.z);
}

void App::dequeueScriptedFrameInput(SScriptedFrameInputState& outFrame)
{
	outFrame = {};

	if (m_scriptedInput.enabled && m_scriptedInput.nextEventIndex < m_scriptedInput.timeline.events.size())
	{
		nbl::system::CCameraScriptedFrameEventUtilities::dequeueScriptedFrameEvents(
			m_scriptedInput.timeline.events,
			m_scriptedInput.nextEventIndex,
			m_realFrameIx,
			outFrame.frameEvents);
	}
	if (m_scriptedInput.enabled && m_scriptedInput.nextActionIndex < m_scriptedInput.actionEvents.size())
	{
		nbl::this_example::CCameraScriptedActionUtilities::dequeueFrameActions(
			m_scriptedInput.actionEvents,
			m_scriptedInput.nextActionIndex,
			m_realFrameIx,
			outFrame.actions);
	}

	nbl::ui::CCameraScriptedUiInputUtilities::appendScriptedUiInputEvents(
		m_nextPresentationTimestamp,
		m_window.get(),
		outFrame.frameEvents.keyboard,
		outFrame.frameEvents.mouse,
		outFrame.keyboard,
		outFrame.mouse);

	if (!outFrame.frameEvents.segmentLabels.empty())
		m_scriptedInput.visualPlanar.segmentLabel = outFrame.frameEvents.segmentLabels.back();
}

void App::applyScriptedFrameActions(std::span<const nbl::this_example::CCameraScriptedActionEvent> scriptedActions)
{
	if (!(m_scriptedInput.enabled && !scriptedActions.empty()))
		return;

	auto applyAction = [&](const nbl::this_example::CCameraScriptedActionEvent& action) -> void
	{
		switch (static_cast<nbl::this_example::ECameraScriptedActionCode>(action.code))
		{
			case nbl::this_example::ECameraScriptedActionCode::SetActiveRenderWindow:
			{
				if (action.value < 0 || static_cast<size_t>(action.value) >= m_viewports.windowBindings.size())
				{
					m_logger->log("[script][warn] action set_active_render_window out of range: %d", ILogger::ELL_WARNING, action.value);
					return;
				}
				m_viewports.activeRenderWindowIx = static_cast<uint32_t>(action.value);
			} break;

			case nbl::this_example::ECameraScriptedActionCode::SetActivePlanar:
			{
				if (action.value < 0)
				{
					m_logger->log("[script][warn] action set_active_planar out of range: %d", ILogger::ELL_WARNING, action.value);
					return;
				}

				auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
				if (!nbl::ui::trySelectBindingPlanar(
						getPlanarProjectionSpan(),
						binding,
						static_cast<uint32_t>(action.value)))
				{
					m_logger->log("[script][warn] action set_active_planar out of range: %d", ILogger::ELL_WARNING, action.value);
					return;
				}
				m_scriptedInput.visualPlanar.valid = true;
				m_scriptedInput.visualPlanar.planarIx = binding.activePlanarIx;
				m_scriptedInput.visualPlanar.startFrame = m_realFrameIx;
			} break;

			case nbl::this_example::ECameraScriptedActionCode::SetProjectionType:
			{
				auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
				const auto type = static_cast<IPlanarProjection::CProjection::ProjectionType>(action.value);
				if (!nbl::ui::trySelectBindingProjectionType(
						getPlanarProjectionSpan(),
						binding,
						type))
				{
					m_logger->log("[script][warn] action set_projection_type invalid value: %d", ILogger::ELL_WARNING, action.value);
				}
			} break;

			case nbl::this_example::ECameraScriptedActionCode::SetProjectionIndex:
			{
				auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
				auto& projections = m_planarProjections[binding.activePlanarIx]->getPlanarProjections();
				if (action.value < 0 || static_cast<size_t>(action.value) >= projections.size())
				{
					m_logger->log("[script][warn] action set_projection_index out of range: %d", ILogger::ELL_WARNING, action.value);
					return;
				}

				nbl::ui::trySelectBindingProjectionIndex(
					getPlanarProjectionSpan(),
					binding,
					static_cast<uint32_t>(action.value));
			} break;

			case nbl::this_example::ECameraScriptedActionCode::SetUseWindow:
				m_viewports.useWindow = action.value != 0;
				break;

			case nbl::this_example::ECameraScriptedActionCode::SetLeftHanded:
				m_viewports.windowBindings[m_viewports.activeRenderWindowIx].leftHandedProjection = action.value != 0;
				break;

			case nbl::this_example::ECameraScriptedActionCode::ResetActiveCamera:
			{
				auto& binding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
				if (binding.activePlanarIx >= m_planarProjections.size())
				{
					m_logger->log("[script][warn] action reset_active_camera active planar out of range: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
					return;
				}
				if (binding.activePlanarIx >= m_presetAuthoring.initialPlanarPresets.size())
				{
					m_logger->log("[script][warn] action reset_active_camera missing initial preset for planar: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
					return;
				}

				auto* camera = m_planarProjections[binding.activePlanarIx]->getCamera();
				if (!nbl::core::CCameraPresetFlowUtilities::applyPreset(m_cameraGoalSolver, camera, m_presetAuthoring.initialPlanarPresets[binding.activePlanarIx]))
					m_logger->log("[script][warn] action reset_active_camera failed for planar: %u", ILogger::ELL_WARNING, binding.activePlanarIx);
			} break;
		}
	};

	for (const auto& action : scriptedActions)
	{
		if (nbl::this_example::CCameraScriptedActionUtilities::hasCode(action, nbl::this_example::ECameraScriptedActionCode::SetActiveRenderWindow))
			applyAction(action);
	}

	for (const auto& action : scriptedActions)
	{
		if (!nbl::this_example::CCameraScriptedActionUtilities::hasCode(action, nbl::this_example::ECameraScriptedActionCode::SetActiveRenderWindow))
			applyAction(action);
	}

	if (m_scriptedInput.log)
	{
		m_logger->log(
			"[script] frame %llu actions=%zu",
			ILogger::ELL_INFO,
			static_cast<unsigned long long>(m_realFrameIx),
			scriptedActions.size());
	}
}

void App::ensureScriptedVisualPlanarState()
{
	if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug && !m_scriptedInput.visualPlanar.valid))
		return;
	if (m_viewports.activeRenderWindowIx >= m_viewports.windowBindings.size())
		return;

	m_scriptedInput.visualPlanar.valid = true;
	m_scriptedInput.visualPlanar.planarIx = m_viewports.windowBindings[m_viewports.activeRenderWindowIx].activePlanarIx;
	m_scriptedInput.visualPlanar.startFrame = m_realFrameIx;
}

void App::updateScriptedMouseButtons(std::span<const SMouseEvent> scriptedMouse)
{
	if (!m_scriptedInput.enabled)
	{
		m_scriptedInput.scriptedMouseButtons.leftDown = false;
		m_scriptedInput.scriptedMouseButtons.rightDown = false;
		return;
	}

	for (const auto& event : scriptedMouse)
	{
		if (event.type != ui::SMouseEvent::EET_CLICK)
			continue;

		if (event.clickEvent.mouseButton == ui::EMB_LEFT_BUTTON)
		{
			if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
				m_scriptedInput.scriptedMouseButtons.leftDown = true;
			else if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
				m_scriptedInput.scriptedMouseButtons.leftDown = false;
		}
		else if (event.clickEvent.mouseButton == ui::EMB_RIGHT_BUTTON)
		{
			if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
				m_scriptedInput.scriptedMouseButtons.rightDown = true;
			else if (event.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
				m_scriptedInput.scriptedMouseButtons.rightDown = false;
		}
	}
}

void App::appendScriptedInputEvents(const SScriptedFrameInputState& scriptedFrame, SCapturedUiEvents& capturedEvents)
{
	updateScriptedMouseButtons(scriptedFrame.mouse);

	if (!scriptedFrame.mouse.empty())
		capturedEvents.mouse.insert(capturedEvents.mouse.end(), scriptedFrame.mouse.begin(), scriptedFrame.mouse.end());
	if (!scriptedFrame.keyboard.empty())
		capturedEvents.keyboard.insert(capturedEvents.keyboard.end(), scriptedFrame.keyboard.begin(), scriptedFrame.keyboard.end());
}

void App::syncDynamicPerspectiveForPlanar(planar_projection_t* planar, ICamera* camera)
{
	if (!planar || !camera)
		return;

	for (auto& projection : planar->getPlanarProjections())
		nbl::core::CCameraProjectionUtilities::syncDynamicPerspectiveProjection(camera, projection);
}

void App::logScriptedVirtualEvents(const char* label, std::span<const CVirtualGimbalEvent> events) const
{
	if (!m_scriptedInput.log)
		return;

	for (const auto& event : events)
	{
		m_logger->log(
			"[script] %s virtual %s magnitude=%.6f",
			ILogger::ELL_INFO,
			label,
			CVirtualGimbalEvent::virtualEventToString(event.type).data(),
			event.magnitude);
	}
}

void App::applyScriptedImguizmoInput(SScriptedFrameInputState& scriptedFrame, const bool skipCameraInput)
{
	scriptedFrame.imguizmoVirtualEvents.clear();
	if (!(m_scriptedInput.enabled && !scriptedFrame.frameEvents.imguizmo.empty() && !skipCameraInput))
		return;

	SActiveScriptedCameraContext runtimeContext = {};
	if (!tryBuildActiveScriptedCameraContext(runtimeContext))
		return;
	auto& binding = *runtimeContext.viewport.binding;
	auto* camera = runtimeContext.viewport.camera;

	CGimbalInputBinder imguizmoBinding;
    CCameraInputBindingUtilities::applyDefaultCameraInputBindingPreset(imguizmoBinding, *camera);
	auto collectedEvents = imguizmoBinding.collectVirtualEvents(m_nextPresentationTimestamp, {
		.imguizmoEvents = { scriptedFrame.frameEvents.imguizmo.data(), scriptedFrame.frameEvents.imguizmo.size() }
	});
	auto& imguizmoEvents = collectedEvents.events;
	const uint32_t virtualEventCount = collectedEvents.imguizmoCount;
	if (!virtualEventCount)
		return;

	scriptedFrame.imguizmoVirtualEvents.assign(imguizmoEvents.begin(), imguizmoEvents.begin() + virtualEventCount);
	const auto virtualEventSpan = std::span<const CVirtualGimbalEvent>(scriptedFrame.imguizmoVirtualEvents.data(), virtualEventCount);
	camera->manipulate(virtualEventSpan);
	appendVirtualEventLog("imguizmo", "ImGuizmo", binding.activePlanarIx, camera, virtualEventSpan.data(), virtualEventCount);
	logScriptedVirtualEvents("imguizmo", virtualEventSpan);
	logScriptedCameraPose("imguizmo", camera);
}

void App::applyScriptedGoals(const CCameraScriptedFrameEvents& scriptedFrameEvents, const bool skipCameraInput)
{
	if (!(m_scriptedInput.enabled && !scriptedFrameEvents.goals.empty() && !skipCameraInput))
		return;

	SActiveScriptedCameraContext runtimeContext = {};
	if (!tryBuildActiveScriptedCameraContext(runtimeContext))
		return;
	auto& planar = *runtimeContext.viewport.planar;
	auto* camera = runtimeContext.viewport.camera;

	auto logGoalFail = [&](const char* fmt, auto&&... args) -> void
	{
		m_scriptedInput.failed = true;
		m_logger->log(fmt, ILogger::ELL_ERROR, std::forward<decltype(args)>(args)...);
	};

	for (const auto& goalEvent : scriptedFrameEvents.goals)
	{
		const auto result = m_cameraGoalSolver.applyDetailed(camera, goalEvent.goal);
		if (!result.succeeded() || (goalEvent.requireExact && !result.exact))
		{
			logGoalFail(
				"[script][fail] goal_apply frame=%llu status=%s exact=%d details=%s",
				static_cast<unsigned long long>(m_realFrameIx),
				result.succeeded() ? "inexact" : "failed",
				result.exact ? 1 : 0,
                CCameraTextUtilities::describeApplyResult(result).c_str());
		}
	}

	syncDynamicPerspectiveForPlanar(&planar, camera);

	logScriptedCameraPose("goal_apply", camera);
}

