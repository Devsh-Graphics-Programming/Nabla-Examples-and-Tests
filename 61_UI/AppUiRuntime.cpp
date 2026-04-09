#include "app/App.hpp"

void App::paceScriptedVisualDebugFrame()
{
	if (!(m_scriptedInput.enabled && m_scriptedInput.visualDebug))
	{
		m_scriptedInput.framePacer.initialized = false;
		return;
	}

	if (m_scriptedInput.visualTargetFps <= 0.f)
		return;

	const auto frameDuration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		std::chrono::duration<double>(1.0 / static_cast<double>(m_scriptedInput.visualTargetFps)));
	const auto now = std::chrono::steady_clock::now();

	if (!m_scriptedInput.framePacer.initialized)
	{
		m_scriptedInput.framePacer.initialized = true;
		m_scriptedInput.framePacer.nextFrame = now + frameDuration;
		return;
	}

	if (now < m_scriptedInput.framePacer.nextFrame)
		std::this_thread::sleep_until(m_scriptedInput.framePacer.nextFrame);

	auto postSleepNow = std::chrono::steady_clock::now();
	while (m_scriptedInput.framePacer.nextFrame < postSleepNow)
		m_scriptedInput.framePacer.nextFrame += frameDuration;
}

bool App::keepRunning()
{
	if (m_cliRuntime.headlessCameraSmokeMode)
		return false;

	if (m_scriptedInput.enabled && m_scriptedInput.hardFail && m_scriptedInput.failed)
	{
		if (!m_cliRuntime.ciMode || m_cliRuntime.ciScreenshotDone)
			std::exit(EXIT_FAILURE);
	}

	if (m_cliRuntime.ciMode && m_cliRuntime.ciStartedAt != clock_t::time_point::min())
	{
		const auto elapsed = clock_t::now() - m_cliRuntime.ciStartedAt;
		if (elapsed > SCameraAppRuntimeDefaults::CiMaxRuntime)
		{
			m_logger->log(
				"[ci][fail] watchdog timeout after %.2f s.",
				ILogger::ELL_ERROR,
				std::chrono::duration<double>(elapsed).count());
			std::exit(EXIT_FAILURE);
		}
	}

	if (m_cliRuntime.ciMode && m_cliRuntime.ciScreenshotDone)
	{
		if (m_scriptedInput.enabled)
		{
			if (m_scriptedInput.nextCaptureIndex < m_scriptedInput.timeline.captureFrames.size())
				return true;
			if (m_scriptedInput.nextEventIndex < m_scriptedInput.timeline.events.size())
				return true;
			if (m_scriptedInput.checkRuntime.nextCheckIndex < m_scriptedInput.timeline.checks.size())
				return true;
		}
		return false;
	}

	return !m_surface->irrecoverable();
}

bool App::onAppTerminated()
{
	if (m_cliRuntime.headlessCameraSmokeMode)
		return m_cliRuntime.headlessCameraSmokePassed;

	return base_t::onAppTerminated();
}

void App::syncWindowInputBinding(SWindowControlBinding& binding)
{
	if (!binding.boundProjectionIx.has_value())
		return;
	if (binding.activePlanarIx >= m_planarProjections.size())
		return;

	auto& planar = m_planarProjections[binding.activePlanarIx];
	if (!planar)
		return;

	const auto projectionIx = binding.boundProjectionIx.value();
	auto& projections = planar->getPlanarProjections();
	if (projectionIx >= projections.size())
		return;

	if (binding.inputBindingPlanarIx == binding.activePlanarIx && binding.inputBindingProjectionIx == projectionIx)
		return;

	binding.inputBinding.copyBindingLayoutFrom(projections[projectionIx].getInputBinding());
	binding.inputBindingPlanarIx = binding.activePlanarIx;
	binding.inputBindingProjectionIx = projectionIx;
}

void App::syncWindowInputBindingToProjection(SWindowControlBinding& binding)
{
	if (!binding.boundProjectionIx.has_value())
		return;
	if (binding.activePlanarIx >= m_planarProjections.size())
		return;

	auto& planar = m_planarProjections[binding.activePlanarIx];
	if (!planar)
		return;

	const auto projectionIx = binding.boundProjectionIx.value();
	auto& projections = planar->getPlanarProjections();
	if (projectionIx >= projections.size())
		return;

	projections[projectionIx].getInputBinding().copyBindingLayoutFrom(binding.inputBinding);
	binding.inputBindingPlanarIx = binding.activePlanarIx;
	binding.inputBindingProjectionIx = projectionIx;
}

bool App::shouldCaptureOSCursor()
{
	if (!m_viewports.enableActiveCameraMovement || !m_viewports.captureCursorInMoveMode)
		return false;
	if (m_cliRuntime.ciMode || m_scriptedInput.enabled)
		return false;
	if (!m_window || !m_window->hasInputFocus() || !m_window->hasMouseFocus())
		return false;
	return true;
}

void App::UpdateBoundCameraMovement()
{
	ImGuiIO& io = ImGui::GetIO();

	if (ImGui::IsKeyPressed(ImGuiKey_Space))
		m_viewports.enableActiveCameraMovement = !m_viewports.enableActiveCameraMovement;

	if (m_viewports.enableActiveCameraMovement)
	{
		io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
		io.MouseDrawCursor = false;
		io.WantCaptureMouse = false;

		if (shouldCaptureOSCursor())
		{
			const ImVec2 viewportSize = io.DisplaySize;
			auto* cc = m_window->getCursorControl();
			if (cc)
			{
				const int32_t posX = m_window->getX();
				const int32_t posY = m_window->getY();

				if (m_viewports.resetCursorToCenter)
				{
					const ICursorControl::SPosition middle{
						static_cast<int32_t>(viewportSize.x / 2 + posX),
						static_cast<int32_t>(viewportSize.y / 2 + posY)
					};
					cc->setPosition(middle);
				}
				else
				{
					const auto currentCursorPos = cc->getPosition();
					ICursorControl::SPosition newPos{};
					newPos.x = std::clamp<int32_t>(currentCursorPos.x, posX, static_cast<int32_t>(viewportSize.x) + posX);
					newPos.y = std::clamp<int32_t>(currentCursorPos.y, posY, static_cast<int32_t>(viewportSize.y) + posY);
					cc->setPosition(newPos);
				}
			}
		}
	}
	else
	{
		io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
		io.MouseDrawCursor = false;
		io.WantCaptureMouse = true;
	}
}

void App::UpdateCursorVisibility()
{
	auto* cc = m_window ? m_window->getCursorControl() : nullptr;
	if (!cc)
		return;

	cc->setVisible(!shouldCaptureOSCursor());
}

void App::UpdateUiMetrics()
{
	m_uiMetrics.lastFrameMs = static_cast<float>(m_presentationTiming.frameDeltaSec * 1000.0);
	m_uiMetrics.lastInputEvents = m_uiMetrics.inputEventsThisFrame;
	m_uiMetrics.lastVirtualEvents = m_uiMetrics.virtualEventsThisFrame;

	m_uiMetrics.frameMs[m_uiMetrics.sampleIndex] = m_uiMetrics.lastFrameMs;
	m_uiMetrics.inputCounts[m_uiMetrics.sampleIndex] = static_cast<float>(m_uiMetrics.inputEventsThisFrame);
	m_uiMetrics.virtualCounts[m_uiMetrics.sampleIndex] = static_cast<float>(m_uiMetrics.virtualEventsThisFrame);

	m_uiMetrics.sampleIndex = (m_uiMetrics.sampleIndex + 1u) % SCameraAppRuntimeDefaults::UiMetricSamples;
	m_uiMetrics.inputEventsThisFrame = 0u;
	m_uiMetrics.virtualEventsThisFrame = 0u;
}

void App::addMatrixTable(const char* topText, const char* tableName, const int rows, const int columns, const float* pointer, const bool withSeparator)
{
	ImGui::Text(topText);
	ImGui::PushStyleColor(ImGuiCol_TableRowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
	ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, ImGui::GetStyleColorVec4(ImGuiCol_WindowBg));
	if (ImGui::BeginTable(tableName, columns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
	{
		for (int y = 0; y < rows; ++y)
		{
			ImGui::TableNextRow();
			for (int x = 0; x < columns; ++x)
			{
				ImGui::TableSetColumnIndex(x);
				if (pointer)
					ImGui::Text("%.3f", *(pointer + (y * columns) + x));
				else
					ImGui::Text("-");
			}
		}
		ImGui::EndTable();
	}
	ImGui::PopStyleColor(2);
	if (withSeparator)
		ImGui::Separator();
}

void App::finalizeUiFrameState()
{
	UpdateBoundCameraMovement();
	UpdateCursorVisibility();
	applyFollowToConfiguredCameras();
	refreshViewportBindingMatrices();
}
