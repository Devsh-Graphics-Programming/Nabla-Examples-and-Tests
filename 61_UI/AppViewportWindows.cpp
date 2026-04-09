#include "app/App.hpp"
#include "app/AppViewportBindingUtilities.hpp"
#include "app/AppViewportWindowUtilities.hpp"

void App::drawWindowedViewportWindows(ImGuiIO& io, SImResourceInfo& info)
{
	syncVisualDebugWindowBindings();
	const bool hideSceneGizmos = m_viewports.enableActiveCameraMovement || (m_scriptedInput.enabled && m_scriptedInput.visualDebug);
	ImGuizmo::Enable(!hideSceneGizmos);

	size_t gizmoIx = 0u;
	const ImGuiCond windowCond = m_cliRuntime.ciMode ? ImGuiCond_Always : ImGuiCond_Appearing;

	for (uint32_t windowIx = 0u; windowIx < m_viewports.windowBindings.size(); ++windowIx)
		drawWindowedViewportWindow(windowIx, windowCond, hideSceneGizmos, gizmoIx, info);

	if (m_viewports.windowBindings.size() > 1u)
		drawViewportSplitOverlayWindow(io.DisplaySize);
}

void App::drawWindowedViewportWindow(uint32_t windowIx, ImGuiCond windowCond, bool hideSceneGizmos, size_t& gizmoIx, SImResourceInfo& info)
{
	const auto& rw = m_viewports.windowInit.renderWindows[windowIx];
	ImGui::SetNextWindowPos({ rw.iPos.x, rw.iPos.y }, windowCond);
	ImGui::SetNextWindowSize({ rw.iSize.x, rw.iSize.y }, windowCond);
	ImGui::SetNextWindowSizeConstraints(SCameraAppViewportDefaults::MinWindowSize, SCameraAppViewportDefaults::MaxWindowSize);

	nbl::ui::pushViewportWindowStyle();
	const std::string ident = "Render Window \"" + std::to_string(windowIx) + "\"";

	ImGui::Begin(ident.data(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus);
	auto& binding = m_viewports.windowBindings[windowIx];
	nbl::ui::SViewportWindowRuntime viewportRuntime = {};
	const auto planarSpan = getPlanarProjectionSpan();
	const bool viewportValid = nbl::ui::tryBuildViewportWindowRuntime(planarSpan, binding, SCameraAppViewportDefaults::FlipGizmoY, viewportRuntime);
	const auto& frame = viewportRuntime.frame;

	if (ImGuiWindow* const window = ImGui::GetCurrentWindow())
		nbl::ui::updateViewportWindowMoveFlag(window, frame);

	if (!viewportValid)
	{
		ImGui::End();
		nbl::ui::popViewportWindowStyle();
		return;
	}
	const auto& viewportState = viewportRuntime.viewportState;

	auto& projection = *viewportState.projection;
	info.textureID = SCameraAppUiTextureSlots::viewport(windowIx);

	ImGuizmo::AllowAxisFlip(binding.allowGizmoAxesToFlip);
	ImGuizmo::SetOrthographic(projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic);
	ImGuizmo::SetDrawlist();
	nbl::ui::drawViewportTextureAndOverlay(
		info,
		viewportRuntime,
		m_sceneInteraction.followTarget,
		m_scriptedInput,
		[&](ImDrawList& drawList, const nbl::ui::SViewportOverlayRect& viewportRect, const nbl::ui::SBoundViewportCameraState& state)
		{
			drawViewportWindowOverlay(drawList, viewportRect, windowIx, binding, state);
		});

	updateActiveRenderWindowFromViewport(windowIx, frame.hovered, frame.focused);

	if (!hideSceneGizmos)
		drawViewportManipulationGizmos(windowIx, binding, viewportState, gizmoIx);

	ImGui::End();
	nbl::ui::popViewportWindowStyle();
}

void App::drawViewportWindowOverlay(
	ImDrawList& drawList,
	const nbl::ui::SViewportOverlayRect& viewportRect,
	uint32_t windowIx,
	const SWindowControlBinding& binding,
	const nbl::ui::SBoundViewportCameraState& viewportState) const
{
	const char* projLabel = viewportState.projection->getParameters().m_type == IPlanarProjection::CProjection::Perspective ? "Persp" : "Ortho";
	nbl::ui::SCameraViewportInfoOverlayData overlayData = {};
	overlayData.headline = "Planar " + std::to_string(binding.activePlanarIx) + " | " + projLabel + " | W" + std::to_string(windowIx);
    overlayData.description = std::string(CCameraTextUtilities::getCameraTypeLabel(viewportState.camera)) + ": " + std::string(CCameraTextUtilities::getCameraTypeDescription(viewportState.camera));
	overlayData.detail = "Frustum: active camera (hidden in owner view)";
	nbl::ui::drawViewportInfoOverlay(drawList, viewportRect, overlayData);
}

void App::updateActiveRenderWindowFromViewport(uint32_t windowIx, bool windowHovered, bool windowFocused)
{
	if (m_scriptedInput.enabled && m_scriptedInput.exclusive)
		return;

	if (!m_scriptedInput.enabled && windowHovered)
		m_viewports.activeRenderWindowIx = windowIx;
	else if (windowFocused)
		m_viewports.activeRenderWindowIx = windowIx;
}

void App::drawViewportSplitOverlayWindow(const ImVec2& displaySize)
{
	const auto& topRw = m_viewports.windowInit.renderWindows[0];
	const float splitY = topRw.iPos.y + topRw.iSize.y;
	const float gap = std::max(0.0f, m_viewports.windowInit.renderWindows[1].iPos.y - splitY);
	ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f), ImGuiCond_Always);
	ImGui::SetNextWindowSize(displaySize, ImGuiCond_Always);
	ImGui::Begin("SplitOverlay", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoBringToFrontOnFocus);
	if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
		nbl::ui::drawViewportSplitOverlay(*drawList, displaySize, splitY, gap);
	ImGui::End();
}

void App::drawFullscreenViewportWindow(ImGuiIO& io, SImResourceInfo& info)
{
	info.textureID = SCameraAppUiTextureSlots::viewport(m_viewports.activeRenderWindowIx);

	ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
	ImGui::SetNextWindowSize(io.DisplaySize);
	ImGui::PushStyleColor(ImGuiCol_WindowBg, nbl::ui::SCameraViewportWindowStyle::WindowBackgroundColor);
	ImGui::Begin("FullScreenWindow", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
	nbl::ui::SViewportWindowRuntime viewportRuntime = {};
	const auto planarSpan = getPlanarProjectionSpan();
	const bool viewportValid = nbl::ui::tryBuildViewportWindowRuntime(planarSpan, m_viewports.windowBindings[m_viewports.activeRenderWindowIx], false, viewportRuntime);
	if (viewportValid)
	{
		nbl::ui::drawViewportTextureAndOverlay(
			info,
			viewportRuntime,
			m_sceneInteraction.followTarget,
			m_scriptedInput,
			[](ImDrawList&, const nbl::ui::SViewportOverlayRect&, const nbl::ui::SBoundViewportCameraState&) {});
	}
	else
	{
		const auto& frame = viewportRuntime.frame;
		ImGui::Image(info, frame.contentRegionSize);
		ImGuizmo::SetRect(frame.cursorPos.x, frame.cursorPos.y, frame.contentRegionSize.x, frame.contentRegionSize.y);
	}

	ImGui::End();
	ImGui::PopStyleColor(1);
}

void App::refreshViewportBindingMatrices()
{
	const auto planarSpan = getPlanarProjectionSpan();
	for (auto& binding : m_viewports.windowBindings)
	{
		nbl::ui::SBoundViewportCameraState viewportState = {};
		nbl::ui::tryBuildWindowBindingMatrices(planarSpan, binding, viewportState);
	}
}
