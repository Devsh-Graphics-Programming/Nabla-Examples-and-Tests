#include "app/App.hpp"

void App::imguiListen()
{
	ImGuiIO& io = ImGui::GetIO();
	if (m_cliRuntime.ciMode)
		io.IniFilename = nullptr;

	ImGuizmo::BeginFrame();

	auto info = SCameraAppUiTextureSlots::makeDefaultViewportResourceInfo();

	if (m_viewports.useWindow)
		drawWindowedViewportWindows(io, info);
	else
		drawFullscreenViewportWindow(io, info);

	drawScriptVisualDebugOverlay(io.DisplaySize);
	DrawControlPanel();
	finalizeUiFrameState();
}

