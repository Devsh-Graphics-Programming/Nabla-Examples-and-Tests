#include "app/App.hpp"
#include "camera/CCameraPersistence.hpp"

bool App::savePresetsToFile(const nbl::system::path& path)
{
	return nbl::system::savePresetCollectionToFile(
		*m_system,
		path,
		std::span<const CameraPreset>(m_presetAuthoring.presets.data(), m_presetAuthoring.presets.size()));
}

bool App::loadPresetsFromFile(const nbl::system::path& path)
{
	return nbl::system::loadPresetCollectionFromFile(*m_system, path, m_presetAuthoring.presets);
}

bool App::saveKeyframesToFile(const nbl::system::path& path)
{
	return nbl::system::saveKeyframeTrackToFile(*m_system, path, m_playbackAuthoring.keyframeTrack);
}

bool App::loadKeyframesFromFile(const nbl::system::path& path)
{
	if (!nbl::system::loadKeyframeTrackFromFile(*m_system, path, m_playbackAuthoring.keyframeTrack))
		return false;

	clampPlaybackTimeToKeyframes();
	if (m_playbackAuthoring.keyframeTrack.keyframes.empty())
		clearApplyStatusBanner(m_playbackAuthoring.applyBanner);
	return true;
}

void App::DrawControlPanel()
{
	const nbl::ui::SCameraControlPanelStyle panelStyle = {};
	const ImVec2 displaySize = ImGui::GetIO().DisplaySize;
	const ImVec2 panelSize = nbl::ui::CCameraControlPanelUiUtilities::calcControlPanelWindowSize(displaySize, panelStyle);
	const ImVec2 panelPos = { 0.0f, 0.0f };
	ImGui::SetNextWindowPos(panelPos, ImGuiCond_Always);
	ImGui::SetNextWindowSize(panelSize, ImGuiCond_Always);

	nbl::ui::CCameraControlPanelUiUtilities::pushControlPanelWindowStyle(panelStyle);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Always);
	ImGui::SetNextWindowBgAlpha(0.0f);
	if (m_cliRuntime.ciMode)
		ImGui::SetNextWindowFocus();
	ImGui::Begin("Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);

	if (auto* drawList = ImGui::GetWindowDrawList(); drawList)
		nbl::ui::CCameraControlPanelUiUtilities::drawControlPanelWindowBackdrop(*drawList, ImGui::GetWindowPos(), ImGui::GetWindowSize(), panelStyle);

	drawControlPanelHeader(panelStyle);
	ImGui::Spacing();
	drawControlPanelToggles(panelStyle);
	ImGui::Separator();
	drawControlPanelTabs(panelStyle);

	ImGui::End();
	nbl::ui::CCameraControlPanelUiUtilities::popControlPanelWindowStyle();
}

