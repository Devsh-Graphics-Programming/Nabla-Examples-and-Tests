#include "app/App.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace
{

using control_panel_style_t = nbl::ui::SCameraControlPanelStyle;

enum class EControlPanelToggleBinding : uint8_t
{
	UseWindow,
	ShowHud,
	ShowEventLog
};

enum class EControlPanelTabGate : uint8_t
{
	Always,
	ShowHud,
	ShowEventLog
};

struct SControlPanelToggleDescriptor final
{
	const char* label = "";
	const char* hint = "";
	EControlPanelToggleBinding binding = EControlPanelToggleBinding::UseWindow;
};

struct SControlPanelTabEntry final
{
	const char* label = "";
	EControlPanelTabGate gate = EControlPanelTabGate::Always;
	void (App::*draw)(const nbl::ui::SCameraControlPanelStyle&) = nullptr;
};

inline constexpr std::array<SControlPanelToggleDescriptor, 3u> ControlPanelToggles = {{
	{ "WINDOW", "Toggle split render windows", EControlPanelToggleBinding::UseWindow },
	{ "STATUS", "Show system and camera status panel", EControlPanelToggleBinding::ShowHud },
	{ "EVENT LOG", "Show virtual event log", EControlPanelToggleBinding::ShowEventLog }
}};

inline float calcControlPanelToggleRowWidth(
	std::span<const SControlPanelToggleDescriptor> toggles,
	const control_panel_style_t& panelStyle,
	const float gap)
{
	float rowWidth = 0.0f;
	for (size_t toggleIx = 0u; toggleIx < toggles.size(); ++toggleIx)
	{
		if (toggleIx > 0u)
			rowWidth += gap;
		rowWidth += nbl::ui::CCameraControlPanelUiUtilities::calcPillWidth(toggles[toggleIx].label, panelStyle.TogglePadding);
	}
	return rowWidth;
}

} // namespace

void App::drawControlPanelHeader(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, panelStyle.CardChildRounding);
	if (ImGui::BeginChild("PanelHeader", ImVec2(0.0f, panelStyle.HeaderWindowHeight), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
	{
		ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderDummyY));
		ImGui::SetWindowFontScale(panelStyle.HeaderTitleFontScale);
		ImGui::TextColored(panelStyle.AccentColor, "Control Panel");
		ImGui::SetWindowFontScale(1.0f);

		const float gap = ImGui::GetStyle().ItemSpacing.x;
		std::array<nbl::ui::SCameraControlPanelBadgeData, 4u> headerBadges = {{
			{ m_viewports.useWindow ? "WINDOW" : "FULL", panelStyle.AccentColor },
			{ m_viewports.enableActiveCameraMovement ? "MOVE ON" : "MOVE OFF", m_viewports.enableActiveCameraMovement ? panelStyle.GoodColor : panelStyle.BadColor },
			{ m_scriptedInput.enabled ? (m_scriptedInput.exclusive ? "SCRIPT EXCL" : "SCRIPT") : "SCRIPT OFF", m_scriptedInput.enabled ? panelStyle.AccentColor : panelStyle.InactiveBadgeColor },
			{ "CI", panelStyle.WarnColor }
		}};
		const size_t headerBadgeCount = m_cliRuntime.ciMode ? headerBadges.size() : headerBadges.size() - 1u;
		nbl::ui::CCameraControlPanelUiUtilities::drawBadgeRow(std::span<const nbl::ui::SCameraControlPanelBadgeData>(headerBadges.data(), headerBadgeCount), panelStyle.BadgeTextColor, gap, panelStyle);

		ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderGapSmall));
		const std::array<nbl::ui::SCameraControlPanelKeyHintGroup, 3u> keyHintGroups = {{
			{ "Move", nbl::ui::SCameraControlPanelHeaderHints::MoveKeys },
			{ "Look", nbl::ui::SCameraControlPanelHeaderHints::LookKeys },
			{ "Zoom", nbl::ui::SCameraControlPanelHeaderHints::ZoomKeys }
		}};
		nbl::ui::CCameraControlPanelUiUtilities::drawKeyHintGroupRow(keyHintGroups, gap, gap * 2.0f, panelStyle.KeyBackgroundColor, panelStyle.KeyTextColor, panelStyle);

		ImGui::Dummy(ImVec2(0.0f, panelStyle.HeaderGapSmall));
		if (ImGui::BeginTable("HeaderMetrics", 3, ImGuiTableFlags_SizingStretchProp))
		{
			const float frameMs = std::max(0.0f, m_uiMetrics.lastFrameMs);
			const float fps = nbl::ui::CCameraControlPanelUiUtilities::calcFramesPerSecond(frameMs, panelStyle);
			const std::array<nbl::ui::SCameraControlPanelMiniStatSpec, 3u> miniStats = {{
				{ "FrameStat", "Frame", panelStyle.AccentColor, panelStyle.DefaultFrameMetricMin },
				{ "InputStat", "Input", panelStyle.AccentColor, panelStyle.DefaultEventMetricMin },
				{ "VirtualStat", "Virtual", panelStyle.AccentColor, panelStyle.DefaultEventMetricMin }
			}};

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			nbl::ui::CCameraControlPanelUiUtilities::drawMiniStat(miniStats[0], m_uiMetrics.frameMs, m_uiMetrics.sampleIndex, [&]
			{
				ImGui::TextColored(panelStyle.AccentColor, "%.1f ms  %.0f fps", frameMs, fps);
			}, panelStyle);

			ImGui::TableSetColumnIndex(1);
			nbl::ui::CCameraControlPanelUiUtilities::drawMiniStat(miniStats[1], m_uiMetrics.inputCounts, m_uiMetrics.sampleIndex, [&]
			{
				ImGui::TextColored(panelStyle.AccentColor, "%u ev", m_uiMetrics.lastInputEvents);
			}, panelStyle);

			ImGui::TableSetColumnIndex(2);
			nbl::ui::CCameraControlPanelUiUtilities::drawMiniStat(miniStats[2], m_uiMetrics.virtualCounts, m_uiMetrics.sampleIndex, [&]
			{
				ImGui::TextColored(panelStyle.AccentColor, "%u ev", m_uiMetrics.lastVirtualEvents);
			}, panelStyle);
			ImGui::EndTable();
		}
	}
	ImGui::EndChild();
	ImGui::PopStyleVar();
}

void App::drawControlPanelToggles(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	const float gap = ImGui::GetStyle().ItemSpacing.x;
	const auto getToggleValue = [&](const EControlPanelToggleBinding binding) -> bool&
	{
		switch (binding)
		{
			case EControlPanelToggleBinding::UseWindow:
				return m_viewports.useWindow;
			case EControlPanelToggleBinding::ShowHud:
				return m_eventLog.showHud;
			case EControlPanelToggleBinding::ShowEventLog:
			default:
				return m_eventLog.showEventLog;
		}
	};

	const float rowWidth = calcControlPanelToggleRowWidth(ControlPanelToggles, panelStyle, gap);
	nbl::ui::CCameraControlPanelUiUtilities::centerControlPanelRow(rowWidth);
	for (size_t toggleIx = 0u; toggleIx < ControlPanelToggles.size(); ++toggleIx)
	{
		if (toggleIx > 0u)
			ImGui::SameLine(0.0f, gap);

		const auto& toggle = ControlPanelToggles[toggleIx];
		nbl::ui::CCameraControlPanelUiUtilities::drawTogglePill(
			toggle.label,
			getToggleValue(toggle.binding),
			panelStyle.AccentColor,
			panelStyle.InactiveBadgeColor,
			panelStyle.BadgeTextColor,
			panelStyle.TogglePadding);
		nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint(toggle.hint);
	}
}

void App::drawControlPanelTabs(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
	static constexpr std::array<SControlPanelTabEntry, 7u> ControlPanelTabs = {{
		{ "Status", EControlPanelTabGate::ShowHud, &App::drawControlPanelStatusTab },
		{ "Projection", EControlPanelTabGate::Always, &App::drawControlPanelProjectionTab },
		{ "Camera", EControlPanelTabGate::Always, &App::drawControlPanelCameraTab },
		{ "Presets", EControlPanelTabGate::Always, &App::drawControlPanelPresetsTab },
		{ "Playback", EControlPanelTabGate::Always, &App::drawControlPanelPlaybackTab },
		{ "Gizmo", EControlPanelTabGate::Always, &App::drawControlPanelGizmoTab },
		{ "Log", EControlPanelTabGate::ShowEventLog, &App::drawControlPanelLogTab }
	}};

	if (!ImGui::BeginTabBar("ControlTabs"))
		return;

	const auto isTabEnabled = [&](const EControlPanelTabGate gate) -> bool
	{
		switch (gate)
		{
			case EControlPanelTabGate::ShowHud:
				return m_eventLog.showHud;
			case EControlPanelTabGate::ShowEventLog:
				return m_eventLog.showEventLog;
			case EControlPanelTabGate::Always:
			default:
				return true;
		}
	};

	for (const auto& tab : ControlPanelTabs)
	{
		if (!isTabEnabled(tab.gate) || !ImGui::BeginTabItem(tab.label))
			continue;

		(this->*tab.draw)(panelStyle);
		ImGui::EndTabItem();
	}

	ImGui::EndTabBar();
}

