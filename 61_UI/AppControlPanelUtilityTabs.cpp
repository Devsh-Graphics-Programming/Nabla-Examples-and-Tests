#include "app/App.hpp"

void App::drawControlPanelGizmoTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
    if (!nbl::ui::beginControlPanelTabChild("GizmoPanel", panelStyle))
    {
        nbl::ui::endControlPanelTabChild();
        return;
    }

    nbl::ui::drawSectionHeader("GizmoHeader", "Gizmo", panelStyle.AccentColor, panelStyle);
    TransformEditorContents();
    nbl::ui::endControlPanelTabChild();
}

void App::drawControlPanelLogTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
    auto& eventLog = m_eventLog;

    if (!nbl::ui::beginControlPanelTabChild("LogPanel", panelStyle))
    {
        nbl::ui::endControlPanelTabChild();
        return;
    }

    nbl::ui::drawSectionHeader("LogHeader", "Virtual Events", panelStyle.AccentColor, panelStyle);
    ImGui::Checkbox("Auto-scroll", &eventLog.autoScroll);
    ImGui::SameLine();
    ImGui::Checkbox("Wrap", &eventLog.wrap);
    ImGui::Separator();

    const ImGuiWindowFlags logFlags = eventLog.wrap ? ImGuiWindowFlags_None : ImGuiWindowFlags_HorizontalScrollbar;
    if (ImGui::BeginChild("LogList", ImVec2(0.0f, 0.0f), false, logFlags))
    {
        const float scrollY = ImGui::GetScrollY();
        const float scrollMax = ImGui::GetScrollMaxY();
        const bool wasAtBottom = scrollY >= scrollMax - panelStyle.EventLogBottomThreshold;
        const size_t start = eventLog.entries.size() > SCameraAppAuthoringDefaults::EventLogVisibleEntries ?
            eventLog.entries.size() - SCameraAppAuthoringDefaults::EventLogVisibleEntries :
            0u;
        if (eventLog.wrap)
            ImGui::PushTextWrapPos(0.0f);
        for (size_t i = start; i < eventLog.entries.size(); ++i)
            ImGui::TextUnformatted(eventLog.entries[i].line.c_str());
        if (eventLog.wrap)
            ImGui::PopTextWrapPos();
        if (eventLog.autoScroll && wasAtBottom && !eventLog.entries.empty())
            ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    nbl::ui::endControlPanelTabChild();
}
