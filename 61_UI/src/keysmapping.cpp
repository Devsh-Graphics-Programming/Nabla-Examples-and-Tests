#include "keysmapping.hpp"
#include "app/AppTypes.hpp"

#include <string>
#include <unordered_map>

inline std::string buildKeyCodeLabel(const ui::E_KEY_CODE keyCode)
{
    return std::string(1u, ui::keyCodeToChar(keyCode, true));
}

inline ImVec4 getBindingActiveStatusColor(const bool active)
{
    return active ? SCameraAppBindingEditorUiDefaults::ActiveStatusColor : SCameraAppBindingEditorUiDefaults::InactiveStatusColor;
}

bool handleAddMapping(const char* tableID, IGimbalBindingLayout* layout, IGimbalBindingLayout::BindingDomain activeBindingDomain, CVirtualGimbalEvent::VirtualEventType& selectedEventType, ui::E_KEY_CODE& newKey, ui::E_MOUSE_CODE& newMouseCode, bool& addMode)
{
    bool anyMapUpdated = false;
    ImGui::BeginTable(tableID, 3, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
    ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, SCameraAppBindingEditorUiDefaults::TableColumnWeight);
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, SCameraAppBindingEditorUiDefaults::TableColumnWeight);
    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, SCameraAppBindingEditorUiDefaults::TableColumnWeight);
    ImGui::TableHeadersRow();

    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    if (ImGui::BeginCombo("##selectEvent", CVirtualGimbalEvent::virtualEventToString(selectedEventType).data()))
    {
        for (const auto& eventType : CVirtualGimbalEvent::VirtualEventsTypeTable)
        {
            bool isSelected = (selectedEventType == eventType);
            if (ImGui::Selectable(CVirtualGimbalEvent::virtualEventToString(eventType).data(), isSelected))
                selectedEventType = eventType;
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::TableSetColumnIndex(1);
    if (activeBindingDomain == IGimbalBindingLayout::Keyboard)
    {
        const auto newKeyDisplay = buildKeyCodeLabel(newKey);
        if (ImGui::BeginCombo("##selectKey", newKeyDisplay.c_str()))
        {
            for (int i = ui::E_KEY_CODE::EKC_A; i <= ui::E_KEY_CODE::EKC_Z; ++i)
            {
                bool isSelected = (newKey == static_cast<ui::E_KEY_CODE>(i));
                const auto label = buildKeyCodeLabel(static_cast<ui::E_KEY_CODE>(i));
                if (ImGui::Selectable(label.c_str(), isSelected))
                    newKey = static_cast<ui::E_KEY_CODE>(i);
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }
    else
    {
        if (ImGui::BeginCombo("##selectMouseKey", ui::mouseCodeToString(newMouseCode).data()))
        {
            for (int i = ui::EMC_LEFT_BUTTON; i < ui::EMC_COUNT; ++i)
            {
                bool isSelected = (newMouseCode == static_cast<ui::E_MOUSE_CODE>(i));
                if (ImGui::Selectable(ui::mouseCodeToString(static_cast<ui::E_MOUSE_CODE>(i)).data(), isSelected))
                    newMouseCode = static_cast<ui::E_MOUSE_CODE>(i);
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }

    ImGui::TableSetColumnIndex(2);
    if (ImGui::Button("Confirm Add", SCameraAppBindingEditorUiDefaults::ActionButtonSize))
    {
        anyMapUpdated |= true;
        if (activeBindingDomain == IGimbalBindingLayout::Keyboard)
            layout->updateKeyboardMapping([&](auto& keys) { keys[newKey] = selectedEventType; });
        else
            layout->updateMouseMapping([&](auto& mouse) { mouse[newMouseCode] = selectedEventType; });
        addMode = false;
    }

    ImGui::EndTable();

    return anyMapUpdated;
}

bool displayKeyMappingsAndVirtualStatesInline(IGimbalBindingLayout* layout, bool spawnWindow)
{
    bool anyMapUpdated = false;

    if (!layout) return anyMapUpdated;

    struct MappingState
    {
        bool addMode = false;
        CVirtualGimbalEvent::VirtualEventType selectedEventType = CVirtualGimbalEvent::VirtualEventType::MoveForward;
        ui::E_KEY_CODE newKey = ui::E_KEY_CODE::EKC_A;
        ui::E_MOUSE_CODE newMouseCode = ui::EMC_LEFT_BUTTON;
        IGimbalBindingLayout::BindingDomain activeBindingDomain = IGimbalBindingLayout::Keyboard;
    };

    static std::unordered_map<IGimbalBindingLayout*, MappingState> cameraStates;
    auto& state = cameraStates[layout];

    const auto& keyboardMappings = layout->getKeyboardVirtualEventMap();
    const auto& mouseMappings = layout->getMouseVirtualEventMap();

    if (spawnWindow)
    {
        ImGui::SetNextWindowSize(SCameraAppBindingEditorUiDefaults::WindowInitialSize, ImGuiCond_FirstUseEver);
        ImGui::Begin("Binding Layouts & Virtual States", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar);
    }

    if (ImGui::BeginTabBar("BindingsTabBar"))
    {
        if (ImGui::BeginTabItem("Keyboard"))
        {
            state.activeBindingDomain = IGimbalBindingLayout::Keyboard;
            ImGui::Separator();

            if (ImGui::Button("Add Key", SCameraAppBindingEditorUiDefaults::ActionButtonSize))
                state.addMode = !state.addMode;

            ImGui::Separator();

            ImGui::BeginTable("KeyboardMappingsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
            ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Key(s)", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Active Status", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Magnitude", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableHeadersRow();

            for (const auto& [keyboardCode, hash] : keyboardMappings)
            {
                ImGui::TableNextRow();
                const char* eventName = CVirtualGimbalEvent::virtualEventToString(hash.event.type).data();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", eventName);

                ImGui::TableSetColumnIndex(1);
                const auto keyString = buildKeyCodeLabel(keyboardCode);
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", keyString.c_str());

                ImGui::TableSetColumnIndex(2);
                bool isActive = (hash.event.magnitude > 0);
                const ImVec4 statusColor = getBindingActiveStatusColor(isActive);
                ImGui::TextColored(statusColor, "%s", isActive ? "Active" : "Inactive");

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.2f", hash.event.magnitude);

                ImGui::TableSetColumnIndex(4);
                if (ImGui::Button(("Delete##deleteKey" + std::to_string(static_cast<int>(keyboardCode))).c_str()))
                {
                    anyMapUpdated |= true;
                    layout->updateKeyboardMapping([keyboardCode](auto& keys) { keys.erase(keyboardCode); });
                    break;
                }
            }
            ImGui::EndTable();

            if (state.addMode)
            {
                ImGui::Separator();
                anyMapUpdated |= handleAddMapping("AddKeyboardMappingTable", layout, state.activeBindingDomain, state.selectedEventType, state.newKey, state.newMouseCode, state.addMode);
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Mouse"))
        {
            state.activeBindingDomain = IGimbalBindingLayout::Mouse;
            ImGui::Separator();

            if (ImGui::Button("Add Key", SCameraAppBindingEditorUiDefaults::ActionButtonSize))
                state.addMode = !state.addMode;

            ImGui::Separator();

            ImGui::BeginTable("MouseMappingsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
            ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Mouse Button(s)", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Active Status", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Magnitude", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.2f);
            ImGui::TableHeadersRow();

            for (const auto& [mouseCode, hash] : mouseMappings)
            {
                ImGui::TableNextRow();
                const char* eventName = CVirtualGimbalEvent::virtualEventToString(hash.event.type).data();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", eventName);

                ImGui::TableSetColumnIndex(1);
                const char* mouseButtonName = ui::mouseCodeToString(mouseCode).data();
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", mouseButtonName);

                ImGui::TableSetColumnIndex(2);
                bool isActive = (hash.event.magnitude > 0);
                const ImVec4 statusColor = getBindingActiveStatusColor(isActive);
                ImGui::TextColored(statusColor, "%s", isActive ? "Active" : "Inactive");

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.2f", hash.event.magnitude);

                ImGui::TableSetColumnIndex(4);
                if (ImGui::Button(("Delete##deleteMouse" + std::to_string(static_cast<int>(mouseCode))).c_str()))
                {
                    anyMapUpdated |= true;
                    layout->updateMouseMapping([mouseCode](auto& mouse) { mouse.erase(mouseCode); });
                    break;
                }
            }
            ImGui::EndTable();

            if (state.addMode)
            {
                ImGui::Separator();
                handleAddMapping("AddMouseMappingTable", layout, state.activeBindingDomain, state.selectedEventType, state.newKey, state.newMouseCode, state.addMode);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    if (spawnWindow)
        ImGui::End();

    return anyMapUpdated;
}

