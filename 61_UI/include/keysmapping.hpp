#ifndef __NBL_KEYSMAPPING_H_INCLUDED__
#define __NBL_KEYSMAPPING_H_INCLUDED__

#include "common.hpp"
#include "camera/CCameraController.hpp"

template<typename T>
void handleAddMapping(const char* tableID, CCameraController<T>* controller, ICameraController::ControllerType activeController, CVirtualGimbalEvent::VirtualEventType& selectedEventType, ui::E_KEY_CODE& newKey, ui::E_MOUSE_CODE& newMouseCode, bool& addMode)
{
    ImGui::BeginTable(tableID, 3, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
    ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, 0.33f);
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 0.33f);
    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.33f);
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
    if (activeController == ICameraController::Keyboard)
    {
        char newKeyDisplay[2] = { ui::keyCodeToChar(newKey, true), '\0' };
        if (ImGui::BeginCombo("##selectKey", newKeyDisplay))
        {
            for (int i = ui::E_KEY_CODE::EKC_A; i <= ui::E_KEY_CODE::EKC_Z; ++i)
            {
                bool isSelected = (newKey == static_cast<ui::E_KEY_CODE>(i));
                char label[2] = { ui::keyCodeToChar(static_cast<ui::E_KEY_CODE>(i), true), '\0' };
                if (ImGui::Selectable(label, isSelected))
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
    if (ImGui::Button("Confirm Add", ImVec2(100, 30)))
    {
        if (activeController == ICameraController::Keyboard)
            controller->updateKeyboardMapping([&](auto& keys) { keys[newKey] = selectedEventType; });
        else
            controller->updateMouseMapping([&](auto& mouse) { mouse[newMouseCode] = selectedEventType; });
        addMode = false;
    }

    ImGui::EndTable();
}

template<typename T = matrix_precision_t>
void displayKeyMappingsAndVirtualStates(CCameraController<T>* controller)
{
    static bool addMode = false, pendingChanges = false;
    static CVirtualGimbalEvent::VirtualEventType selectedEventType = CVirtualGimbalEvent::VirtualEventType::MoveForward;
    static ui::E_KEY_CODE newKey = ui::E_KEY_CODE::EKC_A;
    static ui::E_MOUSE_CODE newMouseCode = ui::EMC_LEFT_BUTTON;
    static ICameraController::ControllerType activeController = ICameraController::Keyboard;

    const auto& keyboardMappings = controller->getKeyboardVirtualEventMap();
    const auto& mouseMappings = controller->getMouseVirtualEventMap();

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(600, 400), ImVec2(600, 69000));

    ImGui::Begin("Controller Mappings & Virtual States", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar);

    if (ImGui::BeginTabBar("ControllersTabBar"))
    {
        if (ImGui::BeginTabItem("Keyboard"))
        {
            activeController = ICameraController::Keyboard;
            ImGui::Separator();

            if (ImGui::Button("Add key", ImVec2(100, 30)))
                addMode = !addMode;

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
                std::string keyString(1, ui::keyCodeToChar(keyboardCode, true));
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", keyString.c_str());

                ImGui::TableSetColumnIndex(2);
                bool isActive = (hash.event.magnitude > 0);
                ImVec4 statusColor = isActive ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                ImGui::TextColored(statusColor, "%s", isActive ? "Active" : "Inactive");

                ImGui::TableSetColumnIndex(3);
                std::array<char, 16> deltaText;
                snprintf(deltaText.data(), deltaText.size(), "%.2f", hash.event.magnitude);
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", deltaText.data());

                ImGui::TableSetColumnIndex(4);
                if (ImGui::Button(("Delete##deleteKey" + std::to_string(static_cast<int>(hash.event.type))).c_str()))
                {
                    controller->updateKeyboardMapping([&](auto& keys) { keys.erase(keyboardCode); });
                    pendingChanges = true;
                    break;
                }
            }
            ImGui::EndTable();

            if (addMode)
            {
                ImGui::Separator();
                handleAddMapping("AddKeyboardMappingTable", controller, activeController, selectedEventType, newKey, newMouseCode, addMode);
            }

            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("Mouse"))
        {
            activeController = ICameraController::Mouse;
            ImGui::Separator();

            if (ImGui::Button("Add key", ImVec2(100, 30)))
                addMode = !addMode;

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
                ImVec4 statusColor = isActive ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
                ImGui::TextColored(statusColor, "%s", isActive ? "Active" : "Inactive");

                ImGui::TableSetColumnIndex(3);
                std::array<char, 16> deltaText;
                snprintf(deltaText.data(), deltaText.size(), "%.2f", hash.event.magnitude);
                ImGui::AlignTextToFramePadding();
                ImGui::TextWrapped("%s", deltaText.data());

                ImGui::TableSetColumnIndex(4);
                if (ImGui::Button(("Delete##deleteMouse" + std::to_string(static_cast<int>(hash.event.type))).c_str()))
                {
                    controller->updateMouseMapping([&](auto& mouse) { mouse.erase(mouseCode); });
                    pendingChanges = true;
                    break;
                }
            }
            ImGui::EndTable();

            if (addMode)
            {
                ImGui::Separator();
                handleAddMapping("AddMouseMappingTable", controller, activeController, selectedEventType, newKey, newMouseCode, addMode);
            }
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

#endif // __NBL_KEYSMAPPING_H_INCLUDED__