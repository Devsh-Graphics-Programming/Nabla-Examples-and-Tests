#ifndef __NBL_KEYSMAPPING_H_INCLUDED__
#define __NBL_KEYSMAPPING_H_INCLUDED__

#include "common.hpp"

template<typename T = matrix_precision_t>
void displayKeyMappingsAndVirtualStates(ICamera<T>* camera)
{
    static bool addMode = false;
    static bool pendingChanges = false;
    static std::unordered_map<CVirtualGimbalEvent::VirtualEventType, std::vector<ui::E_KEY_CODE>> tempKeyMappings;
    static CVirtualGimbalEvent::VirtualEventType selectedEventType = CVirtualGimbalEvent::VirtualEventType::MoveForward;
    static ui::E_KEY_CODE newKey = ui::E_KEY_CODE::EKC_A;

    const uint32_t allowedEventsMask = camera->getAllowedVirtualEvents();

    std::vector<CVirtualGimbalEvent::VirtualEventType> allowedEvents;
    for (const auto& eventType : CVirtualGimbalEvent::VirtualEventsTypeTable)
        if ((eventType & allowedEventsMask))
            allowedEvents.push_back(eventType);

    ImGui::SetNextWindowSize(ImVec2(600, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(600, 400), ImVec2(600, 69000));

    ImGui::Begin("Key Mappings & Virtual States", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_AlwaysVerticalScrollbar);

    ImVec2 windowPadding = ImGui::GetStyle().WindowPadding;
    float verticalPadding = ImGui::GetStyle().FramePadding.y;

    if (ImGui::Button("Add key", ImVec2(100, 30)))
        addMode = !addMode;

    ImGui::Separator();

    ImGui::BeginTable("KeyMappingsTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
    ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Key(s)", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Active Status", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Delta Time (ms)", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableHeadersRow();

    const auto& keysToVirtualEvents = camera->getKeysToVirtualEvents();
    for (const auto& eventType : allowedEvents)
    {
        auto it = std::find_if(keysToVirtualEvents.begin(), keysToVirtualEvents.end(), [eventType](const auto& pair) 
        {
            return pair.second.type == eventType;
        });

        if (it != keysToVirtualEvents.end())
        {
            ImGui::TableNextRow();

            const char* eventName = CVirtualGimbalEvent::virtualEventToString(eventType).data();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - ImGui::CalcTextSize(eventName).x) * 0.5f);
            ImGui::TextWrapped("%s", eventName);

            ImGui::TableSetColumnIndex(1);
            std::vector<ui::E_KEY_CODE> mappedKeys;
            for (const auto& [key, info] : keysToVirtualEvents)
                if (info.type == eventType)
                    mappedKeys.push_back(key);

            if (!mappedKeys.empty())
            {
                std::string concatenatedKeys;
                for (size_t i = 0; i < mappedKeys.size(); ++i)
                {
                    if (i > 0)
                        concatenatedKeys += " | ";
                    if (keysToVirtualEvents.at(mappedKeys[i]).active)
                        concatenatedKeys += "[" + std::string(1, ui::keyCodeToChar(mappedKeys[i], true)) + "]";
                    else
                        concatenatedKeys += std::string(1, ui::keyCodeToChar(mappedKeys[i], true));
                }
                ImGui::AlignTextToFramePadding();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - ImGui::CalcTextSize(concatenatedKeys.c_str()).x) * 0.5f);
                ImGui::TextWrapped("%s", concatenatedKeys.c_str());
            }

            ImGui::TableSetColumnIndex(2);
            ImVec4 statusColor = it->second.active ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
            const char* statusText = it->second.active ? "Active" : "Inactive";
            ImGui::AlignTextToFramePadding();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - ImGui::CalcTextSize(statusText).x) * 0.5f);
            ImGui::TextColored(statusColor, "%s", statusText);

            ImGui::TableSetColumnIndex(3);
            float accumulatedDelta = 0.0f;
            for (const auto& [key, info] : keysToVirtualEvents)
                if (info.type == eventType)
                    accumulatedDelta += info.dtAction;

            char deltaText[16];
            snprintf(deltaText, 16, "%.2f", accumulatedDelta);
            ImGui::AlignTextToFramePadding();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - ImGui::CalcTextSize(deltaText).x) * 0.5f);
            ImGui::TextWrapped("%s", deltaText);

            ImGui::TableSetColumnIndex(4);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - 80) * 0.5f);
            if (ImGui::Button(std::string("Delete##deleteKey" + std::to_string(static_cast<int>(eventType))).c_str(), ImVec2(80, 30)))
            {
                camera->updateKeysToEvent([&](auto& keys)
                {
                    for (auto it = keys.begin(); it != keys.end();)
                    {
                        if (it->second.type == eventType)
                            it = keys.erase(it);
                        else
                            ++it;
                    }
                });
                pendingChanges = true;
            }
            ImGui::PopStyleVar();
        }
    }
    ImGui::EndTable();

    if (addMode)
    {
        ImGui::Separator();

        ImGui::BeginTable("AddKeyMappingTable", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame);
        ImGui::TableSetupColumn("Virtual Event", ImGuiTableColumnFlags_WidthStretch, 0.33f);
        ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthStretch, 0.33f);
        ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthStretch, 0.33f);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::AlignTextToFramePadding();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + verticalPadding);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().FramePadding.x);
        if (ImGui::BeginCombo("##selectEvent", CVirtualGimbalEvent::virtualEventToString(selectedEventType).data(), ImGuiComboFlags_None))
        {
            for (const auto& eventType : allowedEvents)
            {
                bool isSelected = (selectedEventType == eventType);
                if (ImGui::Selectable(CVirtualGimbalEvent::virtualEventToString(eventType).data(), isSelected))
                    selectedEventType = eventType;
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(1);
        ImGui::AlignTextToFramePadding();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + verticalPadding);
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 2 * ImGui::GetStyle().FramePadding.x);
        char newKeyDisplay[2] = { ui::keyCodeToChar(newKey, true), '\0' };
        if (ImGui::BeginCombo("##selectKey", newKeyDisplay, ImGuiComboFlags_None))
        {
            for (int i = ui::E_KEY_CODE::EKC_A; i <= ui::E_KEY_CODE::EKC_Z; ++i)
            {
                bool isSelected = (newKey == static_cast<ui::E_KEY_CODE>(i));
                char label[2] = { ui::keyCodeToChar(static_cast<ui::E_KEY_CODE>(i), true), '\0' };
                if (ImGui::Selectable(label, isSelected))
                {
                    auto duplicateKey = std::find_if(tempKeyMappings[selectedEventType].begin(), tempKeyMappings[selectedEventType].end(),
                        [i](const auto& key) {
                            return key == static_cast<ui::E_KEY_CODE>(i);
                        });

                    if (duplicateKey == tempKeyMappings[selectedEventType].end())
                        newKey = static_cast<ui::E_KEY_CODE>(i);
                }
                if (isSelected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        ImGui::TableSetColumnIndex(2);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (ImGui::GetColumnWidth() - 100) * 0.5f);
        if (ImGui::Button("Confirm Add", ImVec2(100, 30)))
        {
            tempKeyMappings[selectedEventType].push_back(newKey);
            pendingChanges = true;
            addMode = false;

            camera->updateKeysToEvent([&](auto& keys)
            {
                keys.emplace(newKey, selectedEventType);
            });
        }
        ImGui::PopStyleVar();

        ImGui::EndTable();
    }

    ImGui::Dummy(ImVec2(0.0f, verticalPadding));
    ImGui::End();
}

#endif // __NBL_KEYSMAPPING_H_INCLUDED__