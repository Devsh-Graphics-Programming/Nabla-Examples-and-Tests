#include "keysmapping.hpp"
#include "app/AppTypes.hpp"

#include <array>
#include <string>

namespace
{

/// @brief Every key a binding may name, in the order the pickers list them.
const std::vector<ui::E_KEY_CODE>& getSelectableKeyCodes()
{
    static const std::vector<ui::E_KEY_CODE> codes = []()
    {
        std::vector<ui::E_KEY_CODE> out;
        out.push_back(ui::EKC_NONE);
        for (uint32_t code = 1u; code < ui::EKC_COUNT; ++code)
        {
            const auto key = static_cast<ui::E_KEY_CODE>(code);
            // a key with no stable name cannot be round-tripped through a saved binding, so it is not offered
            if (CInputCodeNames::keyCodeToString(key) != "NONE")
                out.push_back(key);
        }
        return out;
    }();
    return codes;
}

std::string getKeyLabel(const ui::E_KEY_CODE key)
{
    if (key == ui::EKC_NONE)
        return "none";
    return std::string(CInputCodeNames::keyCodeToString(key));
}

std::string getGateLabel(const std::optional<ui::E_MOUSE_BUTTON>& gate)
{
    if (!gate.has_value())
        return "always";
    return std::string(CInputCodeNames::mouseButtonToString(gate.value()));
}

bool drawKeyPicker(const char* id, ui::E_KEY_CODE& key)
{
    bool changed = false;
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##key", getKeyLabel(key).c_str()))
    {
        for (const auto candidate : getSelectableKeyCodes())
        {
            const bool selected = candidate == key;
            if (ImGui::Selectable(getKeyLabel(candidate).c_str(), selected))
            {
                key = candidate;
                changed = true;
            }
            if (selected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
    return changed;
}

bool drawGatePicker(const char* id, std::optional<ui::E_MOUSE_BUTTON>& gate)
{
    static constexpr std::array<ui::E_MOUSE_BUTTON, ui::EMB_COUNT> Buttons = {
        ui::EMB_LEFT_BUTTON, ui::EMB_RIGHT_BUTTON, ui::EMB_MIDDLE_BUTTON, ui::EMB_BUTTON_4, ui::EMB_BUTTON_5
    };

    bool changed = false;
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (ImGui::BeginCombo("##gate", getGateLabel(gate).c_str()))
    {
        if (ImGui::Selectable("always", !gate.has_value()))
        {
            gate = std::nullopt;
            changed = true;
        }
        for (const auto button : Buttons)
        {
            const bool selected = gate.has_value() && gate.value() == button;
            if (ImGui::Selectable(std::string(CInputCodeNames::mouseButtonToString(button)).c_str(), selected))
            {
                gate = button;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::PopID();
    return changed;
}

bool drawScalar(const char* id, double& value)
{
    float asFloat = static_cast<float>(value);
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool changed = ImGui::InputFloat("##value", &asFloat, 0.f, 0.f, "%.4f", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();
    if (changed)
        value = static_cast<double>(asFloat);
    return changed;
}

bool drawGain(const char* id, hlsl::float64_t2& gain)
{
    float values[2] = { static_cast<float>(gain.x), static_cast<float>(gain.y) };
    ImGui::PushID(id);
    ImGui::SetNextItemWidth(-FLT_MIN);
    const bool changed = ImGui::InputFloat2("##gain", values, "%.4f", ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::PopID();
    if (changed)
        gain = hlsl::float64_t2(static_cast<double>(values[0]), static_cast<double>(values[1]));
    return changed;
}

bool drawBindingRows(SCameraMouseKeyboardBinding& binding, const uint32_t acceptedAxes)
{
    constexpr auto TableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY;

    bool changed = false;
    if (!ImGui::BeginTable("##camera_binding", 7, TableFlags, ImVec2(0.f, 0.f)))
        return false;

    ImGui::TableSetupColumn("Axis");
    ImGui::TableSetupColumn("+ Key");
    ImGui::TableSetupColumn("- Key");
    ImGui::TableSetupColumn("Rate /s");
    ImGui::TableSetupColumn("Mouse x,y");
    ImGui::TableSetupColumn("Scroll v,h");
    ImGui::TableSetupColumn("Gate");
    ImGui::TableHeadersRow();

    for (uint32_t i = 0u; i < CameraControlAxisCount; ++i)
    {
        const auto axis = cameraControlAxisFromIndex(i);
        if ((acceptedAxes & axis) == 0u)
            continue;

        auto& slot = binding.axes[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(std::string(cameraControlAxisName(axis)).c_str());

        ImGui::TableSetColumnIndex(1);
        changed |= drawKeyPicker("positive", slot.positiveKey);

        ImGui::TableSetColumnIndex(2);
        changed |= drawKeyPicker("negative", slot.negativeKey);

        ImGui::TableSetColumnIndex(3);
        changed |= drawScalar("rate", slot.keyRate);

        ImGui::TableSetColumnIndex(4);
        changed |= drawGain("move", slot.mouseMovementGain);

        ImGui::TableSetColumnIndex(5);
        changed |= drawGain("scroll", slot.mouseScrollGain);

        ImGui::TableSetColumnIndex(6);
        changed |= drawGatePicker("gate", slot.mouseMovementGate);

        ImGui::PopID();
    }

    ImGui::EndTable();
    return changed;
}

}

bool displayCameraBindingTableInline(SCameraMouseKeyboardBinding& binding, const uint32_t acceptedAxes, const bool spawnWindow)
{
    if (!spawnWindow)
        return drawBindingRows(binding, acceptedAxes);

    bool changed = false;
    if (ImGui::Begin("Camera Controls", nullptr, ImGuiWindowFlags_NoSavedSettings))
        changed = drawBindingRows(binding, acceptedAxes);
    ImGui::End();
    return changed;
}
