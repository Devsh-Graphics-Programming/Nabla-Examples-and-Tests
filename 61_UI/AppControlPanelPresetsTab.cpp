#include "app/App.hpp"

#include <algorithm>
#include <vector>

#include "app/AppControlPanelAuthoringUtilities.hpp"

void App::drawControlPanelPresetsTab(const nbl::ui::SCameraControlPanelStyle& panelStyle)
{
    auto& presetAuthoring = m_presetAuthoring;

    if (!nbl::ui::CCameraControlPanelUiUtilities::beginControlPanelTabChild("PresetsPanel", panelStyle))
    {
        nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
        return;
    }

    ImGui::PushItemWidth(-1.0f);
    nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("PresetsHeader", "Presets", panelStyle.AccentColor, panelStyle);
    nbl::ui::CCameraControlPanelUiUtilities::inputTextString("Preset name", presetAuthoring.presetName);
    auto* activeCamera = getActiveCamera();
    const auto presetCaptureUi = analyzeCameraCaptureForUi(activeCamera);
    if (!presetCaptureUi.canCapture)
        ImGui::BeginDisabled();
    if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Add preset", presetCaptureUi.canCapture ? "Store current camera as a preset" : "Preset capture is blocked because there is no active camera or the current goal state is invalid"))
    {
        CameraPreset preset;
        if (nbl::core::CCameraPresetFlowUtilities::tryCapturePreset(m_cameraGoalSolver, activeCamera, presetAuthoring.presetName, preset))
        {
            presetAuthoring.presets.emplace_back(std::move(preset));
            presetAuthoring.selectedPresetIx = static_cast<int>(presetAuthoring.presets.size()) - 1;
        }
    }
    if (!presetCaptureUi.canCapture)
        ImGui::EndDisabled();
    ImGui::SameLine();
    if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Clear presets", "Remove all presets"))
    {
        presetAuthoring.presets.clear();
        presetAuthoring.selectedPresetIx = -1;
    }
    nbl::ui::CCameraControlPanelUiUtilities::drawPolicyStatus({
        .label = "Capture",
        .value = presetCaptureUi.policyLabel,
        .active = presetCaptureUi.canCapture
    }, panelStyle);

    if (!presetAuthoring.presets.empty())
    {
        const char* presetFilterLabels[] = {
            nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(PresetFilterMode::All),
            nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(PresetFilterMode::Exact),
            nbl::ui::CCameraPresentationUtilities::getPresetApplyPresentationFilterLabel(PresetFilterMode::BestEffort)
        };
        int presetFilterIx = static_cast<int>(presetAuthoring.filterMode);
        if (ImGui::Combo("Visibility", &presetFilterIx, presetFilterLabels, IM_ARRAYSIZE(presetFilterLabels)))
            presetAuthoring.filterMode = static_cast<PresetFilterMode>(presetFilterIx);
        nbl::ui::CCameraControlPanelUiUtilities::drawHoverHint("Filter presets for the active camera using exact or best-effort compatibility");

        std::vector<int> filteredPresetIndices;
        filteredPresetIndices.reserve(presetAuthoring.presets.size());
        for (size_t i = 0; i < presetAuthoring.presets.size(); ++i)
        {
            if (presetMatchesFilter(activeCamera, presetAuthoring.presets[i]))
                filteredPresetIndices.push_back(static_cast<int>(i));
        }

        if (filteredPresetIndices.empty())
        {
            ImGui::TextDisabled("No presets match the current filter.");
        }
        else
        {
            if (presetAuthoring.selectedPresetIx < 0 || std::find(filteredPresetIndices.begin(), filteredPresetIndices.end(), presetAuthoring.selectedPresetIx) == filteredPresetIndices.end())
                presetAuthoring.selectedPresetIx = filteredPresetIndices.front();

            int selectedFilteredPresetIx = 0;
            for (int i = 0; i < static_cast<int>(filteredPresetIndices.size()); ++i)
            {
                if (filteredPresetIndices[i] == presetAuthoring.selectedPresetIx)
                {
                    selectedFilteredPresetIx = i;
                    break;
                }
            }

            if (ImGui::BeginListBox("Preset list", ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing() * SCameraAppAuthoringDefaults::PresetListVisibleEntries)))
            {
                for (int i = 0; i < static_cast<int>(filteredPresetIndices.size()); ++i)
                {
                    const int presetIx = filteredPresetIndices[static_cast<size_t>(i)];
                    const bool isSelected = selectedFilteredPresetIx == i;
                    const auto& presetName = presetAuthoring.presets[static_cast<size_t>(presetIx)].name;
                    if (ImGui::Selectable(presetName.c_str(), isSelected))
                    {
                        selectedFilteredPresetIx = i;
                        presetAuthoring.selectedPresetIx = presetIx;
                    }

                    if (isSelected)
                        ImGui::SetItemDefaultFocus();
                }

                ImGui::EndListBox();
            }

            if (presetAuthoring.selectedPresetIx >= 0 && static_cast<size_t>(presetAuthoring.selectedPresetIx) < presetAuthoring.presets.size())
            {
                const auto& preset = presetAuthoring.presets[static_cast<size_t>(presetAuthoring.selectedPresetIx)];
                const auto presetUi = analyzePresetForUi(activeCamera, preset);
                nbl::ui::drawGoalApplyPresentationSummary(presetUi, panelStyle);

                if (!presetUi.canApply)
                    ImGui::BeginDisabled();
                if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Apply preset", presetUi.canApply ? "Apply selected preset to the active camera" : "Apply is blocked because there is no active camera or the preset goal is invalid"))
                    applyPresetFromUi(activeCamera, preset);
                if (!presetUi.canApply)
                    ImGui::EndDisabled();
                ImGui::SameLine();
                if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Remove preset", "Remove selected preset"))
                {
                    presetAuthoring.presets.erase(presetAuthoring.presets.begin() + presetAuthoring.selectedPresetIx);
                    presetAuthoring.selectedPresetIx = -1;
                }
            }
        }
    }

    nbl::ui::drawApplyStatusBanner(
        presetAuthoring.applyBanner.summary,
        presetAuthoring.applyBanner.succeeded,
        presetAuthoring.applyBanner.approximate,
        panelStyle);

    nbl::ui::CCameraControlPanelUiUtilities::drawSectionHeader("PresetsStorageHeader", "Storage", panelStyle.AccentColor, panelStyle);
    nbl::ui::CCameraControlPanelUiUtilities::inputTextString("Preset file", presetAuthoring.presetPath);
    if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Save presets", "Save presets to JSON file"))
    {
        if (!savePresetsToFile(nbl::system::path(presetAuthoring.presetPath)))
            m_logger->log("Failed to save presets to \"%s\".", ILogger::ELL_ERROR, presetAuthoring.presetPath.c_str());
    }
    ImGui::SameLine();
    if (nbl::ui::CCameraControlPanelUiUtilities::drawActionButtonWithHint("Load presets", "Load presets from JSON file"))
    {
        if (!loadPresetsFromFile(nbl::system::path(presetAuthoring.presetPath)))
            m_logger->log("Failed to load presets from \"%s\".", ILogger::ELL_ERROR, presetAuthoring.presetPath.c_str());
    }

    ImGui::PopItemWidth();
    nbl::ui::CCameraControlPanelUiUtilities::endControlPanelTabChild();
}
