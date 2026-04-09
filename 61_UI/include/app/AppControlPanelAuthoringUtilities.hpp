#ifndef _NBL_THIS_EXAMPLE_APP_CONTROL_PANEL_AUTHORING_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_CONTROL_PANEL_AUTHORING_UTILITIES_HPP_

#include <string>
#include <string_view>

#include "camera/CCameraControlPanelUiUtilities.hpp"
#include "camera/CCameraPresentationUtilities.hpp"

namespace nbl::ui
{

inline void drawApplyStatusBanner(
    const std::string_view summary,
    const bool succeeded,
    const bool approximate,
    const SCameraControlPanelStyle& panelStyle)
{
    if (summary.empty())
        return;

    const ImVec4 resultColor = succeeded ? (approximate ? panelStyle.WarnColor : panelStyle.GoodColor) : panelStyle.BadColor;
    ImGui::TextColored(resultColor, "%.*s", static_cast<int>(summary.size()), summary.data());
}

inline void drawGoalApplyPresentationBadges(const SCameraGoalApplyPresentation& presentation, const SCameraControlPanelStyle& panelStyle)
{
    if (presentation.badges.exact)
    {
        CCameraControlPanelUiUtilities::drawBadge("EXACT", panelStyle.GoodColor, panelStyle.BadgeTextColor, panelStyle);
    }
    else if (presentation.badges.bestEffort)
    {
        CCameraControlPanelUiUtilities::drawBadge("BEST-EFFORT", panelStyle.WarnColor, panelStyle.BadgeTextColor, panelStyle);
    }

    if (presentation.badges.dropsState)
    {
        ImGui::SameLine();
        CCameraControlPanelUiUtilities::drawBadge("DROPS STATE", panelStyle.WarnColor, panelStyle.BadgeTextColor, panelStyle);
    }
    else if (presentation.badges.sharedStateOnly)
    {
        ImGui::SameLine();
        CCameraControlPanelUiUtilities::drawBadge("SHARED STATE", panelStyle.AccentColor, panelStyle.BadgeTextColor, panelStyle);
    }

    if (presentation.badges.blocked)
    {
        ImGui::SameLine();
        CCameraControlPanelUiUtilities::drawBadge("BLOCKED", panelStyle.BadColor, panelStyle.BadgeTextColor, panelStyle);
    }
}

inline ImVec4 getGoalApplyPresentationColor(const SCameraGoalApplyPresentation& presentation, const SCameraControlPanelStyle& panelStyle)
{
    return !presentation.hasCamera ? panelStyle.BadColor : (presentation.exact() ? panelStyle.GoodColor : panelStyle.WarnColor);
}

inline void drawGoalApplyPresentationSummary(const SCameraGoalApplyPresentation& presentation, const SCameraControlPanelStyle& panelStyle)
{
    const ImVec4 compatibilityColor = getGoalApplyPresentationColor(presentation, panelStyle);

    ImGui::TextDisabled("Source");
    ImGui::SameLine();
    ImGui::TextColored(panelStyle.MutedColor, "%s", presentation.sourceKindLabel.c_str());
    ImGui::TextDisabled("Goal state");
    ImGui::SameLine();
    ImGui::TextColored(panelStyle.MutedColor, "%s", presentation.goalStateLabel.c_str());
    ImGui::TextDisabled("Policy");
    ImGui::SameLine();
    ImGui::TextColored(presentation.canApply ? compatibilityColor : panelStyle.BadColor, "%s", presentation.policyLabel.c_str());
    ImGui::TextDisabled("Compatibility");
    ImGui::SameLine();
    ImGui::TextColored(compatibilityColor, "%s", presentation.compatibilityLabel.c_str());
    drawGoalApplyPresentationBadges(presentation, panelStyle);
}

inline std::string buildKeyframeLabel(const size_t keyframeIx, const core::CCameraKeyframe& keyframe)
{
    return "[" + std::to_string(keyframeIx) + "] t=" + std::to_string(keyframe.time) + "  " + keyframe.preset.name;
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_CONTROL_PANEL_AUTHORING_UTILITIES_HPP_
