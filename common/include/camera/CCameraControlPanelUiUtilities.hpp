// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_CONTROL_PANEL_UI_UTILITIES_HPP_
#define _C_CAMERA_CONTROL_PANEL_UI_UTILITIES_HPP_

#include <algorithm>
#include <array>
#include <span>
#include <string>
#include <string_view>

#include "imgui/imgui.h"
#include "imgui/misc/cpp/imgui_stdlib.h"

namespace nbl::ui
{

//! Shared visual theme and layout constants for the control panel consumer UI.
struct SCameraControlPanelStyle final
{
    static constexpr float MillisecondsPerSecond = 1000.0f;
    static constexpr float WindowWidthRatio = 0.19f;
    static constexpr float WindowMinWidth = 200.0f;
    static constexpr float WindowMaxWidthRatio = 0.25f;
    static constexpr float WindowHeightRatio = 0.34f;
    static constexpr float WindowMinHeight = 200.0f;
    static constexpr float WindowMaxHeightRatio = 0.50f;

    static constexpr ImVec2 WindowPadding = ImVec2(5.0f, 4.0f);
    static constexpr ImVec2 FramePadding = ImVec2(4.0f, 1.0f);
    static constexpr ImVec2 ItemSpacing = ImVec2(3.0f, 2.0f);
    static constexpr ImVec2 CellPadding = ImVec2(3.0f, 2.0f);
    static constexpr float WindowRounding = 4.0f;
    static constexpr float FrameRounding = 3.0f;
    static constexpr float TabRounding = 3.0f;
    static constexpr float ScrollbarRounding = 4.0f;
    static constexpr float WindowBorderSize = 1.0f;

    static constexpr ImVec4 WindowBgColor = ImVec4(0.05f, 0.06f, 0.08f, 0.0f);
    static constexpr ImVec4 ChildBgColor = ImVec4(0.10f, 0.12f, 0.16f, 0.44f);
    static constexpr ImVec4 BorderColor = ImVec4(0.64f, 0.72f, 0.84f, 0.55f);
    static constexpr ImVec4 FrameBgColor = ImVec4(0.16f, 0.19f, 0.24f, 0.54f);
    static constexpr ImVec4 FrameBgHoveredColor = ImVec4(0.26f, 0.32f, 0.40f, 0.64f);
    static constexpr ImVec4 FrameBgActiveColor = ImVec4(0.30f, 0.36f, 0.45f, 0.70f);
    static constexpr ImVec4 HeaderColor = ImVec4(0.14f, 0.18f, 0.24f, 0.60f);
    static constexpr ImVec4 HeaderHoveredColor = ImVec4(0.24f, 0.30f, 0.40f, 0.70f);
    static constexpr ImVec4 HeaderActiveColor = ImVec4(0.28f, 0.36f, 0.46f, 0.78f);
    static constexpr ImVec4 TabColor = ImVec4(0.14f, 0.18f, 0.24f, 0.60f);
    static constexpr ImVec4 TabHoveredColor = ImVec4(0.24f, 0.30f, 0.40f, 0.70f);
    static constexpr ImVec4 TabActiveColor = ImVec4(0.20f, 0.26f, 0.36f, 0.78f);
    static constexpr ImVec4 TableRowBgColor = ImVec4(0.12f, 0.14f, 0.18f, 0.50f);
    static constexpr ImVec4 TableRowAltBgColor = ImVec4(0.16f, 0.18f, 0.22f, 0.50f);
    static constexpr ImVec4 TextColor = ImVec4(0.98f, 0.99f, 1.0f, 1.0f);
    static constexpr ImVec4 TextDisabledColor = ImVec4(0.82f, 0.86f, 0.90f, 1.0f);
    static constexpr ImVec4 SeparatorColor = ImVec4(0.54f, 0.60f, 0.70f, 0.80f);
    static constexpr ImVec4 SeparatorHoveredColor = ImVec4(0.68f, 0.76f, 0.88f, 0.90f);
    static constexpr ImVec4 SeparatorActiveColor = ImVec4(0.82f, 0.90f, 1.0f, 0.96f);

    static constexpr ImVec4 AccentColor = ImVec4(0.60f, 0.82f, 1.0f, 1.0f);
    static constexpr ImVec4 GoodColor = ImVec4(0.45f, 0.90f, 0.60f, 1.0f);
    static constexpr ImVec4 BadColor = ImVec4(1.0f, 0.50f, 0.45f, 1.0f);
    static constexpr ImVec4 WarnColor = ImVec4(0.95f, 0.80f, 0.45f, 1.0f);
    static constexpr ImVec4 MutedColor = ImVec4(0.92f, 0.93f, 0.95f, 1.0f);
    static constexpr ImVec4 BadgeTextColor = ImVec4(0.10f, 0.11f, 0.13f, 1.0f);
    static constexpr ImVec4 KeyBackgroundColor = ImVec4(0.20f, 0.22f, 0.25f, 1.0f);
    static constexpr ImVec4 KeyTextColor = ImVec4(0.92f, 0.94f, 0.96f, 1.0f);
    static constexpr ImVec4 InactiveBadgeColor = ImVec4(0.35f, 0.36f, 0.38f, 1.0f);

    static constexpr ImVec4 PanelBackgroundColor = ImVec4(0.03f, 0.04f, 0.05f, 0.50f);
    static constexpr ImVec4 PanelEdgeColor = ImVec4(0.62f, 0.70f, 0.84f, 0.60f);
    static constexpr ImVec4 PanelStripeColor = ImVec4(0.28f, 0.56f, 0.90f, 0.70f);
    static constexpr ImVec4 PanelShadowColor = ImVec4(0.0f, 0.0f, 0.0f, 0.12f);
    static constexpr ImVec4 CardTopColor = ImVec4(0.20f, 0.22f, 0.26f, 0.98f);
    static constexpr ImVec4 CardBottomColor = ImVec4(0.12f, 0.13f, 0.15f, 0.98f);
    static constexpr ImVec4 CardBorderColor = ImVec4(0.45f, 0.48f, 0.54f, 1.0f);
    static constexpr ImVec4 SectionChildBackgroundColor = ImVec4(0.14f, 0.18f, 0.22f, 0.52f);
    static constexpr ImVec4 MiniStatChildBackgroundColor = ImVec4(0.14f, 0.16f, 0.19f, 0.75f);

    static constexpr ImVec2 BadgePadding = ImVec2(6.0f, 2.0f);
    static constexpr ImVec2 KeyHintPadding = ImVec2(4.0f, 1.0f);
    static constexpr float BadgeFramePaddingX = 6.0f;
    static constexpr float BadgeFramePaddingY = 2.0f;
    static constexpr float KeyHintFramePaddingX = 4.0f;
    static constexpr float KeyHintFramePaddingY = 1.0f;
    static constexpr float DotRadius = 3.5f;
    static constexpr float DotYOffset = 1.0f;
    static constexpr float DotSpacing = 6.0f;
    static constexpr float SectionChildRounding = 4.0f;
    static constexpr float CardChildRounding = 6.0f;
    static constexpr ImVec2 CardWindowPadding = ImVec2(10.0f, 8.0f);
    static constexpr float PanelShadowOffsetX = 2.0f;
    static constexpr float PanelShadowOffsetY = 3.0f;
    static constexpr float PanelShadowExtentX = 4.0f;
    static constexpr float PanelShadowExtentY = 5.0f;
    static constexpr float PanelStripeWidth = 4.0f;
    static constexpr float PanelShadowRounding = 8.0f;
    static constexpr float PanelRounding = 6.0f;
    static constexpr float SectionHeaderWidth = 2.0f;
    static constexpr float SectionHeaderTextOffsetX = 8.0f;
    static constexpr float SectionHeaderHeight = 20.0f;
    static constexpr float SectionSpacingY = 0.0f;
    static constexpr float CardExtraRows = 1.0f;
    static constexpr float CardHeightPadding = 10.0f;
    static constexpr float MiniStatHeight = 56.0f;
    static constexpr float MiniStatPlotHeight = 24.0f;
    static constexpr float MiniStatChildRounding = 6.0f;
    static constexpr float HeaderWindowHeight = 64.0f;
    static constexpr float HeaderTitleFontScale = 1.08f;
    static constexpr float HeaderMetricFontScale = 1.05f;
    static constexpr float HeaderDummyY = 1.0f;
    static constexpr float HeaderGapSmall = 2.0f;
    static constexpr float TabChildRounding = 4.0f;
    static constexpr ImVec2 TogglePadding = ImVec2(6.0f, 2.0f);
    static constexpr float KeyframeListHeight = 120.0f;
    static constexpr float EventLogBottomThreshold = 5.0f;

    static constexpr float DefaultFrameMetricMin = 16.0f;
    static constexpr float DefaultEventMetricMin = 4.0f;

    static constexpr ImGuiTableFlags SummaryTableFlags = ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg | ImGuiTableFlags_PadOuterX;
    static constexpr float SummaryLabelColumnWidth = 120.0f;
};

struct SCameraControlPanelBadgeData final
{
    const char* label = "";
    ImVec4 background = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
};

struct SCameraControlPanelKeyHintGroup final
{
    const char* label = "";
    std::span<const char* const> keys = {};
};

struct SCameraControlPanelMiniStatSpec final
{
    const char* id = "";
    const char* label = "";
    ImVec4 color = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    float minValue = 0.0f;
};

struct SCameraControlPanelCheckboxSpec final
{
    const char* label = "";
    bool* value = nullptr;
    const char* hint = "";
};

struct SCameraControlPanelSliderSpec final
{
    const char* label = "";
    float* value = nullptr;
    float minValue = 0.0f;
    float maxValue = 0.0f;
    const char* format = "%.3f";
    ImGuiSliderFlags flags = ImGuiSliderFlags_None;
    const char* hint = "";
};

struct SCameraControlPanelStatusLineSpec final
{
    const char* label = "";
    std::string_view value = {};
    ImVec4 dotColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    ImVec4 valueColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
};

struct SCameraControlPanelPolicyStatusSpec final
{
    const char* label = "";
    std::string_view value = {};
    bool active = false;
};

struct SCameraControlPanelHeaderHints final
{
    static inline constexpr std::array<const char*, 4u> MoveKeys = { "W", "A", "S", "D" };
    static inline constexpr std::array<const char*, 1u> LookKeys = { "RMB" };
    static inline constexpr std::array<const char*, 1u> ZoomKeys = { "MW" };
};

struct SCameraControlPanelToggleLabels final
{
    static inline constexpr std::array<const char*, 3u> Labels = { "WINDOW", "STATUS", "EVENT LOG" };
};

struct CCameraControlPanelUiUtilities final
{
    template<typename DrawValueFn>
    static inline void drawSummaryRow(const char* label, DrawValueFn&& drawValue)
    {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted(label);
        ImGui::TableSetColumnIndex(1);
        drawValue();
    }

    static inline void drawDot(const ImVec4& color, const SCameraControlPanelStyle& style = {})
    {
        const ImVec2 cursor = ImGui::GetCursorScreenPos();
        ImGui::GetWindowDrawList()->AddCircleFilled(
            ImVec2(cursor.x + style.DotRadius, cursor.y + style.DotRadius + style.DotYOffset),
            style.DotRadius,
            ImGui::ColorConvertFloat4ToU32(color));
        ImGui::Dummy(ImVec2(style.DotRadius * 2.0f + style.SectionHeaderWidth, style.DotRadius * 2.0f));
        ImGui::SameLine(0.0f, style.DotSpacing);
    }

    static inline void drawStatusLine(const SCameraControlPanelStatusLineSpec& spec, const SCameraControlPanelStyle& style = {})
    {
        drawSummaryRow(spec.label, [&]()
        {
            drawDot(spec.dotColor, style);
            ImGui::TextColored(spec.valueColor, "%.*s", static_cast<int>(spec.value.size()), spec.value.data());
        });
    }

    static inline void drawPolicyStatus(const SCameraControlPanelPolicyStatusSpec& spec, const SCameraControlPanelStyle& style = {})
    {
        ImGui::TextDisabled("%s", spec.label);
        ImGui::SameLine();
        ImGui::TextColored(spec.active ? style.GoodColor : style.BadColor, "%.*s", static_cast<int>(spec.value.size()), spec.value.data());
    }

    static inline ImVec2 calcControlPanelWindowSize(const ImVec2& displaySize, const SCameraControlPanelStyle& style = {})
    {
        return ImVec2(
            std::clamp(displaySize.x * style.WindowWidthRatio, style.WindowMinWidth, displaySize.x * style.WindowMaxWidthRatio),
            std::clamp(displaySize.y * style.WindowHeightRatio, style.WindowMinHeight, displaySize.y * style.WindowMaxHeightRatio));
    }

    static inline float calcFramesPerSecond(const float frameMs, const SCameraControlPanelStyle& style = {})
    {
        return frameMs > 0.0f ? (style.MillisecondsPerSecond / frameMs) : 0.0f;
    }

    static inline float calcPillWidth(const char* label, const ImVec2& padding)
    {
        return ImGui::CalcTextSize(label).x + padding.x * 2.0f;
    }

    static inline void centerControlPanelRow(const float contentWidth)
    {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, (ImGui::GetContentRegionAvail().x - contentWidth) * 0.5f));
    }

    static inline void pushControlPanelWindowStyle(const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, style.WindowPadding);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, style.FramePadding);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, style.ItemSpacing);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, style.WindowRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, style.FrameRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_TabRounding, style.TabRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarRounding, style.ScrollbarRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, style.WindowBorderSize);
        ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, style.CellPadding);

        ImGui::PushStyleColor(ImGuiCol_WindowBg, style.WindowBgColor);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, style.ChildBgColor);
        ImGui::PushStyleColor(ImGuiCol_Border, style.BorderColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, style.FrameBgColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, style.FrameBgHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, style.FrameBgActiveColor);
        ImGui::PushStyleColor(ImGuiCol_Header, style.HeaderColor);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, style.HeaderHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, style.HeaderActiveColor);
        ImGui::PushStyleColor(ImGuiCol_Tab, style.TabColor);
        ImGui::PushStyleColor(ImGuiCol_TabHovered, style.TabHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_TabActive, style.TabActiveColor);
        ImGui::PushStyleColor(ImGuiCol_TableRowBg, style.TableRowBgColor);
        ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, style.TableRowAltBgColor);
        ImGui::PushStyleColor(ImGuiCol_Text, style.TextColor);
        ImGui::PushStyleColor(ImGuiCol_TextDisabled, style.TextDisabledColor);
        ImGui::PushStyleColor(ImGuiCol_Separator, style.SeparatorColor);
        ImGui::PushStyleColor(ImGuiCol_SeparatorHovered, style.SeparatorHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_SeparatorActive, style.SeparatorActiveColor);
    }

    static inline void popControlPanelWindowStyle()
    {
        ImGui::PopStyleColor(19);
        ImGui::PopStyleVar(9);
    }

    static inline bool inputTextString(
        const char* label,
        std::string& value,
        ImGuiInputTextFlags flags = 0)
    {
        return ImGui::InputText(label, &value, flags);
    }

    static inline void drawControlPanelWindowBackdrop(ImDrawList& drawList, const ImVec2& panelPos, const ImVec2& panelSize, const SCameraControlPanelStyle& style = {})
    {
        const ImVec2 panelMax(panelPos.x + panelSize.x, panelPos.y + panelSize.y);
        drawList.AddRectFilled(
            ImVec2(panelPos.x + style.PanelShadowOffsetX, panelPos.y + style.PanelShadowOffsetY),
            ImVec2(panelPos.x + panelSize.x + style.PanelShadowExtentX, panelPos.y + panelSize.y + style.PanelShadowExtentY),
            ImGui::ColorConvertFloat4ToU32(style.PanelShadowColor),
            style.PanelShadowRounding);
        drawList.AddRectFilled(panelPos, panelMax, ImGui::ColorConvertFloat4ToU32(style.PanelBackgroundColor), style.PanelRounding);
        drawList.AddRect(panelPos, panelMax, ImGui::ColorConvertFloat4ToU32(style.PanelEdgeColor), style.PanelRounding);
        drawList.AddRectFilled(
            panelPos,
            ImVec2(panelPos.x + style.PanelStripeWidth, panelPos.y + panelSize.y),
            ImGui::ColorConvertFloat4ToU32(style.PanelStripeColor),
            style.PanelRounding);
    }

    static inline float calcCameraControlPanelCardHeight(const int rows, const SCameraControlPanelStyle& style = {})
    {
        return ImGui::GetFrameHeightWithSpacing() * (static_cast<float>(rows) + style.CardExtraRows) + style.CardHeightPadding;
    }

    static inline void drawBadge(const char* label, const ImVec4& bg, const ImVec4& fg, const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleColor(ImGuiCol_Button, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg);
        ImGui::PushStyleColor(ImGuiCol_Text, fg);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.BadgeFramePaddingX, style.BadgeFramePaddingY));
        ImGui::Button(label);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
    }

    static inline float calcBadgeRowWidth(
        const std::span<const SCameraControlPanelBadgeData> badges,
        const float gap,
        const ImVec2& badgePadding)
    {
        float width = 0.0f;
        for (size_t i = 0; i < badges.size(); ++i)
        {
            if (i > 0u)
                width += gap;
            width += calcPillWidth(badges[i].label, badgePadding);
        }
        return width;
    }

    static inline void drawBadgeRow(
        const std::span<const SCameraControlPanelBadgeData> badges,
        const ImVec4& textColor,
        const float gap,
        const SCameraControlPanelStyle& style = {})
    {
        if (badges.empty())
            return;

        centerControlPanelRow(calcBadgeRowWidth(badges, gap, style.BadgePadding));
        for (size_t i = 0; i < badges.size(); ++i)
        {
            if (i > 0u)
                ImGui::SameLine(0.0f, gap);
            drawBadge(badges[i].label, badges[i].background, textColor, style);
        }
    }

    static inline void drawKeyHint(const char* label, const ImVec4& bg, const ImVec4& fg, const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleColor(ImGuiCol_Button, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg);
        ImGui::PushStyleColor(ImGuiCol_Text, fg);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.KeyHintFramePaddingX, style.KeyHintFramePaddingY));
        ImGui::SmallButton(label);
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
    }

    static inline float calcKeyHintGroupWidth(
        const SCameraControlPanelKeyHintGroup& group,
        const float gap,
        const ImVec2& keyPadding)
    {
        float width = ImGui::CalcTextSize(group.label).x;
        for (const char* key : group.keys)
            width += gap + calcPillWidth(key, keyPadding);
        return width;
    }

    static inline void drawKeyHintGroup(
        const SCameraControlPanelKeyHintGroup& group,
        const float gap,
        const ImVec4& keyBackground,
        const ImVec4& keyText,
        const SCameraControlPanelStyle& style = {})
    {
        ImGui::TextDisabled("%s", group.label);
        for (const char* key : group.keys)
        {
            ImGui::SameLine(0.0f, gap);
            drawKeyHint(key, keyBackground, keyText, style);
        }
    }

    static inline void drawKeyHintGroupRow(
        const std::span<const SCameraControlPanelKeyHintGroup> groups,
        const float gap,
        const float groupGap,
        const ImVec4& keyBackground,
        const ImVec4& keyText,
        const SCameraControlPanelStyle& style = {})
    {
        float rowWidth = 0.0f;
        for (size_t i = 0; i < groups.size(); ++i)
        {
            if (i > 0u)
                rowWidth += groupGap;
            rowWidth += calcKeyHintGroupWidth(groups[i], gap, style.KeyHintPadding);
        }

        centerControlPanelRow(rowWidth);
        for (size_t i = 0; i < groups.size(); ++i)
        {
            if (i > 0u)
                ImGui::SameLine(0.0f, groupGap);
            drawKeyHintGroup(groups[i], gap, keyBackground, keyText, style);
        }
    }

    static inline void drawTogglePill(
        const char* label,
        bool& value,
        const ImVec4& onColor,
        const ImVec4& offColor,
        const ImVec4& textColor,
        const ImVec2& padding)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, value ? onColor : offColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, value ? onColor : offColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, value ? onColor : offColor);
        ImGui::PushStyleColor(ImGuiCol_Text, textColor);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding);
        if (ImGui::Button(label))
            value = !value;
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(4);
    }

    template<size_t SampleCount, typename DrawValueFn>
    static inline void drawMiniStat(
        const SCameraControlPanelMiniStatSpec& stat,
        const std::array<float, SampleCount>& series,
        const size_t metricIndex,
        DrawValueFn&& drawValue,
        const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, style.MiniStatChildBackgroundColor);
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.MiniStatChildRounding);
        if (ImGui::BeginChild(stat.id, ImVec2(0.0f, style.MiniStatHeight), true, ImGuiWindowFlags_NoScrollbar))
        {
            ImGui::TextDisabled("%s", stat.label);
            ImGui::SetWindowFontScale(style.HeaderMetricFontScale);
            drawValue();
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PushStyleColor(ImGuiCol_PlotLines, stat.color);
            float maxValue = stat.minValue;
            for (const float value : series)
                maxValue = std::max(maxValue, value);
            ImGui::PlotLines("##plot", series.data(), static_cast<int>(SampleCount), static_cast<int>(metricIndex), nullptr, 0.0f, maxValue, ImVec2(0.0f, style.MiniStatPlotHeight));
            ImGui::PopStyleColor();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
    }

    static inline void drawHoverHint(const char* text)
    {
        if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            return;
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(text);
        ImGui::EndTooltip();
    }

    static inline bool drawCheckboxWithHint(const SCameraControlPanelCheckboxSpec& spec)
    {
        if (!spec.value)
            return false;

        const bool changed = ImGui::Checkbox(spec.label, spec.value);
        if (spec.hint && spec.hint[0] != '\0')
            drawHoverHint(spec.hint);
        return changed;
    }

    static inline bool drawSliderFloatWithHint(const SCameraControlPanelSliderSpec& spec)
    {
        if (!spec.value)
            return false;

        const bool changed = ImGui::SliderFloat(spec.label, spec.value, spec.minValue, spec.maxValue, spec.format, spec.flags);
        if (spec.hint && spec.hint[0] != '\0')
            drawHoverHint(spec.hint);
        return changed;
    }

    static inline bool drawActionButtonWithHint(const char* label, const char* hint)
    {
        const bool pressed = ImGui::Button(label);
        if (hint && hint[0] != '\0')
            drawHoverHint(hint);
        return pressed;
    }

    static inline bool beginControlPanelTabChild(const char* id, const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.TabChildRounding);
        return ImGui::BeginChild(id, ImVec2(0.0f, 0.0f), true);
    }

    static inline void endControlPanelTabChild()
    {
        ImGui::EndChild();
        ImGui::PopStyleVar();
    }

    static inline void drawSectionHeader(const char* id, const char* label, const ImVec4& accent, const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.SectionChildRounding);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, style.SectionChildBackgroundColor);
        if (ImGui::BeginChild(id, ImVec2(0.0f, style.SectionHeaderHeight), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
        {
            const ImVec2 pos = ImGui::GetWindowPos();
            const ImVec2 size = ImGui::GetWindowSize();
            ImGui::GetWindowDrawList()->AddRectFilled(
                pos,
                ImVec2(pos.x + style.SectionHeaderWidth, pos.y + size.y),
                ImGui::ColorConvertFloat4ToU32(accent),
                style.SectionChildRounding);
            ImGui::SetCursorPosX(style.SectionHeaderTextOffsetX);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(accent, "%s", label);
        }
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
        ImGui::Spacing();
    }

    static inline bool beginCard(const char* id, const float height, const ImVec4& top, const ImVec4& bottom, const ImVec4& border, const SCameraControlPanelStyle& style = {})
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, style.CardChildRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, style.CardWindowPadding);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        const bool open = ImGui::BeginChild(id, ImVec2(0.0f, height), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        const ImVec2 pos = ImGui::GetWindowPos();
        const ImVec2 size = ImGui::GetWindowSize();
        ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
            pos,
            ImVec2(pos.x + size.x, pos.y + size.y),
            ImGui::ColorConvertFloat4ToU32(top),
            ImGui::ColorConvertFloat4ToU32(top),
            ImGui::ColorConvertFloat4ToU32(bottom),
            ImGui::ColorConvertFloat4ToU32(bottom));
        ImGui::GetWindowDrawList()->AddRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), ImGui::ColorConvertFloat4ToU32(border), style.CardChildRounding);
        return open;
    }

    static inline void endCard()
    {
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(2);
    }
};

} // namespace nbl::ui

#endif // _C_CAMERA_CONTROL_PANEL_UI_UTILITIES_HPP_
