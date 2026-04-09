// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_VIEWPORT_OVERLAY_UTILITIES_HPP_
#define _C_CAMERA_VIEWPORT_OVERLAY_UTILITIES_HPP_

#include <algorithm>
#include <string>

#include "CCameraFollowRegressionUtilities.hpp"
#include "imgui/imgui.h"

namespace nbl::ui
{

//! Screen-space viewport rectangle used by debug overlay helpers.
struct SViewportOverlayRect final
{
    ImVec2 position = ImVec2(0.0f, 0.0f);
    ImVec2 size = ImVec2(0.0f, 0.0f);

    inline bool valid() const
    {
        return size.x > 1.0f && size.y > 1.0f;
    }

    inline ImVec2 getCenter() const
    {
        return ImVec2(position.x + size.x * 0.5f, position.y + size.y * 0.5f);
    }

    inline ImVec2 ndcToScreen(const ImVec2& ndcPoint) const
    {
        return ImVec2(
            position.x + (ndcPoint.x * 0.5f + 0.5f) * size.x,
            position.y + (-ndcPoint.y * 0.5f + 0.5f) * size.y);
    }
};

//! Shared style bundle for the follow-target viewport overlay.
struct SCameraFollowTargetViewportOverlayStyle final
{
    static constexpr float CenteredNdcRadius = system::SCameraFollowRegressionThresholds::DefaultProjectedNdcTolerance;
    static constexpr float CenterRadius = 16.0f;
    static constexpr float CenterCrossHalfExtent = 22.0f;
    static constexpr float CenterLineThickness = 2.0f;
    static constexpr float CenterCircleThickness = 2.5f;
    static constexpr float CenteredTargetRadius = 18.0f;
    static constexpr float DefaultTargetRadius = 14.0f;
    static constexpr float TargetCrossHalfExtent = 14.0f;
    static constexpr float LinkLineThickness = 2.0f;
    static constexpr float LabelOffsetX = 16.0f;
    static constexpr float LabelOffsetY = -28.0f;
    static constexpr int32_t CircleSegments = 32;
    static constexpr int32_t FilledCircleSegments = 24;
    static constexpr ImU32 CenterColor = IM_COL32(255, 170, 72, 235);
    static constexpr ImU32 CenteredTargetColor = IM_COL32(64, 255, 164, 245);
    static constexpr ImU32 DefaultTargetColor = IM_COL32(90, 220, 255, 245);
    static constexpr ImU32 CenteredTargetFillColor = IM_COL32(24, 120, 76, 120);
    static constexpr ImU32 DefaultTargetFillColor = IM_COL32(20, 92, 124, 120);
    static constexpr ImU32 CenteredLinkColor = IM_COL32(96, 255, 186, 200);
    static constexpr ImU32 DefaultLinkColor = IM_COL32(120, 220, 255, 200);

    float centeredNdcRadius = CenteredNdcRadius;
    float centerRadius = CenterRadius;
    float centerCrossHalfExtent = CenterCrossHalfExtent;
    float centerLineThickness = CenterLineThickness;
    float centerCircleThickness = CenterCircleThickness;
    float centeredTargetRadius = CenteredTargetRadius;
    float defaultTargetRadius = DefaultTargetRadius;
    float targetCrossHalfExtent = TargetCrossHalfExtent;
    float linkLineThickness = LinkLineThickness;
    float labelOffsetX = LabelOffsetX;
    float labelOffsetY = LabelOffsetY;
    int32_t circleSegments = CircleSegments;
    int32_t filledCircleSegments = FilledCircleSegments;
    ImU32 centerColor = CenterColor;
    ImU32 centeredTargetColor = CenteredTargetColor;
    ImU32 defaultTargetColor = DefaultTargetColor;
    ImU32 centeredTargetFillColor = CenteredTargetFillColor;
    ImU32 defaultTargetFillColor = DefaultTargetFillColor;
    ImU32 centeredLinkColor = CenteredLinkColor;
    ImU32 defaultLinkColor = DefaultLinkColor;
};

//! Shared visual style for transparent viewport windows used by scene image panels.
struct SCameraViewportWindowStyle final
{
    static constexpr float WindowRounding = 0.0f;
    static constexpr float WindowBorderSize = 0.0f;
    static constexpr ImVec2 WindowPadding = ImVec2(0.0f, 0.0f);
    static constexpr ImVec4 WindowBackgroundColor = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    float windowRounding = WindowRounding;
    float windowBorderSize = WindowBorderSize;
    ImVec2 windowPadding = WindowPadding;
    ImVec4 windowBackgroundColor = WindowBackgroundColor;
};

//! Shared data bundle for the top-right camera/projection overlay rendered inside one viewport window.
struct SCameraViewportInfoOverlayData final
{
    std::string headline;
    std::string description;
    std::string detail;

    inline bool valid() const
    {
        return !headline.empty() && !description.empty() && !detail.empty();
    }
};

//! Shared style bundle for the top-right camera/projection overlay rendered inside one viewport window.
struct SCameraViewportInfoOverlayStyle final
{
    static constexpr ImVec2 Padding = ImVec2(6.0f, 4.0f);
    static constexpr float LineGap = 2.0f;
    static constexpr float Margin = 6.0f;
    static constexpr float CornerRounding = 6.0f;
    static constexpr ImU32 BackgroundColor = IM_COL32(13, 15, 20, 204);
    static constexpr ImU32 BorderColor = IM_COL32(153, 168, 194, 204);
    static constexpr ImU32 HeadlineColor = IM_COL32(245, 250, 255, 255);
    static constexpr ImU32 DescriptionColor = IM_COL32(199, 209, 230, 255);
    static constexpr ImU32 DetailColor = IM_COL32(245, 230, 92, 255);

    ImVec2 padding = Padding;
    float lineGap = LineGap;
    float margin = Margin;
    float cornerRounding = CornerRounding;
    ImU32 backgroundColor = BackgroundColor;
    ImU32 borderColor = BorderColor;
    ImU32 headlineColor = HeadlineColor;
    ImU32 descriptionColor = DescriptionColor;
    ImU32 detailColor = DetailColor;
};

//! Shared style bundle for small hover-info popups near the mouse cursor.
struct SCameraHoverInfoOverlayStyle final
{
    static constexpr ImVec4 WindowBackgroundColor = ImVec4(0.20f, 0.20f, 0.20f, 0.80f);
    static constexpr ImVec4 BorderColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    static constexpr float BorderSize = 1.5f;
    static constexpr ImVec2 MouseOffset = ImVec2(10.0f, 10.0f);

    ImVec4 windowBackgroundColor = WindowBackgroundColor;
    ImVec4 borderColor = BorderColor;
    float borderSize = BorderSize;
    ImVec2 mouseOffset = MouseOffset;
};

//! Shared style bundle for the split divider overlay between stacked viewport windows.
struct SCameraViewportSplitOverlayStyle final
{
    static constexpr float MinimumGapFill = 2.0f;
    static constexpr float DividerLineThickness = 2.0f;
    static constexpr ImU32 GapFillColor = IM_COL32(13, 15, 20, 217);
    static constexpr ImU32 DividerLineColor = IM_COL32(204, 214, 235, 191);

    float minimumGapFill = MinimumGapFill;
    float dividerLineThickness = DividerLineThickness;
    ImU32 gapFillColor = GapFillColor;
    ImU32 dividerLineColor = DividerLineColor;
};

struct CCameraViewportOverlayUtilities final
{
    static inline void pushViewportWindowStyle(const SCameraViewportWindowStyle& style = {})
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, style.windowRounding);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, style.windowBorderSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, style.windowPadding);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, style.windowBackgroundColor);
    }

    static inline void popViewportWindowStyle()
    {
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(3);
    }

    static inline void drawViewportInfoOverlay(
        ImDrawList& drawList,
        const SViewportOverlayRect& viewportRect,
        const SCameraViewportInfoOverlayData& data,
        const SCameraViewportInfoOverlayStyle& style = {})
    {
        if (!viewportRect.valid() || !data.valid())
            return;

        const ImVec2 headlineSize = ImGui::CalcTextSize(data.headline.c_str());
        const ImVec2 descriptionSize = ImGui::CalcTextSize(data.description.c_str());
        const ImVec2 detailSize = ImGui::CalcTextSize(data.detail.c_str());
        const float width = std::max(std::max(headlineSize.x, descriptionSize.x), detailSize.x);
        const float height = headlineSize.y + descriptionSize.y + detailSize.y + style.lineGap * 2.0f + style.padding.y * 2.0f;
        ImVec2 overlayPos(
            viewportRect.position.x + viewportRect.size.x - width - style.padding.x * 2.0f - style.margin,
            viewportRect.position.y + style.margin);
        overlayPos.x = std::max(overlayPos.x, viewportRect.position.x + style.margin);
        const ImVec2 overlayMax(overlayPos.x + width + style.padding.x * 2.0f, overlayPos.y + height);

        drawList.AddRectFilled(overlayPos, overlayMax, style.backgroundColor, style.cornerRounding);
        drawList.AddRect(overlayPos, overlayMax, style.borderColor, style.cornerRounding);
        drawList.AddText(ImVec2(overlayPos.x + style.padding.x, overlayPos.y + style.padding.y), style.headlineColor, data.headline.c_str());
        drawList.AddText(
            ImVec2(overlayPos.x + style.padding.x, overlayPos.y + style.padding.y + headlineSize.y + style.lineGap),
            style.descriptionColor,
            data.description.c_str());
        drawList.AddText(
            ImVec2(overlayPos.x + style.padding.x, overlayPos.y + style.padding.y + headlineSize.y + descriptionSize.y + style.lineGap * 2.0f),
            style.detailColor,
            data.detail.c_str());
    }

    static inline void beginHoverInfoOverlay(const char* name, const ImVec2& mousePos, const SCameraHoverInfoOverlayStyle& style = {})
    {
        ImGui::PushStyleColor(ImGuiCol_WindowBg, style.windowBackgroundColor);
        ImGui::PushStyleColor(ImGuiCol_Border, style.borderColor);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, style.borderSize);
        ImGui::SetNextWindowPos(ImVec2(mousePos.x + style.mouseOffset.x, mousePos.y + style.mouseOffset.y), ImGuiCond_Always);
        ImGui::Begin(name, nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings);
    }

    static inline void endHoverInfoOverlay()
    {
        ImGui::End();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }

    static inline void drawViewportSplitOverlay(
        ImDrawList& drawList,
        const ImVec2& displaySize,
        const float splitY,
        const float gap,
        const SCameraViewportSplitOverlayStyle& style = {})
    {
        if (gap >= style.minimumGapFill)
        {
            drawList.AddRectFilled(
                ImVec2(0.0f, splitY),
                ImVec2(displaySize.x, splitY + gap),
                style.gapFillColor);
            return;
        }

        drawList.AddLine(
            ImVec2(0.0f, splitY),
            ImVec2(displaySize.x, splitY),
            style.dividerLineColor,
            style.dividerLineThickness);
    }

    static inline void drawFollowTargetViewportOverlay(
        ImDrawList& drawList,
        const system::SCameraProjectionContext& projectionContext,
        const core::CTrackedTarget& trackedTarget,
        const SViewportOverlayRect& viewportRect,
        const SCameraFollowTargetViewportOverlayStyle& style = {})
    {
        if (!viewportRect.valid())
            return;

        system::SCameraProjectedTargetMetrics projectedTarget = {};
        if (!system::CCameraFollowRegressionUtilities::tryComputeProjectedFollowTargetMetrics(projectionContext, trackedTarget, projectedTarget))
            return;

        const bool centered = projectedTarget.radius <= style.centeredNdcRadius;
        const ImVec2 center = viewportRect.getCenter();
        const ImVec2 target = viewportRect.ndcToScreen(ImVec2(projectedTarget.ndc.x, projectedTarget.ndc.y));
        const float targetRadius = centered ? style.centeredTargetRadius : style.defaultTargetRadius;
        const ImU32 targetColor = centered ? style.centeredTargetColor : style.defaultTargetColor;
        const ImU32 targetFillColor = centered ? style.centeredTargetFillColor : style.defaultTargetFillColor;
        const ImU32 linkColor = centered ? style.centeredLinkColor : style.defaultLinkColor;

        drawList.AddCircle(center, style.centerRadius, style.centerColor, style.circleSegments, style.centerCircleThickness);
        drawList.AddLine(
            ImVec2(center.x - style.centerCrossHalfExtent, center.y),
            ImVec2(center.x + style.centerCrossHalfExtent, center.y),
            style.centerColor,
            style.centerLineThickness);
        drawList.AddLine(
            ImVec2(center.x, center.y - style.centerCrossHalfExtent),
            ImVec2(center.x, center.y + style.centerCrossHalfExtent),
            style.centerColor,
            style.centerLineThickness);

        drawList.AddLine(center, target, linkColor, style.linkLineThickness);
        drawList.AddCircleFilled(target, targetRadius, targetFillColor, style.filledCircleSegments);
        drawList.AddCircle(target, targetRadius, targetColor, style.circleSegments, style.centerCircleThickness);
        drawList.AddLine(
            ImVec2(target.x - style.targetCrossHalfExtent, target.y),
            ImVec2(target.x + style.targetCrossHalfExtent, target.y),
            targetColor,
            style.centerLineThickness);
        drawList.AddLine(
            ImVec2(target.x, target.y - style.targetCrossHalfExtent),
            ImVec2(target.x, target.y + style.targetCrossHalfExtent),
            targetColor,
            style.centerLineThickness);

        drawList.AddText(
            ImVec2(target.x + style.labelOffsetX, target.y + style.labelOffsetY),
            targetColor,
            "FOLLOW TARGET");
    }
};

} // namespace nbl::ui

#endif // _C_CAMERA_VIEWPORT_OVERLAY_UTILITIES_HPP_
