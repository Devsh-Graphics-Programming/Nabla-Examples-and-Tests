// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SCRIPT_VISUAL_DEBUG_OVERLAY_UTILITIES_HPP_
#define _C_CAMERA_SCRIPT_VISUAL_DEBUG_OVERLAY_UTILITIES_HPP_

#include <algorithm>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>

#include "imgui/imgui.h"

namespace nbl::ui
{

//! Shared data bundle for the scripted visual debug HUD.
struct SCameraScriptVisualDebugOverlayData final
{
    std::string title;
    std::string headline;
    std::string progressLine;
    std::string hintLine;

    inline bool valid() const
    {
        return !headline.empty() && !progressLine.empty();
    }
};

//! Shared camera/debug state used to format one scripted visual debug HUD payload.
struct SCameraScriptVisualDebugStatus final
{
    std::string_view title = "SCRIPT VISUAL DEBUG";
    std::string_view cameraLabel = "Unknown";
    std::string_view cameraHint = "Unspecified camera behavior";
    uint32_t cameraIndex = 0u;
    uint32_t cameraCount = 0u;
    uint32_t planarIndex = 0u;
    bool hasHoldFrames = false;
    uint64_t progressFrames = 0u;
    uint64_t holdFrames = 0u;
    float targetFps = 60.0f;
    uint64_t absoluteFrame = 0u;
    std::string_view segmentLabel = {};
    bool hasDynamicFov = false;
    float dynamicFovDeg = 0.0f;
    bool followActive = false;
    std::string_view followModeDescription = "Follow off";
    bool followLockValid = false;
    float followLockAngleDeg = 0.0f;
    float followTargetDistance = 0.0f;
    float followTargetCenterNdcRadius = 0.0f;
};

//! Shared style bundle for the scripted visual debug HUD.
struct SCameraScriptVisualDebugOverlayStyle final
{
    static constexpr float TitleSize = 50.0f;
    static constexpr float HeadlineSize = 38.0f;
    static constexpr float ProgressSize = 28.0f;
    static constexpr float HintSize = 24.0f;
    static constexpr float MarginTop = 18.0f;
    static constexpr float PaddingX = 24.0f;
    static constexpr float PaddingY = 16.0f;
    static constexpr float LineGap = 6.0f;
    static constexpr float CornerRounding = 14.0f;
    static constexpr float BorderThickness = 2.5f;
    static constexpr ImU32 BackgroundColor = IM_COL32(6, 8, 12, 232);
    static constexpr ImU32 BorderColor = IM_COL32(255, 166, 64, 255);
    static constexpr ImU32 TitleColor = IM_COL32(255, 206, 120, 255);
    static constexpr ImU32 HeadlineColor = IM_COL32(255, 244, 224, 255);
    static constexpr ImU32 ProgressColor = IM_COL32(202, 222, 255, 255);
    static constexpr ImU32 HintColor = IM_COL32(170, 204, 255, 255);

    float titleSize = TitleSize;
    float headlineSize = HeadlineSize;
    float progressSize = ProgressSize;
    float hintSize = HintSize;
    float marginTop = MarginTop;
    float paddingX = PaddingX;
    float paddingY = PaddingY;
    float lineGap = LineGap;
    float cornerRounding = CornerRounding;
    float borderThickness = BorderThickness;
    ImU32 backgroundColor = BackgroundColor;
    ImU32 borderColor = BorderColor;
    ImU32 titleColor = TitleColor;
    ImU32 headlineColor = HeadlineColor;
    ImU32 progressColor = ProgressColor;
    ImU32 hintColor = HintColor;
};

//! Draw the scripted visual debug HUD on the foreground draw list.
inline void drawScriptVisualDebugOverlay(
    const ImVec2& displaySize,
    const SCameraScriptVisualDebugOverlayData& data,
    const SCameraScriptVisualDebugOverlayStyle& style = {})
{
    if (!data.valid())
        return;

    ImFont* font = ImGui::GetFont();
    ImDrawList* drawList = ImGui::GetForegroundDrawList();
    if (!font || !drawList)
        return;

    const float textWrap = std::numeric_limits<float>::max();
    const ImVec2 titleSize = font->CalcTextSizeA(style.titleSize, textWrap, 0.0f, data.title.c_str());
    const ImVec2 headlineSize = font->CalcTextSizeA(style.headlineSize, textWrap, 0.0f, data.headline.c_str());
    const ImVec2 progressSize = font->CalcTextSizeA(style.progressSize, textWrap, 0.0f, data.progressLine.c_str());
    const ImVec2 hintSize = font->CalcTextSizeA(style.hintSize, textWrap, 0.0f, data.hintLine.c_str());
    const float panelWidth = std::max(std::max(titleSize.x, headlineSize.x), std::max(progressSize.x, hintSize.x)) + style.paddingX * 2.0f;
    const float panelHeight = titleSize.y + headlineSize.y + progressSize.y + hintSize.y + style.lineGap * 3.0f + style.paddingY * 2.0f;
    const ImVec2 panelMin((displaySize.x - panelWidth) * 0.5f, style.marginTop);
    const ImVec2 panelMax(panelMin.x + panelWidth, panelMin.y + panelHeight);

    drawList->AddRectFilled(panelMin, panelMax, style.backgroundColor, style.cornerRounding);
    drawList->AddRect(panelMin, panelMax, style.borderColor, style.cornerRounding, 0, style.borderThickness);

    const float titleX = panelMin.x + (panelWidth - titleSize.x) * 0.5f;
    const float headlineX = panelMin.x + (panelWidth - headlineSize.x) * 0.5f;
    const float progressX = panelMin.x + (panelWidth - progressSize.x) * 0.5f;
    const float hintX = panelMin.x + (panelWidth - hintSize.x) * 0.5f;
    const float titleY = panelMin.y + style.paddingY;
    const float headlineY = titleY + titleSize.y + style.lineGap;
    const float progressY = headlineY + headlineSize.y + style.lineGap;
    const float hintY = progressY + progressSize.y + style.lineGap;

    drawList->AddText(font, style.titleSize, ImVec2(titleX, titleY), style.titleColor, data.title.c_str());
    drawList->AddText(font, style.headlineSize, ImVec2(headlineX, headlineY), style.headlineColor, data.headline.c_str());
    drawList->AddText(font, style.progressSize, ImVec2(progressX, progressY), style.progressColor, data.progressLine.c_str());
    drawList->AddText(font, style.hintSize, ImVec2(hintX, hintY), style.hintColor, data.hintLine.c_str());
}

//! Build the display strings for one scripted visual debug HUD snapshot.
inline SCameraScriptVisualDebugOverlayData buildScriptVisualDebugOverlayData(const SCameraScriptVisualDebugStatus& status)
{
    SCameraScriptVisualDebugOverlayData out = {};
    out.title = std::string(status.title);
    out.headline = "Camera " + std::to_string(status.cameraIndex + 1u) + "/" + std::to_string(status.cameraCount) + "  " + std::string(status.cameraLabel);

    char progressBuffer[256] = {};
    if (status.hasHoldFrames)
    {
        const float safeFps = std::max(status.targetFps, 1.0f);
        const double elapsedSeconds = static_cast<double>(status.progressFrames) / static_cast<double>(safeFps);
        const double holdSeconds = static_cast<double>(status.holdFrames) / static_cast<double>(safeFps);
        std::snprintf(
            progressBuffer,
            sizeof(progressBuffer),
            "Planar %u  Segment %.1f/%.1f s  Frame %llu/%llu",
            status.planarIndex,
            elapsedSeconds,
            holdSeconds,
            static_cast<unsigned long long>(status.progressFrames),
            static_cast<unsigned long long>(status.holdFrames));
    }
    else
    {
        std::snprintf(
            progressBuffer,
            sizeof(progressBuffer),
            "Planar %u  Frame %llu",
            status.planarIndex,
            static_cast<unsigned long long>(status.absoluteFrame));
    }
    out.progressLine = progressBuffer;
    if (!status.segmentLabel.empty())
        out.progressLine += "  |  " + std::string(status.segmentLabel);

    out.hintLine = std::string(status.cameraHint);
    if (status.hasDynamicFov)
    {
        char fovBuffer[96] = {};
        std::snprintf(fovBuffer, sizeof(fovBuffer), "  |  Dynamic FOV %.2f deg", status.dynamicFovDeg);
        out.hintLine += fovBuffer;
    }

    if (status.followActive)
    {
        out.hintLine += "  |  " + std::string(status.followModeDescription);
        if (status.followLockValid)
        {
            char followBuffer[192] = {};
            std::snprintf(
                followBuffer,
                sizeof(followBuffer),
                "  |  lock %.2f deg  |  target %.2f  |  center err %.3f",
                status.followLockAngleDeg,
                status.followTargetDistance,
                status.followTargetCenterNdcRadius);
            out.hintLine += followBuffer;
        }
        else
        {
            out.hintLine += "  |  lock n/a  |  target n/a  |  center err n/a";
        }
    }
    else
    {
        out.hintLine += "  |  Follow off";
    }

    return out;
}

} // namespace nbl::ui

#endif // _C_CAMERA_SCRIPT_VISUAL_DEBUG_OVERLAY_UTILITIES_HPP_
