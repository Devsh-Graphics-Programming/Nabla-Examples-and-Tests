// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "imgui/imgui_internal.h"
#include "app_resources/common.hlsl"
#include "app_resources/imgui.opts.hlsl"

void IESViewer::uiListener()
{
    const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;

    SImResourceInfo info;
    info.textureID = ext::imgui::UI::FontAtlasTexId + resourceIx + 1u;
    info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

    auto& ies = m_assets[m_activeAssetIx];
    const auto name = path(ies.key).filename().string();
    auto* profile = ies.getProfile();
    const float lowerBound = (float)profile->getHoriAngles().front();
    const float upperBound = (float)profile->getHoriAngles().back();
    const bool singleAngle = (upperBound == lowerBound);

    auto angle = ImClamp(ies.zDegree, lowerBound, upperBound);
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 imageSize(640.f, 640.f);

    // 2D Plot
    {
        ImDrawList* fg = ImGui::GetForegroundDrawList();
        float x = vp->Pos.x + 8.f;
        float y = vp->Pos.y + 8.f;

        fg->AddText(ImVec2(x, y), ImGui::GetColorU32(ImGuiCol_Text), IES::modeToRS(ies.mode));
        y += ImGui::GetTextLineHeightWithSpacing();

        fg->AddText(ImVec2(x, y), ImGui::GetColorU32(ImGuiCol_Text), IES::symmetryToRS(profile->getSymmetry()));
        y += ImGui::GetTextLineHeightWithSpacing();

        fg->AddText(ImVec2(x, y), ImGui::GetColorU32(ImGuiCol_Text), name.c_str());
        y += ImGui::GetTextLineHeightWithSpacing();

        char b1[64]; snprintf(b1, sizeof(b1), "%.3f\xC2\xB0", angle);
        fg->AddText(ImVec2(x, y), ImGui::GetColorU32(ImGuiCol_Text), b1);
    }

    {
        const ImVec2 imageCenter(
            vp->Pos.x + vp->Size.x * 0.5f,
            vp->Pos.y + vp->Size.y * 0.25f
        );

        ImGui::SetNextWindowPos(imageCenter, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);

        ImGuiWindowFlags imgFlags =
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoNav |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse;

        if (ImGui::Begin("2D Plot", nullptr, imgFlags))
        {
            ImGui::Image(info, imageSize);
        }
        ImGui::End();

        ImGui::PopStyleVar(2);
    }

    {
        const float pad = 8.f;
        const float sliderW = 74.f;
        const float sliderH = ImMin(vp->Size.y - pad * 2.f, 260.f);
        ImGui::SetNextWindowPos(ImVec2(vp->Pos.x + vp->Size.x - sliderW - pad, vp->Pos.y + pad), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(sliderW, sliderH), ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBackground;

        if (ImGui::Begin("AngleSliderOverlay", nullptr, flags))
        {
            ImGui::InvisibleButton("##fader_area", ImGui::GetContentRegionAvail());
            ImVec2 rmin = ImGui::GetItemRectMin();
            ImVec2 rmax = ImGui::GetItemRectMax();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            ImU32 col = IM_COL32(220, 60, 60, 255);

            float knobR = 7.f;
            float trackX = rmax.x - 12.f;
            float y0 = rmin.y + knobR + 1.f;
            float y1 = rmax.y - knobR - 1.f;

            dl->AddLine(ImVec2(trackX, y0), ImVec2(trackX, y1), col, 3.f);

            if (singleAngle)
            {
                float y = (y0 + y1) * 0.5f;
                dl->AddLine(ImVec2(trackX - 22.f, y), ImVec2(trackX - 8.f, y), ImGui::GetColorU32(ImGuiCol_Text));
                char tb[32]; snprintf(tb, sizeof(tb), "%.0f", lowerBound);
                ImVec2 ts = ImGui::CalcTextSize(tb);
                dl->AddText(ImVec2(trackX - 24.f - ts.x, y - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), tb);
            }
            else
            {
                for (int i = 0; i < 5; ++i)
                {
                    float v = lowerBound + (upperBound - lowerBound) * (float(i) / 4.f);
                    float t = (v - lowerBound) / (upperBound - lowerBound);
                    float y = y1 - t * (y1 - y0);
                    dl->AddLine(ImVec2(trackX - 22.f, y), ImVec2(trackX - 8.f, y), ImGui::GetColorU32(ImGuiCol_Text));
                    char tb[32]; snprintf(tb, sizeof(tb), "%.0f", v);
                    ImVec2 ts = ImGui::CalcTextSize(tb);
                    dl->AddText(ImVec2(trackX - 24.f - ts.x, y - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), tb);
                }
            }

            float t = singleAngle ? 0.5f : (angle - lowerBound) / (upperBound - lowerBound);
            float knobY = y1 - t * (y1 - y0);
            dl->AddCircleFilled(ImVec2(trackX, knobY), knobR, col);
            dl->AddCircle(ImVec2(trackX, knobY), knobR, ImGui::GetColorU32(ImGuiCol_Border));

            if (!singleAngle && (ImGui::IsItemHovered() || ImGui::IsItemActive()) && ImGui::IsMouseDown(0))
            {
                float my = ImClamp(ImGui::GetIO().MousePos.y, y0, y1);
                float nt = (y1 - my) / (y1 - y0);
                angle = lowerBound + nt * (upperBound - lowerBound);
            }
        }
        ImGui::End();
        ImGui::PopStyleVar(2);
    }

    ies.zDegree = angle;

    // 3D plot
    {
        info.textureID += device_base_t::MaxFramesInFlight;

        {
            const ImVec2 imageCenter(
                vp->Pos.x + vp->Size.x * 0.5f,
                vp->Pos.y + vp->Size.y * 0.75f
            );

            ImGui::SetNextWindowPos(imageCenter, ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));

            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);

            ImGuiWindowFlags imgFlags =
                ImGuiWindowFlags_NoSavedSettings |
                ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoNav |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse;

            if (ImGui::Begin("3D Plot", nullptr, imgFlags))
            {
                ImGui::Image(info, imageSize);
            }
            ImGui::End();

            ImGui::PopStyleVar(2);
        }
    }
}