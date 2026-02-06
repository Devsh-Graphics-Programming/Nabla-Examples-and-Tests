// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include "imgui/imgui.h"
#include "imgui/imgui_internal.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "app_resources/common.hlsl"
#include "app_resources/false_color.hlsl"
#include "app_resources/imgui.opts.hlsl"
#include "nbl/builtin/hlsl/math/thin_lens_projection.hlsl"
#include "nbl/builtin/hlsl/math/linalg/fast_affine.hlsl"
#include "nbl/builtin/hlsl/math/octahedral.hlsl"

void IESViewer::uiListener()
{
    const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;

    SImResourceInfo info;
    info.textureID = ext::imgui::UI::FontAtlasTexId + resourceIx + 1u;
    info.samplerIx = (uint16_t)ext::imgui::UI::DefaultSamplerIx::USER;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 viewportPos = vp->Pos;
    const ImVec2 viewportSize = vp->Size;
    auto* cursorControl = m_window->getCursorControl();
    const auto cursorPosition = cursorControl ? cursorControl->getPosition() : ICursorControl::SPosition{};
    const int32_t windowX = m_window->getX();
    const int32_t windowY = m_window->getY();
    const int32_t windowW = static_cast<int32_t>(m_window->getWidth());
    const int32_t windowH = static_cast<int32_t>(m_window->getHeight());
    const bool cursorInsideWindow = cursorControl &&
        cursorPosition.x >= windowX && cursorPosition.x < windowX + windowW &&
        cursorPosition.y >= windowY && cursorPosition.y < windowY + windowH;
    ImGui::GetIO().MouseDrawCursor = cursorInsideWindow && !uiState.cameraControlEnabled;
    const ImVec2 bottomSize(viewportSize.x, viewportSize.y);
    const ImVec2 bottomPos(viewportPos.x, viewportPos.y);
    const auto legendColor = [&](float v, bool useFalseColor) -> ImU32
    {
        const float clamped = ImClamp(v, 0.0f, 1.0f);
        if (useFalseColor)
        {
            const auto col = hlsl::this_example::ies::falseColor(clamped);
            return ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, 1.0f));
        }
        return ImGui::ColorConvertFloat4ToU32(ImVec4(clamped, clamped, clamped, 1.0f));
    };
    const auto showHint = [&](const char* text)
    {
        if (!uiState.showHints || !text || text[0] == '\0')
            return;
        if (!ImGui::IsItemHovered())
            return;
        ImGui::BeginTooltip();
        ImGui::TextUnformatted(text);
        ImGui::EndTooltip();
    };
    std::vector<const char*> assetLabelPtrs;
    assetLabelPtrs.reserve(m_assetLabels.size());
    for (const auto& label : m_assetLabels)
        assetLabelPtrs.push_back(label.c_str());

    size_t activeIx = uiState.activeAssetIx;
    if (activeIx >= m_assets.size())
        activeIx = 0u;
    int activeIxUi = static_cast<int>(activeIx);
    float candelaValue = 0.0f;
    bool candelaValid = false;
    ImVec2 plotRectMin(0.f, 0.f);
    ImVec2 plotRectMax(0.f, 0.f);
    bool plotRectValid = false;
    bool plotHovered = false;
    uiState.plot2DRectValid = false;

    auto& ies = m_assets[activeIx];
    auto* profile = ies.getProfile();
	const auto& accessor = profile->getAccessor();
    const auto& properties = accessor.getProperties();

    const float lowerBound = accessor.hAngles.front();
    const float upperBound = accessor.hAngles.back();
    const bool singleAngle = (upperBound == lowerBound);

    constexpr size_t kSmallBufSize = 32;
    auto angle = ImClamp(ies.zDegree, lowerBound, upperBound);

    auto updateCameraProjection = [&]()
    {
        if (m_plot3DWidth == 0u || m_plot3DHeight == 0u)
            return;
        const float aspect = float(m_plot3DWidth) / float(m_plot3DHeight);
        const auto projectionMatrix = buildProjectionMatrixPerspectiveFovLH<float32_t>(hlsl::radians(uiState.cameraFovDeg), aspect, 0.1f, 10000.0f);
        camera.setProjectionMatrix(projectionMatrix);
    };

    auto draw3DControls = [&]()
    {
        bool interpolateCandela = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_OCTAHEDRAL_UV_INTERPOLATE);

        if (ImGui::Checkbox("interpolate candelas", &interpolateCandela))
        {
            if (interpolateCandela)
                uiState.mode.sphere |= hlsl::this_example::ies::E_SPHERE_MODE::ESM_OCTAHEDRAL_UV_INTERPOLATE;
            else
                uiState.mode.sphere &= static_cast<hlsl::this_example::ies::E_SPHERE_MODE>(
                    ~hlsl::this_example::ies::E_SPHERE_MODE::ESM_OCTAHEDRAL_UV_INTERPOLATE
                );
        }
        showHint("Interpolate candela values in the octahedral map.");

        bool falseColor = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_FALSE_COLOR);

        if (ImGui::Checkbox("false color", &falseColor))
        {
            if (falseColor)
                uiState.mode.sphere |= hlsl::this_example::ies::E_SPHERE_MODE::ESM_FALSE_COLOR;
            else
                uiState.mode.sphere &= static_cast<hlsl::this_example::ies::E_SPHERE_MODE>(
                    ~hlsl::this_example::ies::E_SPHERE_MODE::ESM_FALSE_COLOR
                );
        }
        showHint("Use false color palette for the 3D plot.");

        bool showOctaMap = uiState.showOctaMapPreview;
        if (ImGui::Checkbox("octahedral map", &showOctaMap))
            uiState.showOctaMapPreview = showOctaMap;
        showHint("Show octahedral map preview under the 2D plot.");

        bool showHints = uiState.showHints;
        if (ImGui::Checkbox("show hints", &showHints))
            uiState.showHints = showHints;
        showHint("Toggle help tooltips.");

        bool cubePlot = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_CUBE);

        if (ImGui::Checkbox("cube plot", &cubePlot))
        {
            if (cubePlot)
                uiState.mode.sphere |= hlsl::this_example::ies::E_SPHERE_MODE::ESM_CUBE;
            else
                uiState.mode.sphere &= static_cast<hlsl::this_example::ies::E_SPHERE_MODE>(
                    ~hlsl::this_example::ies::E_SPHERE_MODE::ESM_CUBE
                );
        }
        showHint("Render the plot on a cube instead of a sphere.");

        bool wireframe = uiState.wireframeEnabled;
        if (ImGui::Checkbox("wireframe", &wireframe))
            uiState.wireframeEnabled = wireframe;
        showHint("Show wireframe topology in the 3D plot.");

        bool cameraControl = uiState.cameraControlEnabled;
        if (ImGui::Checkbox("camera control (space)", &cameraControl))
            uiState.cameraControlEnabled = cameraControl;
        showHint("Enable camera movement with mouse and keyboard.");

        bool speedChanged = false;
        bool fovChanged = false;
        if (ImGui::BeginTable("##camera_controls", 2, ImGuiTableFlags_SizingStretchProp))
        {
            float labelWidth = 0.0f;
            labelWidth = ImMax(labelWidth, ImGui::CalcTextSize("move speed").x);
            labelWidth = ImMax(labelWidth, ImGui::CalcTextSize("rotate speed").x);
            labelWidth = ImMax(labelWidth, ImGui::CalcTextSize("fov").x);
            labelWidth += ImGui::GetStyle().CellPadding.x * 2.0f;
            labelWidth = ImMin(labelWidth, ImGui::GetContentRegionAvail().x * 0.6f);
            ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, labelWidth);
            ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
            auto sliderRow = [&](const char* label, float* value, float min, float max, const char* fmt, const char* hint)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted(label);
                showHint(hint);
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-FLT_MIN);
                ImGui::PushID(label);
                const bool changed = ImGui::SliderFloat("##value", value, min, max, fmt, ImGuiSliderFlags_AlwaysClamp);
                ImGui::PopID();
                showHint(hint);
                return changed;
            };

            speedChanged |= sliderRow("move speed", &uiState.cameraMoveSpeed, 0.1f, 10.0f, "%.2f", "Camera movement speed.");
            speedChanged |= sliderRow("rotate speed", &uiState.cameraRotateSpeed, 0.1f, 5.0f, "%.2f", "Camera rotation speed.");
            fovChanged |= sliderRow("fov", &uiState.cameraFovDeg, 30.0f, 120.0f, "%.0f", "Camera field of view.");

            ImGui::EndTable();
        }

        if (speedChanged && uiState.cameraControlEnabled)
        {
            camera.setMoveSpeed(uiState.cameraMoveSpeed);
            camera.setRotateSpeed(uiState.cameraRotateSpeed);
        }

        if (fovChanged)
            updateCameraProjection();

    };

    const float panelMargin = 8.f;
    const float panelWidth = ImClamp(viewportSize.x * 0.25f, 260.0f, 420.0f);
    const float panelMaxHeight = ImMax(240.0f, viewportSize.y * 0.9f);
    ImGui::SetNextWindowPos(ImVec2(viewportPos.x + panelMargin, viewportPos.y + panelMargin), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(panelWidth, 0.0f), ImVec2(panelWidth, panelMaxHeight));
    ImGui::SetNextWindowBgAlpha(0.7f);
    ImGuiWindowFlags panelFlags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoNav |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoResize;

    if (ImGui::Begin("IES Panel", nullptr, panelFlags))
    {
        const auto& resolution = accessor.properties.optimalIESResolution;

        constexpr size_t kInfoBufSize = 64;
        std::array<char, kInfoBufSize> bAngle{};
        std::array<char, kInfoBufSize> bAngles{};
        std::array<char, kInfoBufSize> bRes{};
        std::array<char, kInfoBufSize> bMax{};
        std::array<char, kInfoBufSize> bAvg{};
        std::array<char, kInfoBufSize> bAvgFull{};
        const auto hCount = accessor.hAnglesCount();
        const auto vCount = accessor.vAnglesCount();
        std::snprintf(bAngle.data(), bAngle.size(), "%.3f deg", angle);
        std::snprintf(bAngles.data(), bAngles.size(), "angles: %u x %u", hCount, vCount);
        std::snprintf(bRes.data(), bRes.size(), "resolution: %u x %u", resolution.x, resolution.y);
        std::snprintf(bMax.data(), bMax.size(), "max cd: %.3f", properties.maxCandelaValue);
        std::snprintf(bAvg.data(), bAvg.size(), "avg: %.3f", properties.avgEmmision);
        std::snprintf(bAvgFull.data(), bAvgFull.size(), "avg full: %.3f", properties.fullDomainAvgEmission);
        const std::string symmetryLabel = nbl::system::to_string(properties.getSymmetry());
        const std::string typeLabel = nbl::system::to_string(properties.getType());
        const std::string versionLabel = nbl::system::to_string(properties.getVersion());
        float leftWidth = 0.0f;
        leftWidth = ImMax(leftWidth, ImGui::CalcTextSize(symmetryLabel.c_str()).x);
        leftWidth = ImMax(leftWidth, ImGui::CalcTextSize(versionLabel.c_str()).x);
        leftWidth = ImMax(leftWidth, ImGui::CalcTextSize(bAngles.data()).x);
        leftWidth = ImMax(leftWidth, ImGui::CalcTextSize(bMax.data()).x);
        leftWidth = ImMax(leftWidth, ImGui::CalcTextSize(bAvgFull.data()).x);
        leftWidth += ImGui::GetStyle().CellPadding.x * 2.0f;
        leftWidth = ImMin(leftWidth, ImGui::GetContentRegionAvail().x * 0.6f);
        if (ImGui::BeginTable("##profile_info", 2, ImGuiTableFlags_SizingFixedFit))
        {
            ImGui::TableSetupColumn("left", ImGuiTableColumnFlags_WidthFixed, leftWidth);
            ImGui::TableSetupColumn("right", ImGuiTableColumnFlags_WidthStretch);
            auto rightText = [&](const char* text, const char* hint)
            {
                const float avail = ImGui::GetContentRegionAvail().x;
                const float textWidth = ImGui::CalcTextSize(text).x;
                const char* displayText = text;
                std::string clipped;
                if (textWidth > avail && avail > 0.0f)
                {
                    const char* ell = "...";
                    const float ellW = ImGui::CalcTextSize(ell).x;
                    const float target = ImMax(0.0f, avail - ellW);
                    const int len = static_cast<int>(std::strlen(text));
                    int lo = 0;
                    int hi = len;
                    while (lo < hi)
                    {
                        int mid = (lo + hi + 1) / 2;
                        float w = ImGui::CalcTextSize(text, text + mid).x;
                        if (w <= target)
                            lo = mid;
                        else
                            hi = mid - 1;
                    }
                    clipped.assign(text, text + lo);
                    clipped.append(ell);
                    displayText = clipped.c_str();
                }
                const float displayWidth = ImGui::CalcTextSize(displayText).x;
                if (displayWidth < avail)
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - displayWidth));
                ImGui::TextUnformatted(displayText);
                showHint(hint);
            };
            auto row = [&](const char* left, const char* right, const char* leftHint, const char* rightHint)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted(left);
                showHint(leftHint);
                ImGui::TableSetColumnIndex(1);
                rightText(right, rightHint);
            };

            row(symmetryLabel.c_str(), typeLabel.c_str(), "IES symmetry mode.", "IES photometric type.");
            row(versionLabel.c_str(), assetLabelPtrs.empty() ? ies.key.c_str() : assetLabelPtrs[activeIx], "IES standard/version.", "Active IES profile file.");
            row(bAngles.data(), bRes.data(), "Horizontal and vertical angle count.", "Octahedral map resolution.");
            row(bMax.data(), bAvg.data(), "Maximum candela value.", "Average candela value.");
            row(bAvgFull.data(), bAngle.data(), "Average candela over full domain.", "Current horizontal angle.");

            ImGui::EndTable();
        }

        ImGui::Separator();

        const ImVec2 avail = ImGui::GetContentRegionAvail();
        ImVec2 plotSize(0.f, 0.f);
        float plotSide = ImMax(0.0f, avail.x);
        if (plotSide > 0.0f)
        {
            plotSize = ImVec2(plotSide, plotSide);
            ImVec2 plotPos = ImGui::GetCursorScreenPos();
            {
                const std::string modeLabel = nbl::system::to_string(uiState.mode.view);
                const char* title = modeLabel.c_str();
                const ImVec2 titleSize = ImGui::CalcTextSize(title);
                const float titleX = ImMax(0.0f, (ImGui::GetContentRegionAvail().x - titleSize.x) * 0.5f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + titleX);
                ImGui::TextUnformatted(title);
                showHint("2D candlepower distribution curve.");
            }

            plotPos = ImGui::GetCursorScreenPos();
            ImGui::Image(info, plotSize, ImVec2(0.f, 0.f), ImVec2(1.f, 0.5f));
            const ImVec2 itemMin = ImGui::GetItemRectMin();
            const ImVec2 itemMax = ImGui::GetItemRectMax();
            uiState.plot2DRectMin = float32_t2(itemMin.x, itemMin.y);
            uiState.plot2DRectMax = float32_t2(itemMax.x, itemMax.y);
            uiState.plot2DRectValid = true;
            showHint("2D candlepower distribution curve.");

            ImDrawList* dl = ImGui::GetWindowDrawList();

            const float pad = 6.f;
            const float barWidth = 16.f;
            const float sliderH = ImMax(0.f, plotSize.y - pad * 2.f);
            const float sliderX = plotPos.x + plotSize.x - barWidth - pad;
            const float sliderY = plotPos.y + pad;

            if (sliderH > 0.0f)
            {
                ImGui::SetCursorScreenPos(ImVec2(sliderX, sliderY));
                ImGui::InvisibleButton("##angle_slider", ImVec2(barWidth, sliderH));
                showHint("Adjust horizontal angle.");
                ImVec2 rmin = ImGui::GetItemRectMin();
                ImVec2 rmax = ImGui::GetItemRectMax();
                ImU32 col = IM_COL32(220, 60, 60, 255);

                float knobR = 7.f;
                float trackX = rmax.x - barWidth * 0.5f;
                float y0 = rmin.y + knobR + 1.f;
                float y1 = rmax.y - knobR - 1.f;

                dl->AddLine(ImVec2(trackX, y0), ImVec2(trackX, y1), col, 3.f);

                if (singleAngle)
                {
                    float y = (y0 + y1) * 0.5f;
                    dl->AddLine(ImVec2(trackX - 22.f, y), ImVec2(trackX - 8.f, y), ImGui::GetColorU32(ImGuiCol_Text));
                    std::array<char, kSmallBufSize> tb{};
                    std::snprintf(tb.data(), tb.size(), "%.0f", lowerBound);
                    ImVec2 ts = ImGui::CalcTextSize(tb.data());
                    dl->AddText(ImVec2(trackX - 24.f - ts.x, y - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), tb.data());
                }
                else
                {
                    for (int i = 0; i < 5; ++i)
                    {
                        float v = lowerBound + (upperBound - lowerBound) * (float(i) / 4.f);
                        float t = (v - lowerBound) / (upperBound - lowerBound);
                        float y = y1 - t * (y1 - y0);
                        dl->AddLine(ImVec2(trackX - 22.f, y), ImVec2(trackX - 8.f, y), ImGui::GetColorU32(ImGuiCol_Text));
                        std::array<char, kSmallBufSize> tb{};
                        std::snprintf(tb.data(), tb.size(), "%.0f", v);
                        ImVec2 ts = ImGui::CalcTextSize(tb.data());
                        dl->AddText(ImVec2(trackX - 24.f - ts.x, y - ts.y * 0.5f), ImGui::GetColorU32(ImGuiCol_Text), tb.data());
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
        }

        if (plotSize.x > 0.0f && plotSize.y > 0.0f && uiState.showOctaMapPreview)
        {
            ImGui::Spacing();
            {
                const char* title = "Octahedral Map";
                const ImVec2 titleSize = ImGui::CalcTextSize(title);
                const float titleX = ImMax(0.0f, (ImGui::GetContentRegionAvail().x - titleSize.x) * 0.5f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + titleX);
                ImGui::TextUnformatted(title);
                showHint("Octahedral map preview.");
            }
            ImGui::Image(info, plotSize, ImVec2(0.f, 0.5f), ImVec2(1.f, 1.f));
            showHint("Octahedral map preview.");
        }

        ImGui::Separator();
        draw3DControls();
        ImGui::Separator();

        if (!assetLabelPtrs.empty())
        {
            ImGui::TextUnformatted("profile");
            ImGui::SameLine();
            if (ImGui::ArrowButton("##profile_prev", ImGuiDir_Up))
            {
                activeIx = (activeIx + assetLabelPtrs.size() - 1u) % assetLabelPtrs.size();
                activeIxUi = static_cast<int>(activeIx);
            }
            ImGui::SameLine();
            if (ImGui::ArrowButton("##profile_next", ImGuiDir_Down))
            {
                activeIx = (activeIx + 1u) % assetLabelPtrs.size();
                activeIxUi = static_cast<int>(activeIx);
            }
            showHint("Select active IES profile. Use up/down arrows.");
            ImGui::NewLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
            if (ImGui::Combo("##profile", &activeIxUi, assetLabelPtrs.data(), static_cast<int>(assetLabelPtrs.size())))
                activeIx = static_cast<size_t>(activeIxUi);
            showHint("Select active IES profile.");
        }
    }
    ImGui::End();

    ies.zDegree = angle;
    uiState.activeAssetIx = activeIx;
	// 3D plot
	{
		info.textureID += device_base_t::MaxFramesInFlight;

		{
			ImGui::SetNextWindowPos(bottomPos, ImGuiCond_Always);
            ImGui::SetNextWindowSize(bottomSize, ImGuiCond_Always);

			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);

			ImGuiWindowFlags imgFlags =
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_NoMove |
				ImGuiWindowFlags_NoSavedSettings |
				ImGuiWindowFlags_NoBringToFrontOnFocus |
				ImGuiWindowFlags_NoNav |
				ImGuiWindowFlags_NoScrollbar |
				ImGuiWindowFlags_NoScrollWithMouse;

			if (ImGui::Begin("3D Plot", nullptr, imgFlags))
			{
                const ImVec2 avail = ImGui::GetContentRegionAvail();
                const ImVec2 plotSize(ImMax(0.0f, avail.x), ImMax(0.0f, avail.y));
				ImVec2 imgPos = ImGui::GetCursorScreenPos();
                ImGui::Image(info, plotSize);
                plotRectMin = ImGui::GetItemRectMin();
                plotRectMax = ImGui::GetItemRectMax();
                plotRectValid = true;
                plotHovered = ImGui::IsItemHovered();

                const float margin = 8.0f;
                const float barWidth = 16.0f;
                const float barHeight = ImMax(80.0f, plotSize.y - margin * 2.0f);
                if (plotSize.x > barWidth + margin * 2.0f && plotSize.y > margin * 2.0f)
                {
                    const bool useFalseColorLegend = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_FALSE_COLOR);
                    ImVec2 barMin(imgPos.x + plotSize.x - barWidth - margin, imgPos.y + margin);
                    ImVec2 barMax(barMin.x + barWidth, barMin.y + barHeight);

                    ImDrawList* dl = ImGui::GetWindowDrawList();
                    const int steps = 64;
                    for (int i = 0; i < steps; ++i)
                    {
                        const float t0 = float(i) / float(steps);
                        const float t1 = float(i + 1) / float(steps);
                        const float y0 = barMin.y + (1.0f - t1) * barHeight;
                        const float y1 = barMin.y + (1.0f - t0) * barHeight;
                        const float v = (t0 + t1) * 0.5f;
                        const ImU32 col = legendColor(v, useFalseColorLegend);
                        dl->AddRectFilled(ImVec2(barMin.x, y0), ImVec2(barMax.x, y1), col);
                    }
                    dl->AddRect(barMin, barMax, ImGui::GetColorU32(ImGuiCol_Border));

                    const ImU32 textCol = ImGui::GetColorU32(ImGuiCol_Text);
                    for (uint32_t i = 0u; i < hlsl::this_example::ies::FalseColorStopCount; ++i)
                    {
                        const float stop = hlsl::this_example::ies::falseColorStop(i);
                        const float y = barMin.y + (1.0f - stop) * barHeight;
                        dl->AddLine(ImVec2(barMin.x - 4.0f, y), ImVec2(barMin.x, y), textCol);
                        const float cdValue = stop * properties.maxCandelaValue;
                        std::array<char, kSmallBufSize> label{};
                        std::snprintf(label.data(), label.size(), "%.0f cd", cdValue);
                        ImVec2 labelSize = ImGui::CalcTextSize(label.data());
                        dl->AddText(ImVec2(barMin.x - labelSize.x - 6.0f, y - labelSize.y * 0.5f), textCol, label.data());
                    }
                }
			}
			ImGui::End();

			ImGui::PopStyleVar(2);
		}
	}

    if (plotRectValid && plotHovered && activeIx < m_assets.size())
    {
        const float plotW = plotRectMax.x - plotRectMin.x;
        const float plotH = plotRectMax.y - plotRectMin.y;
        const ImVec2 mousePos = ImGui::GetIO().MousePos;
        if (plotW > 1.0f && plotH > 1.0f &&
            mousePos.x >= plotRectMin.x && mousePos.x <= plotRectMax.x &&
            mousePos.y >= plotRectMin.y && mousePos.y <= plotRectMax.y)
        {
            const auto& iesCandela = m_assets[activeIx];
            const auto* profileCandela = iesCandela.getProfile();
            const auto& accessorCandela = profileCandela->getAccessor();
            const auto& resolutionCandela = accessorCandela.properties.optimalIESResolution;

            const float u = (mousePos.x - plotRectMin.x) / plotW;
            const float v = (mousePos.y - plotRectMin.y) / plotH;
            const float ndcX = u * 2.0f - 1.0f;
            const float ndcY = v * 2.0f - 1.0f;

            float32_t4x4 viewProj = camera.getConcatenatedMatrix();
            const auto invViewProj = inverse(viewProj);

            const float32_t4 nearPoint(ndcX, ndcY, 0.0f, 1.0f);
            const float32_t4 farPoint(ndcX, ndcY, 1.0f, 1.0f);
            auto nearWorld = mul(invViewProj, nearPoint);
            auto farWorld = mul(invViewProj, farPoint);
            nearWorld /= nearWorld.w;
            farWorld /= farWorld.w;

            using core_vec_t = std::remove_cv_t<std::remove_reference_t<decltype(camera.getPosition())>>;
            const auto toHlslVec3 = [](const core_vec_t& v)
            {
                return float32_t3(v.x, v.y, v.z);
            };

            const float32_t3 origin = toHlslVec3(camera.getPosition());
            const float32_t3 farPos = float32_t3(farWorld);
            float32_t3 direction = normalize(farPos - origin);

            float32_t3 hitPos(0.f);
            bool hit = false;
            const bool cubePlot = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_CUBE);
            if (cubePlot)
            {
                float tmin = -1.0e20f;
                float tmax = 1.0e20f;
                auto update = [&](float originAxis, float dirAxis) -> bool
                {
                    const float eps = 1.0e-6f;
                    if (abs(dirAxis) < eps)
                    {
                        if (originAxis < -m_plotRadius || originAxis > m_plotRadius)
                            return false;
                        return true;
                    }
                    float t1 = (-m_plotRadius - originAxis) / dirAxis;
                    float t2 = (m_plotRadius - originAxis) / dirAxis;
                    if (t1 > t2)
                    {
                        float tmp = t1;
                        t1 = t2;
                        t2 = tmp;
                    }
                    tmin = hlsl::max(tmin, t1);
                    tmax = hlsl::min(tmax, t2);
                    return tmin <= tmax;
                };

                if (update(origin.x, direction.x) && update(origin.y, direction.y) && update(origin.z, direction.z))
                {
                    const float t = (tmax < 0.0f) ? tmin : tmax;
                    if (t >= 0.0f)
                    {
                        hitPos = origin + direction * t;
                        hit = true;
                    }
                }
            }
            else
            {
                const float b = dot(origin, direction);
                const float c = dot(origin, origin) - m_plotRadius * m_plotRadius;
                const float disc = b * b - c;
                if (disc >= 0.0f)
                {
                    const float sqrtDisc = sqrt(disc);
                    const float tFar = -b + sqrtDisc;
                    const float tNear = -b - sqrtDisc;
                    const float t = (tFar < 0.0f) ? tNear : tFar;
                    if (t >= 0.0f)
                    {
                        hitPos = origin + direction * t;
                        hit = true;
                    }
                }
            }

            if (hit)
            {
                using octahedral_t = math::OctahedralTransform<float32_t>;
                const float32_t3 dir = normalize(hitPos);
                const uint32_t resX = resolutionCandela.x;
                const uint32_t resY = resolutionCandela.y;
                if (resX > 0u && resY > 0u)
                {
                    const float32_t2 res(static_cast<float>(resX), static_cast<float>(resY));
                    const float32_t2 halfMinusHalfPixel = float32_t2(0.5f, 0.5f) - float32_t2(0.5f, 0.5f) / res;
                    float32_t2 uv = octahedral_t::dirToUV(dir, halfMinusHalfPixel);
                    const bool interpolateCandela = uiState.mode.sphere.hasFlags(hlsl::this_example::ies::ESM_OCTAHEDRAL_UV_INTERPOLATE);
                    if (!interpolateCandela)
                    {
                        const auto pixel = floor(uv * res);
                        uv = (pixel + float32_t2(0.5f, 0.5f)) / res;
                    }

                    const auto texture = CIESProfile::texture_t::create(accessorCandela.properties.maxCandelaValue, resolutionCandela);
                    const float normalized = texture.__call(accessorCandela, uv);
                    candelaValue = texture.info.maxValueRecip > 0.0f ? (normalized / texture.info.maxValueRecip) : 0.0f;
                    candelaValid = true;
                }
            }
        }
    }

    if (candelaValid && !uiState.cameraControlEnabled)
    {
        ImGui::BeginTooltip();
        ImGui::Text("candela: %.3f cd", candelaValue);
        ImGui::EndTooltip();
    }
}



