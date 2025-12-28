// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"
#include "imgui/imgui_internal.h"
#include "app_resources/common.hlsl"
#include "app_resources/false_color.hlsl"
#include "app_resources/imgui.opts.hlsl"

using namespace this_example;

void IESViewer::uiListener()
{
    const auto resourceIx = m_realFrameIx % device_base_t::MaxFramesInFlight;

    SImResourceInfo info;
    info.textureID = ext::imgui::UI::FontAtlasTexId + resourceIx + 1u;
    info.samplerIx = (uint16_t)nbl::ext::imgui::UI::DefaultSamplerIx::USER;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImVec2 viewportPos = vp->Pos;
    const ImVec2 viewportSize = vp->Size;
    auto* cursorControl = m_window->getCursorControl();
    const auto cursorPosition = cursorControl ? cursorControl->getPosition() : nbl::ui::ICursorControl::SPosition{};
    const int32_t windowX = m_window->getX();
    const int32_t windowY = m_window->getY();
    const int32_t windowW = static_cast<int32_t>(m_window->getWidth());
    const int32_t windowH = static_cast<int32_t>(m_window->getHeight());
    const bool cursorInsideWindow = cursorControl &&
        cursorPosition.x >= windowX && cursorPosition.x < windowX + windowW &&
        cursorPosition.y >= windowY && cursorPosition.y < windowY + windowH;
    ImGui::GetIO().MouseDrawCursor = cursorInsideWindow && !m_cameraControlEnabled;
    const ImVec2 bottomSize(viewportSize.x, viewportSize.y);
    const ImVec2 bottomPos(viewportPos.x, viewportPos.y);
    const auto legendColor = [&](float v, bool useFalseColor) -> ImU32
    {
        const float clamped = ImClamp(v, 0.0f, 1.0f);
        if (useFalseColor)
        {
            const auto col = this_example::ies::falseColor(clamped);
            return ImGui::ColorConvertFloat4ToU32(ImVec4(col.x, col.y, col.z, 1.0f));
        }
        return ImGui::ColorConvertFloat4ToU32(ImVec4(clamped, clamped, clamped, 1.0f));
    };
    std::vector<const char*> assetLabelPtrs;
    assetLabelPtrs.reserve(m_assetLabels.size());
    for (const auto& label : m_assetLabels)
        assetLabelPtrs.push_back(label.c_str());

    size_t activeIx = m_activeAssetIx;
    if (activeIx >= m_assets.size())
        activeIx = 0u;
    int activeIxUi = static_cast<int>(activeIx);
    float candelaValue = 0.0f;
    bool candelaValid = false;
    ImVec2 plotRectMin(0.f, 0.f);
    ImVec2 plotRectMax(0.f, 0.f);
    bool plotRectValid = false;
    bool plotHovered = false;

    auto& ies = m_assets[activeIx];
    auto* profile = ies.getProfile();
	const auto& accessor = profile->getAccessor();
    const auto& properties = accessor.getProperties();

    const float lowerBound = accessor.hAngles.front();
    const float upperBound = accessor.hAngles.back();
    const bool singleAngle = (upperBound == lowerBound);

    constexpr float kMinFlatten = 0.0f;
    auto angle = ImClamp(ies.zDegree, lowerBound, upperBound);

    auto updateCameraProjection = [&]()
    {
        if (m_plot3DWidth == 0u || m_plot3DHeight == 0u)
            return;
        const float aspect = float(m_plot3DWidth) / float(m_plot3DHeight);
        auto projectionMatrix = core::matrix4SIMD::buildProjectionMatrixPerspectiveFovLH(core::radians(m_cameraFovDeg), aspect, 0.1f, 10000.0f);
        camera.setProjectionMatrix(projectionMatrix);
    };

    auto draw3DControls = [&](float controlWidth)
    {
        bool interpolateCandela =
            mode.sphere.hasFlags(this_example::ies::ESM_OCTAHEDRAL_UV_INTERPOLATE);

        if (ImGui::Checkbox("interpolate candelas", &interpolateCandela))
        {
            if (interpolateCandela)
                mode.sphere |= this_example::ies::E_SPHERE_MODE::ESM_OCTAHEDRAL_UV_INTERPOLATE;
            else
                mode.sphere &= static_cast<this_example::ies::E_SPHERE_MODE>(
                    ~this_example::ies::E_SPHERE_MODE::ESM_OCTAHEDRAL_UV_INTERPOLATE
                );
        }

        bool falseColor =
            mode.sphere.hasFlags(this_example::ies::ESM_FALSE_COLOR);

        if (ImGui::Checkbox("false color", &falseColor))
        {
            if (falseColor)
                mode.sphere |= this_example::ies::E_SPHERE_MODE::ESM_FALSE_COLOR;
            else
                mode.sphere &= static_cast<this_example::ies::E_SPHERE_MODE>(
                    ~this_example::ies::E_SPHERE_MODE::ESM_FALSE_COLOR
                );
        }

        bool showOctaMap = m_showOctaMapPreview;
        if (ImGui::Checkbox("octahedral map", &showOctaMap))
            m_showOctaMapPreview = showOctaMap;

        bool cubePlot =
            mode.sphere.hasFlags(this_example::ies::ESM_CUBE);

        if (ImGui::Checkbox("cube plot", &cubePlot))
        {
            if (cubePlot)
                mode.sphere |= this_example::ies::E_SPHERE_MODE::ESM_CUBE;
            else
                mode.sphere &= static_cast<this_example::ies::E_SPHERE_MODE>(
                    ~this_example::ies::E_SPHERE_MODE::ESM_CUBE
                );
        }

        bool wireframe = m_wireframeEnabled;
        if (ImGui::Checkbox("wireframe", &wireframe))
            m_wireframeEnabled = wireframe;

        bool cameraControl = m_cameraControlEnabled;
        if (ImGui::Checkbox("camera control (space)", &cameraControl))
            m_cameraControlEnabled = cameraControl;

        bool speedChanged = false;
        ImGui::SetNextItemWidth(controlWidth);
        speedChanged |= ImGui::SliderFloat("move speed", &m_cameraMoveSpeed, 0.1f, 10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetNextItemWidth(controlWidth);
        speedChanged |= ImGui::SliderFloat("rotate speed", &m_cameraRotateSpeed, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
        if (speedChanged && m_cameraControlEnabled)
        {
            camera.setMoveSpeed(m_cameraMoveSpeed);
            camera.setRotateSpeed(m_cameraRotateSpeed);
        }

        bool fovChanged = false;
        ImGui::SetNextItemWidth(controlWidth);
        fovChanged |= ImGui::SliderFloat("fov", &m_cameraFovDeg, 30.0f, 120.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
        if (fovChanged)
            updateCameraProjection();

        float flatten = ImClamp(ies.flatten, kMinFlatten, 1.0f);
        bool flattenChanged = false;
        ImGui::SetNextItemWidth(controlWidth);
        flattenChanged |= ImGui::SliderFloat("flatten", &flatten, kMinFlatten, 1.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(64.0f);
        flattenChanged |= ImGui::InputFloat("##flatten_value", &flatten, 0.0f, 0.0f, "%.3f");
        if (flattenChanged)
        {
            flatten = ImClamp(flatten, kMinFlatten, 1.0f);
            ies.flatten = flatten;
            if (m_activeAssetIx < m_candelaDirty.size())
                m_candelaDirty[m_activeAssetIx] = true;
            auto* mapped = reinterpret_cast<IESTextureInfo*>(
                reinterpret_cast<uint8_t*>(ies.buffers.textureInfo.buffer->getBoundMemory().memory->getMappedPointer()) +
                ies.buffers.textureInfo.offset);
            const auto& resolution = accessor.properties.optimalIESResolution;
            *mapped = CIESProfile::texture_t::createInfo(accessor, resolution, ies.flatten, true);

            auto bound = ies.buffers.textureInfo.buffer->getBoundMemory();
            if (bound.memory->haveToMakeVisible())
            {
                const ILogicalDevice::MappedMemoryRange range(
                    bound.memory,
                    bound.offset + ies.buffers.textureInfo.offset,
                    sizeof(IESTextureInfo));
                m_device->flushMappedMemoryRanges(1, &range);
            }
        }
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

        char b1[64];
        snprintf(b1, sizeof(b1), "%.3f deg", angle);
        if (ImGui::BeginTable("##profile_info", 2, ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(IES::symmetryToRS(properties.getSymmetry()));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(IES::typeToRS(properties.getType()));

            ImGui::TableNextColumn();
            ImGui::TextUnformatted(IES::versionToRS(properties.getVersion()));
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(assetLabelPtrs.empty() ? ies.key.c_str() : assetLabelPtrs[activeIx]);

            ImGui::TableNextColumn();
            ImGui::Text("angles: %u x %u", accessor.hAnglesCount(), accessor.vAnglesCount());
            ImGui::TableNextColumn();
            ImGui::Text("resolution: %u x %u", resolution.x, resolution.y);

            ImGui::TableNextColumn();
            ImGui::Text("max cd: %.3f", properties.maxCandelaValue);
            ImGui::TableNextColumn();
            ImGui::Text("avg: %.3f", properties.avgEmmision);

            ImGui::TableNextColumn();
            ImGui::Text("avg full: %.3f", properties.fullDomainAvgEmission);
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(b1);

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
                const char* title = IES::modeToRS(mode.view);
                const ImVec2 titleSize = ImGui::CalcTextSize(title);
                const float titleX = ImMax(0.0f, (ImGui::GetContentRegionAvail().x - titleSize.x) * 0.5f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + titleX);
                ImGui::TextUnformatted(title);
            }

            plotPos = ImGui::GetCursorScreenPos();
            ImGui::Image(info, plotSize, ImVec2(0.f, 0.f), ImVec2(1.f, 0.5f));

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
        }

        if (plotSize.x > 0.0f && plotSize.y > 0.0f && m_showOctaMapPreview)
        {
            ImGui::Spacing();
            {
                const char* title = "Octahedral Map";
                const ImVec2 titleSize = ImGui::CalcTextSize(title);
                const float titleX = ImMax(0.0f, (ImGui::GetContentRegionAvail().x - titleSize.x) * 0.5f);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + titleX);
                ImGui::TextUnformatted(title);
            }
            ImGui::Image(info, plotSize, ImVec2(0.f, 0.5f), ImVec2(1.f, 1.f));
        }

        ImGui::Separator();
        draw3DControls(ImMax(120.0f, ImMin(panelWidth - panelMargin * 2.0f, 260.0f)));
        ImGui::Separator();

        if (!assetLabelPtrs.empty())
        {
            ImGui::SetNextItemWidth(ImMin(260.0f, panelWidth - panelMargin * 2.0f));
            if (ImGui::Combo("profile", &activeIxUi, assetLabelPtrs.data(), static_cast<int>(assetLabelPtrs.size())))
                activeIx = static_cast<size_t>(activeIxUi);
        }
    }
    ImGui::End();

    ies.zDegree = angle;
    m_activeAssetIx = activeIx;
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
                    const bool useFalseColorLegend = mode.sphere.hasFlags(this_example::ies::ESM_FALSE_COLOR);
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
                    for (uint32_t i = 0u; i < this_example::ies::FalseColorStopCount; ++i)
                    {
                        const float stop = this_example::ies::falseColorStop(i);
                        const float y = barMin.y + (1.0f - stop) * barHeight;
                        dl->AddLine(ImVec2(barMin.x - 4.0f, y), ImVec2(barMin.x, y), textCol);
                        const float cdValue = stop * properties.maxCandelaValue;
                        char label[32];
                        snprintf(label, sizeof(label), "%.0f cd", cdValue);
                        ImVec2 labelSize = ImGui::CalcTextSize(label);
                        dl->AddText(ImVec2(barMin.x - labelSize.x - 6.0f, y - labelSize.y * 0.5f), textCol, label);
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
            const auto& propertiesCandela = accessorCandela.getProperties();
            const auto& resolutionCandela = accessorCandela.properties.optimalIESResolution;

            const float u = (mousePos.x - plotRectMin.x) / plotW;
            const float v = (mousePos.y - plotRectMin.y) / plotH;
            const float ndcX = u * 2.0f - 1.0f;
            const float ndcY = v * 2.0f - 1.0f;

            core::matrix4SIMD invViewProj;
            if (camera.getConcatenatedMatrix().getInverseTransform(invViewProj))
            {
                core::vectorSIMDf nearPoint(ndcX, ndcY, 0.0f, 1.0f);
                core::vectorSIMDf farPoint(ndcX, ndcY, 1.0f, 1.0f);
                invViewProj.transformVect(nearPoint);
                invViewProj.transformVect(farPoint);
                nearPoint /= nearPoint.wwww();
                farPoint /= farPoint.wwww();

                const core::vectorSIMDf origin = camera.getPosition();
                core::vectorSIMDf direction = farPoint - origin;
                direction.makeSafe3D();
                direction = core::normalize(direction);

                core::vectorSIMDf hitPos;
                bool hit = false;
                const bool cubePlot = mode.sphere.hasFlags(this_example::ies::ESM_CUBE);
                if (cubePlot)
                {
                    float tmin = -1.0e20f;
                    float tmax = 1.0e20f;
                    auto update = [&](float originAxis, float dirAxis) -> bool
                    {
                        const float eps = 1.0e-6f;
                        if (core::abs<float>(dirAxis) < eps)
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
                        tmin = core::max(tmin, t1);
                        tmax = core::min(tmax, t2);
                        return tmin <= tmax;
                    };

                    if (update(origin.x, direction.x) && update(origin.y, direction.y) && update(origin.z, direction.z))
                    {
                        float t = tmax;
                        if (t < 0.0f)
                            t = tmin;
                        if (t >= 0.0f)
                        {
                            hitPos = origin + direction * t;
                            hit = true;
                        }
                    }
                }
                else
                {
                    const float b = core::dot(origin, direction)[0];
                    const float c = core::dot(origin, origin)[0] - m_plotRadius * m_plotRadius;
                    const float disc = b * b - c;
                    if (disc >= 0.0f)
                    {
                        const float sqrtDisc = core::sqrt<float>(disc);
                        float t = -b + sqrtDisc;
                        if (t < 0.0f)
                            t = -b - sqrtDisc;
                        if (t >= 0.0f)
                        {
                            hitPos = origin + direction * t;
                            hit = true;
                        }
                    }
                }

                if (hit)
                {
                    core::vectorSIMDf dir = core::normalize(hitPos);
                    const float sum = core::abs<float>(dir.x) + core::abs<float>(dir.y) + core::abs<float>(dir.z);
                    core::vectorSIMDf s = dir / sum;
                    if (s.z < 0.0f)
                    {
                        const float sx = s.x;
                        const float sy = s.y;
                        s.x = (sx < 0.0f ? -1.0f : 1.0f) * (1.0f - core::abs<float>(sy));
                        s.y = (sy < 0.0f ? -1.0f : 1.0f) * (1.0f - core::abs<float>(sx));
                    }

                    float uvx = s.x * 0.5f + 0.5f;
                    float uvy = s.y * 0.5f + 0.5f;

                    const uint32_t resX = resolutionCandela.x;
                    const uint32_t resY = resolutionCandela.y;
                    if (resX > 0u && resY > 0u)
                    {
                        const float resFx = static_cast<float>(resX);
                        const float resFy = static_cast<float>(resY);

                        const bool interpolateCandela = mode.sphere.hasFlags(this_example::ies::ESM_OCTAHEDRAL_UV_INTERPOLATE);
                        if (!interpolateCandela)
                        {
                            float px = core::floor<float>(uvx * resFx + 0.5f);
                            float py = core::floor<float>(uvy * resFy + 0.5f);
                            uvx = px / resFx;
                            uvy = py / resFy;
                        }

                        const float scaleX = 1.0f - 1.0f / resFx;
                        const float scaleY = 1.0f - 1.0f / resFy;
                        const float uvCornerX = (uvx - 0.5f) * scaleX + 0.5f;
                        const float uvCornerY = (uvy - 0.5f) * scaleY + 0.5f;

                        const float tx = uvCornerX * resFx - 0.5f;
                        const float ty = uvCornerY * resFy - 0.5f;

                        int x0 = static_cast<int>(core::floor<float>(tx));
                        int y0 = static_cast<int>(core::floor<float>(ty));
                        int x1 = x0 + 1;
                        int y1 = y0 + 1;
                        const float fx = tx - static_cast<float>(x0);
                        const float fy = ty - static_cast<float>(y0);

                        x0 = ImClamp(x0, 0, static_cast<int>(resX - 1u));
                        y0 = ImClamp(y0, 0, static_cast<int>(resY - 1u));
                        x1 = ImClamp(x1, 0, static_cast<int>(resX - 1u));
                        y1 = ImClamp(y1, 0, static_cast<int>(resY - 1u));

                        const auto info = CIESProfile::texture_t::createInfo(accessorCandela, resolutionCandela, iesCandela.flatten, true);
                        const auto sample = [&](int x, int y) -> float
                        {
                            return CIESProfile::texture_t::eval(accessorCandela, info, nbl::hlsl::uint32_t2(static_cast<uint32_t>(x), static_cast<uint32_t>(y)));
                        };

                        const float c00 = sample(x0, y0);
                        const float c10 = sample(x1, y0);
                        const float c01 = sample(x0, y1);
                        const float c11 = sample(x1, y1);

                        const float cx0 = c00 + (c10 - c00) * fx;
                        const float cx1 = c01 + (c11 - c01) * fx;
                        const float c = cx0 + (cx1 - cx0) * fy;

                        candelaValue = c * propertiesCandela.maxCandelaValue;
                        candelaValid = true;
                    }
                }
            }
        }
    }

    if (candelaValid && !m_cameraControlEnabled)
    {
        ImGui::BeginTooltip();
        ImGui::Text("candela: %.3f cd", candelaValue);
        ImGui::EndTooltip();
    }
}
