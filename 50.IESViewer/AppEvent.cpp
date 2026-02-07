// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"

void IESViewer::processMouse(const nbl::ui::IMouseEventChannel::range_t& events)
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        auto ev = *it;

        if (ev.type == nbl::ui::SMouseEvent::EET_SCROLL)
        {
            auto* cursorControl = m_window ? m_window->getCursorControl() : nullptr;
            if (!cursorControl || !uiState.plot2DRectValid)
                continue;
            const auto cursor = cursorControl->getPosition();
            const float cursorX = static_cast<float>(cursor.x);
            const float cursorY = static_cast<float>(cursor.y);
            if (cursorX < uiState.plot2DRectMin.x || cursorX > uiState.plot2DRectMax.x ||
                cursorY < uiState.plot2DRectMin.y || cursorY > uiState.plot2DRectMax.y)
                continue;

            auto& ies = m_assets[uiState.activeAssetIx];
            const auto& accessor = ies.getProfile()->getAccessor();

            auto impulse = ev.scrollEvent.verticalScroll * 0.02f;
            ies.zDegree = std::clamp<float>(ies.zDegree + impulse, accessor.hAngles.front(), accessor.hAngles.back());
        }
    }
}

void IESViewer::processKeyboard(const nbl::ui::IKeyboardEventChannel::range_t& events)
{
    for (auto it = events.begin(); it != events.end(); it++)
    {
        const auto ev = *it;

            if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
            {
            if (ev.keyCode == nbl::ui::EKC_UP_ARROW)
                uiState.activeAssetIx = std::clamp<size_t>(uiState.activeAssetIx + 1, 0, m_assets.size() - 1u);
            else if (ev.keyCode == nbl::ui::EKC_DOWN_ARROW)
                uiState.activeAssetIx = std::clamp<size_t>(uiState.activeAssetIx - 1, 0, m_assets.size() - 1u);

            auto& ies = m_assets[uiState.activeAssetIx];

            if (ev.keyCode == nbl::ui::EKC_C)
                uiState.mode.view = IES::EM_CDC;
            else if (ev.keyCode == nbl::ui::EKC_V)
                uiState.mode.view = IES::EM_OCTAHEDRAL_MAP;
            else if (ev.keyCode == nbl::ui::EKC_ESCAPE && uiState.cameraControlEnabled)
                uiState.cameraControlEnabled = false;
            else if (ev.keyCode == nbl::ui::EKC_SPACE)
                uiState.cameraControlEnabled = !uiState.cameraControlEnabled;

            if (ev.keyCode == nbl::ui::EKC_Q)
                requestExit();
        }
    }
}
