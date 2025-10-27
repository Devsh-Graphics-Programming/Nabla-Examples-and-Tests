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
            auto& ies = assets[activeAssetIx];
            auto* profile = ies.getProfile();

            auto impulse = ev.scrollEvent.verticalScroll * 0.02f;
            ies.zDegree = std::clamp<float>(ies.zDegree + impulse, profile->getHoriAngles().front(), profile->getHoriAngles().back());
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
                activeAssetIx = std::clamp<size_t>(activeAssetIx + 1, 0, assets.size() - 1u);
            else if (ev.keyCode == nbl::ui::EKC_DOWN_ARROW)
                activeAssetIx = std::clamp<size_t>(activeAssetIx - 1, 0, assets.size() - 1u);

            auto& ies = assets[activeAssetIx];

            if (ev.keyCode == nbl::ui::EKC_C)
                ies.mode = IES::EM_CDC;
            else if (ev.keyCode == nbl::ui::EKC_V)
                ies.mode = IES::EM_IES_C;
            else if (ev.keyCode == nbl::ui::EKC_S)
                ies.mode = IES::EM_SPERICAL_C;
            else if (ev.keyCode == nbl::ui::EKC_D)
                ies.mode = IES::EM_DIRECTION;
            else if (ev.keyCode == nbl::ui::EKC_M)
                ies.mode = IES::EM_PASS_T_MASK;

            if (ev.keyCode == nbl::ui::EKC_Q)
                running = false;
        }
    }
}