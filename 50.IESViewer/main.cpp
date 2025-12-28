// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "App.hpp"

// TODO
#define APP_WINDOW_WIDTH 669*2u
#define APP_WINDOW_HEIGHT APP_WINDOW_WIDTH

#define APP_DEPTH_BUFFER_FORMAT EF_UNKNOWN

IESViewer::IESViewer(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
    : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),
    device_base_t({ APP_WINDOW_WIDTH, APP_WINDOW_HEIGHT }, APP_DEPTH_BUFFER_FORMAT, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
{

}

NBL_MAIN_FUNC(IESViewer)
