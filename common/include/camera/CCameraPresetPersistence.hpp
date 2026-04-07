// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PRESET_PERSISTENCE_HPP_
#define _C_CAMERA_PRESET_PERSISTENCE_HPP_

#include <iosfwd>

#include "CCameraPreset.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

bool writeGoal(std::ostream& out, const core::CCameraGoal& goal, int indent = 2);
bool readGoal(std::istream& in, core::CCameraGoal& goal);

bool saveGoalToFile(const path& path, const core::CCameraGoal& goal, int indent = 2);
bool loadGoalFromFile(const path& path, core::CCameraGoal& goal);

bool writePreset(std::ostream& out, const core::CCameraPreset& preset, int indent = 2);
bool readPreset(std::istream& in, core::CCameraPreset& preset);

bool savePresetToFile(const path& path, const core::CCameraPreset& preset, int indent = 2);
bool loadPresetFromFile(const path& path, core::CCameraPreset& preset);

} // namespace nbl::system

#endif // _C_CAMERA_PRESET_PERSISTENCE_HPP_
