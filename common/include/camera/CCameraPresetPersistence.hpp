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

//! Serialize one camera goal into an existing stream.
bool writeGoal(std::ostream& out, const core::CCameraGoal& goal, int indent = 2);
//! Deserialize one camera goal from an existing stream.
bool readGoal(std::istream& in, core::CCameraGoal& goal);

//! Save one camera goal to a file.
bool saveGoalToFile(const path& path, const core::CCameraGoal& goal, int indent = 2);
//! Load one camera goal from a file.
bool loadGoalFromFile(const path& path, core::CCameraGoal& goal);

//! Serialize one camera preset into an existing stream.
bool writePreset(std::ostream& out, const core::CCameraPreset& preset, int indent = 2);
//! Deserialize one camera preset from an existing stream.
bool readPreset(std::istream& in, core::CCameraPreset& preset);

//! Save one camera preset to a file.
bool savePresetToFile(const path& path, const core::CCameraPreset& preset, int indent = 2);
//! Load one camera preset from a file.
bool loadPresetFromFile(const path& path, core::CCameraPreset& preset);

} // namespace nbl::system

#endif // _C_CAMERA_PRESET_PERSISTENCE_HPP_
