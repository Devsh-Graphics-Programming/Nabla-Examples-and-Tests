// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PERSISTENCE_HPP_
#define _C_CAMERA_PERSISTENCE_HPP_

#include <iosfwd>
#include <span>
#include <vector>

#include "CCameraKeyframeTrackPersistence.hpp"
#include "CCameraPresetPersistence.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

bool writePresetCollection(std::ostream& out, std::span<const core::CCameraPreset> presets, int indent = 2);
bool readPresetCollection(std::istream& in, std::vector<core::CCameraPreset>& presets);

bool savePresetCollectionToFile(const path& path, std::span<const core::CCameraPreset> presets, int indent = 2);
bool loadPresetCollectionFromFile(const path& path, std::vector<core::CCameraPreset>& presets);

} // namespace nbl::system

#endif // _C_CAMERA_PERSISTENCE_HPP_
