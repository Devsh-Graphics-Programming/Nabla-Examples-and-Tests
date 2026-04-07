// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_
#define _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_

#include <iosfwd>
#include <string>

#include "CCameraSequenceScript.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

bool readCameraSequenceScript(std::istream& in, core::CCameraSequenceScript& out, std::string* error = nullptr);
bool loadCameraSequenceScriptFromFile(const path& path, core::CCameraSequenceScript& out, std::string* error = nullptr);

} // namespace nbl::system

#endif // _C_CAMERA_SEQUENCE_SCRIPT_PERSISTENCE_HPP_
