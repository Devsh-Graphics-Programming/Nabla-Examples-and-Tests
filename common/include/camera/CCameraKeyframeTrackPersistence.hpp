// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

#include <iosfwd>

#include "CCameraKeyframeTrack.hpp"
#include "nbl/system/path.h"

namespace nbl::system
{

class ISystem;

//! Serialize one camera keyframe track into an existing stream.
bool writeKeyframeTrack(std::ostream& out, const core::CCameraKeyframeTrack& track, int indent = 2);
//! Deserialize one camera keyframe track from an existing stream.
bool readKeyframeTrack(std::istream& in, core::CCameraKeyframeTrack& track);

//! Save one camera keyframe track to a file.
bool saveKeyframeTrackToFile(ISystem& system, const path& path, const core::CCameraKeyframeTrack& track, int indent = 2);
//! Load one camera keyframe track from a file.
bool loadKeyframeTrackFromFile(ISystem& system, const path& path, core::CCameraKeyframeTrack& track);

} // namespace nbl::system

#endif // _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
