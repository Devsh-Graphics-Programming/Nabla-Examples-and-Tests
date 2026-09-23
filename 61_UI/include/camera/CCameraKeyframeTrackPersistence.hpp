// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// TODO: moved out of nbl::ext::Cameras into this example pending a rework of the camera tooling layer
// (goal / preset / keyframe / playback / persistence / follow / scripted runtime). It sits at global scope
// like the example's other headers. See README.md in this folder.

#ifndef _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
#define _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_

#include <string>
#include <string_view>

#include "CCameraKeyframeTrack.hpp"
#include "nbl/system/ISystem.h"
#include "nbl/system/path.h"

using namespace nbl;
using namespace nbl::ext::cameras;


struct CCameraKeyframeTrackPersistenceUtilities final
{
    /// @brief Serialize one camera keyframe track to JSON text.
    static std::string serializeKeyframeTrack(const CCameraKeyframeTrack& track, int indent = 2);
    /// @brief Deserialize one camera keyframe track from JSON text.
    static bool deserializeKeyframeTrack(std::string_view text, CCameraKeyframeTrack& track, std::string* error = nullptr);

    /// @brief Save one camera keyframe track to a file.
    static bool saveKeyframeTrackToFile(system::ISystem& system, const system::path& path, const CCameraKeyframeTrack& track, int indent = 2);
    /// @brief Load one camera keyframe track from a file.
    static bool loadKeyframeTrackFromFile(system::ISystem& system, const system::path& path, CCameraKeyframeTrack& track, std::string* error = nullptr);
};


#endif // _C_CAMERA_KEYFRAME_TRACK_PERSISTENCE_HPP_
