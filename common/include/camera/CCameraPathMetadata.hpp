// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_PATH_METADATA_HPP_
#define _C_CAMERA_PATH_METADATA_HPP_

#include <string_view>

namespace nbl::core
{

struct SCameraPathRigMetadata final
{
    static inline constexpr std::string_view KindLabel = "Path Rig";
    static inline constexpr std::string_view KindDescription = "Path-model camera with typed s/u/v/roll state";
    static inline constexpr std::string_view Identifier = "Target-relative Path Rig";
    static inline constexpr std::string_view DefaultModelDescription = "Adjust a target-relative path rig with s/u/v/roll state";
};

} // namespace nbl::core

#endif // _C_CAMERA_PATH_METADATA_HPP_
