// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_PCH_HPP_
#define _NBL_EXAMPLES_PCH_HPP_

//! public Nabla declarations
/*
    NOTE: currently our whole public and private interface is broken
    and private headers leak to public includes
*/
#include "nabla.h"

//! common example headers

// why isnt this in `nabla.h` ?
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "nbl/examples/common/SimpleWindowedApplication.hpp"
#include "nbl/examples/common/MonoWindowApplication.hpp"
#include "nbl/examples/common/InputSystem.hpp"
#include "nbl/examples/common/CEventCallback.hpp"

#include "nbl/examples/cameras/CCamera.hpp"

#include "nbl/examples/geometry/CGeometryCreatorScene.hpp"
#include "nbl/examples/geometry/CSimpleDebugRenderer.hpp"


#endif // _NBL_EXAMPLES_COMMON_PCH_HPP_