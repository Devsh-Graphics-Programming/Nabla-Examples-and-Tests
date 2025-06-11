// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_PCH_HPP_
#define _NBL_EXAMPLES_PCH_HPP_

//! public declarations
/*
    NOTE: currently our whole public and private interface is broken
    and private headers leak to public includes
*/
#include <nabla.h>

#include "nbl/examples/common/SimpleWindowedApplication.hpp"
#include "nbl/examples/common/InputSystem.hpp"
#include "nbl/examples/common/CEventCallback.hpp"

#include "nbl/examples/cameras/CCamera.hpp"

//! note: one can add common std headers here not present in nabla.h or 
//! any headers shared between examples, you cannot put there include
//! files which require unique preprocessor definitions for each example

#endif // _NBL_EXAMPLES_COMMON_PCH_HPP_