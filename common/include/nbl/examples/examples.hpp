// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_HPP_
#define _NBL_EXAMPLES_HPP_


//! Precompiled header shared across all examples
#include "nbl/examples/PCH.hpp"

//! Example specific headers that must not be included in the PCH
/*
    NOTE: Add here if they depend on preprocessor definitions
    or macros that are specific to individual example targets
    (eg. defined in CMake)
*/

// #include "..."

// cannot be in PCH because depens on definition of `this_example` for Example's builtins
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

#define NBL_EXPOSE_NAMESPACES \
using namespace nbl; \
using namespace core; \
using namespace hlsl; \
using namespace system; \
using namespace asset; \
using namespace ui; \
using namespace video; \
using namespace scene; \
using namespace nbl::examples;

#endif // _NBL_EXAMPLES_HPP_