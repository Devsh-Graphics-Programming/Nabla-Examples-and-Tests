// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_EXAMPLES_API_HPP_
#define _NBL_EXAMPLES_API_HPP_

//! PCH for examples
/*
    PCH is compiled only once *if* an example can be promoted to use it, it is
    when its compile options & definitions set is the same as nblExamplesAPI's
    each example links to, otherwise it compiles its own PCH
*/
#include "nbl/examples/PCH.hpp"

//! common headers used across examples which cannot be part of PCH
/*
    NOTE: put here if a header requires defines which may be differ
*/

// broken? probably to refactor or even remove?
// #include "nbl/examples/geometry/CGeometryCreatorScene.hpp"


#endif // _NBL_EXAMPLES_API_HPP_