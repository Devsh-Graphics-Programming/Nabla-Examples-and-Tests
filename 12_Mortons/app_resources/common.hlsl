//// Copyright (C) 2023-2024 - DevSH Graphics Programming Sp. z O.O.
//// This file is part of the "Nabla Engine".
//// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_
#define _NBL_EXAMPLES_TESTS_12_MORTON_COMMON_INCLUDED_

// because DXC doesn't properly support `_Static_assert`
// TODO: add a message, and move to macros.h or cpp_compat
#define STATIC_ASSERT(...) { nbl::hlsl::conditional<__VA_ARGS__, int, void>::type a = 0; }

#include <boost/preprocessor.hpp>

#include <nbl/builtin/hlsl/morton.hlsl>

// tgmath.hlsl and intrinsics.hlsl tests

using namespace nbl::hlsl;
struct InputTestValues
{
	
};

struct TestValues
{

	void fillTestValues(NBL_CONST_REF_ARG(InputTestValues) input)
	{

	}
};

#endif
