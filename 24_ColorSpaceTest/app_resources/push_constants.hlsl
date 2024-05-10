// Copyright (C) 2024-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// Tests for combined and separable image samplers. Make sure exactly one is defined
#define COMBINED_IMMUTABLE
//#define COMBINED_MUTABLE
//#define SEPARATED_IMMUTABLE
//#define SEPARATED_MUTABLE

struct push_constants_t
{
	uint16_t2 grid;
};