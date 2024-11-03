// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include "camera/ICameraControl.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

template<typename T>
class ICamera : public ICameraController<T>
{ 
public:
	using base_t = typename ICameraController<T>;

	struct Traits
	{
		using controller_t = base_t;
		using gimbal_t = typename controller_t::CGimbal;
		using matrix_precision_t = typename T; // TODO: actually all vectors/scalars should have precision type T and because of projection matrix constraints allowed is only float32_t & float64_t
	};

	ICamera() : base_t() {}
	~ICamera() = default;

	// Returns a gimbal which *models the camera view*, note that a camera type implementation may have multiple gimbals under the hood
	virtual const Traits::gimbal_t& getGimbal() = 0u;
};

}

#endif // _I_CAMERA_HPP_