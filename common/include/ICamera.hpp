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

template<ProjectionMatrix T = float64_t4x4>
class ICamera : public ICameraController<typename T>
{ 
public:
	using base_t = typename ICameraController<typename T>;

	struct Traits
	{
		using controller_t = base_t;
		using projection_t = typename controller_t::projection_t;
		using gimbal_t = typename controller_t::CGimbal;
		using gimbal_virtual_event_t = typename gimbal_t::CVirtualEvent;
		using controller_virtual_event_t = typename controller_t::CVirtualEvent;
	};

	ICamera() : base_t() {}
	~ICamera() = default;
};

}

#endif // _I_CAMERA_HPP_