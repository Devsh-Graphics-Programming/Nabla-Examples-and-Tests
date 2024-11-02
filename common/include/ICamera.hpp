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
		using projection_t = typename controller_t::projection_t;
		using gimbal_t = typename controller_t::CGimbal;
		using controller_virtual_event_t = typename controller_t::CVirtualEvent;
		using matrix_precision_t = typename T; // TODO: actually all vectors/scalars should have precision type T and because of projection matrix constraints allowed is only float32_t & float64_t
	};

	ICamera(core::smart_refctd_ptr<typename Traits::projection_t> projection)
		: base_t(), m_projection(core::smart_refctd_ptr(projection)) {}
	~ICamera() = default;

	// NOTE: I dont like to make it virtual but if we assume we can 
	// have more gimbals & we dont store single one in the interface
	// then one of them must be the one we model the camera view with
	// eg "Follow Camera" -> range of 2 gimbals but only first
	// models the camera view
	virtual const Traits::gimbal_t& getGimbal() = 0u;
	
	inline const matrix<typename Traits::matrix_precision_t, 3, 4>& getViewMatrix() const { return m_viewMatrix; }
	inline Traits::projection_t* getProjection() { return m_projection.get(); }

protected:
	// Recomputes view matrix for a given gimbal. Note that a camera type implementation could have multiple gimbals - not all of them will be used to model the camera view itself but one 
	// (TODO: (*) unless? I guess its to decide if we talk bout single view or to allow to have view per gimbal, but imho we should have 1 -> could just spam more instances of camera type to cover more views)
	inline void recomputeViewMatrix(Traits::gimbal_t& gimbal)
	{
		gimbal.computeViewMatrix(m_viewMatrix, m_projection->isLeftHanded());
	}

	const core::smart_refctd_ptr<typename Traits::projection_t> m_projection; // TODO: move it from here
	matrix<typename Traits::matrix_precision_t, 3, 4> m_viewMatrix;
};

}

#endif // _I_CAMERA_HPP_