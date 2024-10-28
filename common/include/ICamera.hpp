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
		using controller_virtual_event_t = typename controller_t::CVirtualEvent;
	};

	ICamera(core::smart_refctd_ptr<typename Traits::gimbal_t>&& gimbal, core::smart_refctd_ptr<typename Traits::projection_t> projection, const float32_t3& target = {0,0,0})
		: base_t(core::smart_refctd_ptr(gimbal)), m_projection(core::smart_refctd_ptr(projection)) { setTarget(target); }
	~ICamera() = default;

	inline void setPosition(const float32_t3& position)
	{
		const auto* gimbal = base_t::m_gimbal.get();
		gimbal->setPosition(position);
		recomputeViewMatrix();
	}

	inline void setTarget(const float32_t3& position)
	{
		m_target = position;
	
		const auto* gimbal = base_t::m_gimbal.get();
		auto localTarget = m_target - gimbal->getPosition();

		// TODO: use gimbal to perform a rotation!

		recomputeViewMatrix();
	}

	inline const float32_t3& getPosition() { return base_t::m_gimbal->getPosition(); }
	inline const float32_t3& getTarget() { return m_target; }
	inline const float32_t3x4& getViewMatrix() const { return m_viewMatrix; }
	inline Traits::projection_t* getProjection() { return m_projection.get(); }

protected:
	inline void recomputeViewMatrix()
	{
		// TODO: adjust for handedness (axes flip)
		const bool isLeftHanded = m_projection->isLeftHanded();
		const auto* gimbal = base_t::m_gimbal.get();
		const auto& position = gimbal->getPosition();

		const auto [xaxis, yaxis, zaxis] = std::make_tuple(gimbal->getXAxis(), gimbal->getYAxis(), gimbal->getZAxis());
		m_viewMatrix[0u] = float32_t4(xaxis, -hlsl::dot(xaxis, position));
		m_viewMatrix[1u] = float32_t4(yaxis, -hlsl::dot(yaxis, position));
		m_viewMatrix[2u] = float32_t4(zaxis, -hlsl::dot(zaxis, position));
	}

	const core::smart_refctd_ptr<typename Traits::projection_t> m_projection;
	float32_t3x4 m_viewMatrix;
	float32_t3 m_target;
};

}

#endif // _I_CAMERA_HPP_