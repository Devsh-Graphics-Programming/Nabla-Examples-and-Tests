// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _CAMERA_IMPL_
#define _CAMERA_IMPL_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include "camera/ICameraControl.hpp"

// FPS Camera, we will have more types soon

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

template<ProjectionMatrix T = float64_t4x4>
class Camera : public ICameraController<typename T>
{ 
public:
	using matrix_t = typename T;
	using base_t = typename ICameraController<typename T>;
	using gimbal_t = typename base_t::CGimbal;
    using gimbal_virtual_event_t = typename gimbal_t::CVirtualEvent;
	using controller_virtual_event_t = typename base_t::CVirtualEvent;

	Camera(core::smart_refctd_ptr<gimbal_t>&& gimbal)
		: base_t(core::smart_refctd_ptr(gimbal)) {}
	~Camera() = default;

public:

	void manipulate(base_t::SUpdateParameters parameters)
	{
		auto* gimbal = base_t::m_gimbal.get();

		auto process = [&](const std::vector<base_t::CVirtualEvent>& virtualEvents) -> void
		{
			const auto forward = gimbal->getForwardDirection();
			const auto up = gimbal->getPatchedUpVector();
			const bool leftHanded = gimbal->isLeftHanded();

			// strafe vector we move along when requesting left/right movements
			const auto strafeLeftRight = leftHanded ? glm::normalize(glm::cross(forward, up)) : glm::normalize(glm::cross(up, forward));

			constexpr auto MoveSpeedScale = 0.003f;
			constexpr auto RotateSpeedScale = 0.003f;

			const auto dMoveFactor = base_t::m_moveSpeed * MoveSpeedScale;
			const auto dRotateFactor = base_t::m_rotateSpeed * RotateSpeedScale;

			// TODO: UB/LB for pitch [-88,88]!!! we are not in cosmos but handle FPS camera

			for (const controller_virtual_event_t& ev : virtualEvents)
			{
                const auto dMoveValue = ev.value * dMoveFactor;
                const auto dRotateValue = ev.value * dRotateFactor;

                gimbal_virtual_event_t gimbalEvent;

                switch (ev.type)
                {
                    case base_t::MoveForward:
                    {
                        gimbalEvent.type = gimbal_t::Strafe;
                        gimbalEvent.manipulation.strafe.direction = forward;
                        gimbalEvent.manipulation.strafe.distance = dMoveValue;
                    } break;

                    case base_t::MoveBackward:
                    {
                        gimbalEvent.type = gimbal_t::Strafe;
                        gimbalEvent.manipulation.strafe.direction = -forward;
                        gimbalEvent.manipulation.strafe.distance = dMoveValue;
                    } break;

                    case base_t::MoveLeft:
                    {
                        gimbalEvent.type = gimbal_t::Strafe;
                        gimbalEvent.manipulation.strafe.direction = -strafeLeftRight;
                        gimbalEvent.manipulation.strafe.distance = dMoveValue;
                    } break;

                    case base_t::MoveRight:
                    {
                        gimbalEvent.type = gimbal_t::Strafe;
                        gimbalEvent.manipulation.strafe.direction = strafeLeftRight;
                        gimbalEvent.manipulation.strafe.distance = dMoveValue;
                    } break;

                    case base_t::TiltUp:
                    {
                        gimbalEvent.type = gimbal_t::Rotate;
                        gimbalEvent.manipulation.rotation.pitch = dRotateValue;
                        gimbalEvent.manipulation.rotation.roll = 0.0f;
                        gimbalEvent.manipulation.rotation.yaw = 0.0f;
                    } break;

                    case base_t::TiltDown:
                    {
                        gimbalEvent.type = gimbal_t::Rotate;
                        gimbalEvent.manipulation.rotation.pitch = -dRotateValue;
                        gimbalEvent.manipulation.rotation.roll = 0.0f;
                        gimbalEvent.manipulation.rotation.yaw = 0.0f;
                    } break;

                    case base_t::PanLeft:
                    {
                        gimbalEvent.type = gimbal_t::Rotate;
                        gimbalEvent.manipulation.rotation.pitch = 0.0f;
                        gimbalEvent.manipulation.rotation.roll = 0.0f;
                        gimbalEvent.manipulation.rotation.yaw = -dRotateValue;
                    } break;

                    case base_t::PanRight:
                    {
                        gimbalEvent.type = gimbal_t::Rotate;
                        gimbalEvent.manipulation.rotation.pitch = 0.0f;
                        gimbalEvent.manipulation.rotation.roll = 0.0f;
                        gimbalEvent.manipulation.rotation.yaw = dRotateValue;
                    } break;

                    default:
                        continue;
                }

                gimbal->manipulate(gimbalEvent);
			}
		};

		gimbal->begin();
		{
			process(base_t::processMouse(parameters.mouseEvents));
			process(base_t::processKeyboard(parameters.keyboardEvents));
		}
		gimbal->end();
	}
};

}

#endif // _CAMERA_IMPL_