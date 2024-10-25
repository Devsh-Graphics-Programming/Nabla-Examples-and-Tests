// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_HPP_
#define _C_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

// FPS Camera
template<ProjectionMatrix T = float64_t4x4>
class Camera final : public ICamera<typename T>
{ 
public:
    using base_t = ICamera<typename T>;
    using traits_t = typename base_t::Traits;

	Camera() : base_t() { traits_t::controller_t::initKeysToEvent(); }
	~Camera() = default;

	virtual void manipulate(traits_t::gimbal_t* gimbal, std::span<const typename traits_t::controller_virtual_event_t> virtualEvents) override
	{
        if (!gimbal)
            return; // TODO: LOG

        if (!gimbal->isRecordingManipulation())
            return; // TODO: LOG

        const auto forward = gimbal->getForwardDirection();
        const auto up = gimbal->getPatchedUpVector();
        const bool leftHanded = gimbal->getProjection()->isLeftHanded();

        // strafe vector we move along when requesting left/right movements
        const auto strafeLeftRight = leftHanded ? glm::normalize(glm::cross(forward, up)) : glm::normalize(glm::cross(up, forward));

        constexpr auto MoveSpeedScale = 0.003f;
        constexpr auto RotateSpeedScale = 0.003f;

        const auto dMoveFactor = traits_t::controller_t::m_moveSpeed * MoveSpeedScale;
        const auto dRotateFactor = traits_t::controller_t::m_rotateSpeed * RotateSpeedScale;

        // TODO: UB/LB for pitch [-88,88]!!! we are not in cosmos but handle FPS camera in default case

        for (const traits_t::controller_virtual_event_t& ev : virtualEvents)
        {
            const auto dMoveValue = ev.value * dMoveFactor;
            const auto dRotateValue = ev.value * dRotateFactor;

            typename traits_t::gimbal_virtual_event_t gimbalEvent;

            switch (ev.type)
            {
            case traits_t::controller_t::MoveForward:
            {
                gimbalEvent.type = traits_t::gimbal_t::Strafe;
                gimbalEvent.manipulation.strafe.direction = forward;
                gimbalEvent.manipulation.strafe.distance = dMoveValue;
            } break;

            case traits_t::controller_t::MoveBackward:
            {
                gimbalEvent.type = traits_t::gimbal_t::Strafe;
                gimbalEvent.manipulation.strafe.direction = -forward;
                gimbalEvent.manipulation.strafe.distance = dMoveValue;
            } break;

            case traits_t::controller_t::MoveLeft:
            {
                gimbalEvent.type = traits_t::gimbal_t::Strafe;
                gimbalEvent.manipulation.strafe.direction = -strafeLeftRight;
                gimbalEvent.manipulation.strafe.distance = dMoveValue;
            } break;

            case traits_t::controller_t::MoveRight:
            {
                gimbalEvent.type = traits_t::gimbal_t::Strafe;
                gimbalEvent.manipulation.strafe.direction = strafeLeftRight;
                gimbalEvent.manipulation.strafe.distance = dMoveValue;
            } break;

            case traits_t::controller_t::TiltUp:
            {
                gimbalEvent.type = traits_t::gimbal_t::Rotate;
                gimbalEvent.manipulation.rotation.pitch = dRotateValue;
                gimbalEvent.manipulation.rotation.roll = 0.0f;
                gimbalEvent.manipulation.rotation.yaw = 0.0f;
            } break;

            case traits_t::controller_t::TiltDown:
            {
                gimbalEvent.type = traits_t::gimbal_t::Rotate;
                gimbalEvent.manipulation.rotation.pitch = -dRotateValue;
                gimbalEvent.manipulation.rotation.roll = 0.0f;
                gimbalEvent.manipulation.rotation.yaw = 0.0f;
            } break;

            case traits_t::controller_t::PanLeft:
            {
                gimbalEvent.type = traits_t::gimbal_t::Rotate;
                gimbalEvent.manipulation.rotation.pitch = 0.0f;
                gimbalEvent.manipulation.rotation.roll = 0.0f;
                gimbalEvent.manipulation.rotation.yaw = -dRotateValue;
            } break;

            case traits_t::controller_t::PanRight:
            {
                gimbalEvent.type = traits_t::gimbal_t::Rotate;
                gimbalEvent.manipulation.rotation.pitch = 0.0f;
                gimbalEvent.manipulation.rotation.roll = 0.0f;
                gimbalEvent.manipulation.rotation.yaw = dRotateValue;
            } break;

            default:
                continue;
            }

            gimbal->manipulate(gimbalEvent);
        }
	}
};

}

#endif // _C_CAMERA_HPP_