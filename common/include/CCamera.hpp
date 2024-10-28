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

	Camera(core::smart_refctd_ptr<typename traits_t::gimbal_t>&& gimbal, core::smart_refctd_ptr<typename traits_t::projection_t> projection, const float32_t3& target = { 0,0,0 })
        : base_t(core::smart_refctd_ptr(gimbal), core::smart_refctd_ptr(projection), target) 
    { 
        traits_t::controller_t::initKeysToEvent(); 
        base_t::recomputeViewMatrix();
    }
	~Camera() = default;

	virtual void manipulate(std::span<const typename traits_t::controller_virtual_event_t> virtualEvents) override
	{
        auto* gimbal = traits_t::controller_t::m_gimbal.get();
        auto* projection = base_t::getProjection();

        assert(gimbal); // TODO
        assert(projection); // TODO

        const auto [forward, up, right] = std::make_tuple(gimbal->getZAxis(), gimbal->getYAxis(), gimbal->getXAxis());
        const bool isLeftHanded = projection->isLeftHanded(); // TODO?

        const auto moveDirection = float32_t3(gimbal->getOrientation() * glm::vec3(0.0f, 0.0f, 1.0f));

        constexpr auto MoveSpeedScale = 0.003f;
        constexpr auto RotateSpeedScale = 0.003f;

        const auto dMoveFactor = traits_t::controller_t::m_moveSpeed * MoveSpeedScale;
        const auto dRotateFactor = traits_t::controller_t::m_rotateSpeed * RotateSpeedScale;

        // TODO: UB/LB for pitch [-88,88]!!! we are not in cosmos but handle FPS camera in default case
        // TODO: accumulate move & rotate scalars then do single move & rotate gimbal manipulation

        for (const traits_t::controller_virtual_event_t& ev : virtualEvents)
        {
            const float dMoveValue = ev.value * dMoveFactor;
            const float dRotateValue = ev.value * dRotateFactor;

            switch (ev.type)
            {
                case traits_t::controller_t::MoveForward:
                {
                    gimbal->move(moveDirection * dMoveValue);
                } break;

                case traits_t::controller_t::MoveBackward:
                {
                    gimbal->move(moveDirection * (-dMoveValue));
                } break;

                case traits_t::controller_t::MoveRight:
                {
                    gimbal->move(right * dMoveValue);
                } break;

                case traits_t::controller_t::MoveLeft:
                {
                    gimbal->move(right * (-dMoveValue));
                } break;

                case traits_t::controller_t::TiltUp:
                {
                    gimbal->rotate(right, dRotateValue);
                } break;

                case traits_t::controller_t::TiltDown:
                {
                    gimbal->rotate(right, -dRotateValue);
                } break;

                case traits_t::controller_t::PanRight:
                {
                    gimbal->rotate(up, dRotateValue);
                } break;

                case traits_t::controller_t::PanLeft:
                {
                    gimbal->rotate(up, -dRotateValue);
                } break;

                default:
                    continue;
            }
        }

        base_t::recomputeViewMatrix();
	}
};

}

#endif // _C_CAMERA_HPP_