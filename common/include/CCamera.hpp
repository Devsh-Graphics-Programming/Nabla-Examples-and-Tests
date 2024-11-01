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
        assert(gimbal);

        constexpr float MoveSpeedScale = 0.01f, RotateSpeedScale = 0.003f, MaxVerticalAngle = glm::radians(88.0f), MinVerticalAngle = -MaxVerticalAngle;
        const auto& gForward = gimbal->getZAxis(), gRight = gimbal->getXAxis();

        struct
        {
            float dPitch = 0.f, dYaw = 0.f;
            float32_t3 dMove = { 0.f, 0.f, 0.f };
        } accumulated;

        for (const auto& event : virtualEvents)
        {
            const float moveScalar = event.value * MoveSpeedScale;
            const float rotateScalar = event.value * RotateSpeedScale;

            switch (event.type)
            {
            case traits_t::controller_t::MoveForward:
                accumulated.dMove += gForward * moveScalar;
                break;
            case traits_t::controller_t::MoveBackward:
                accumulated.dMove -= gForward * moveScalar;
                break;
            case traits_t::controller_t::MoveRight:
                accumulated.dMove += gRight * moveScalar;
                break;
            case traits_t::controller_t::MoveLeft:
                accumulated.dMove -= gRight * moveScalar;
                break;
            case traits_t::controller_t::TiltUp:
                accumulated.dPitch += rotateScalar;
                break;
            case traits_t::controller_t::TiltDown:
                accumulated.dPitch -= rotateScalar;
                break;
            case traits_t::controller_t::PanRight:
                accumulated.dYaw += rotateScalar;
                break;
            case traits_t::controller_t::PanLeft:
                accumulated.dYaw -= rotateScalar;
                break;
            default:
                break;
            }
        }

        float currentPitch = atan2(glm::length(glm::vec2(gForward.x, gForward.z)), gForward.y) - glm::half_pi<float>();
        float currentYaw = atan2(gForward.x, gForward.z);

        currentPitch = std::clamp(currentPitch + accumulated.dPitch, MinVerticalAngle, MaxVerticalAngle);
        currentYaw += accumulated.dYaw;

        glm::quat orientation = glm::quat(glm::vec3(currentPitch, currentYaw, 0.0f));
        gimbal->setOrientation(orientation);
        gimbal->move(accumulated.dMove);

        base_t::recomputeViewMatrix();
    }
};

}

#endif // _C_CAMERA_HPP_