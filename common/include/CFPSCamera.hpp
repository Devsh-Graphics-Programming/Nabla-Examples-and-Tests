// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_HPP_
#define _C_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

// FPS Camera
template<typename T = float64_t>
class CFPSCamera final : public ICamera<T>
{ 
public:
    using base_t = ICamera<T>;
    using traits_t = typename base_t::Traits;

    CFPSCamera(const float32_t3& position, glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
        : base_t(), m_gimbal({ .position = position, .orientation = orientation, .withView = true })
    { 
        initKeysToEvent(); 
    }
	~CFPSCamera() = default;

    const typename traits_t::gimbal_t& getGimbal() override
    {
        return m_gimbal;
    }

    virtual void manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        if (!virtualEvents.size())
            return;

        constexpr float MoveSpeedScale = 0.01f, RotateSpeedScale = 0.003f, MaxVerticalAngle = glm::radians(88.0f), MinVerticalAngle = -MaxVerticalAngle;
        const auto& gForward = m_gimbal.getZAxis(), gRight = m_gimbal.getXAxis();

        struct
        {
            float dPitch = 0.f, dYaw = 0.f;
            float32_t3 dMove = { 0.f, 0.f, 0.f };
        } accumulated;

        for (const auto& event : virtualEvents)
        {
            const float moveScalar = event.magnitude * MoveSpeedScale;
            const float rotateScalar = event.magnitude * RotateSpeedScale;

            switch (event.type)
            {
            case CVirtualGimbalEvent::MoveForward:
                accumulated.dMove += gForward * moveScalar;
                break;
            case CVirtualGimbalEvent::MoveBackward:
                accumulated.dMove -= gForward * moveScalar;
                break;
            case CVirtualGimbalEvent::MoveRight:
                accumulated.dMove += gRight * moveScalar;
                break;
            case CVirtualGimbalEvent::MoveLeft:
                accumulated.dMove -= gRight * moveScalar;
                break;
            case CVirtualGimbalEvent::TiltUp:
                accumulated.dPitch += rotateScalar;
                break;
            case CVirtualGimbalEvent::TiltDown:
                accumulated.dPitch -= rotateScalar;
                break;
            case CVirtualGimbalEvent::PanRight:
                accumulated.dYaw += rotateScalar;
                break;
            case CVirtualGimbalEvent::PanLeft:
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

        m_gimbal.begin();
        {
            m_gimbal.setOrientation(orientation);
            m_gimbal.move(accumulated.dMove);
        }
        m_gimbal.end();
    }

private:
    void initKeysToEvent() override
    {
        traits_t::controller_t::updateKeysToEvent([](CVirtualGimbalEvent::keys_to_virtual_events_t& keys)
        {
            keys[CVirtualGimbalEvent::MoveForward] = ui::E_KEY_CODE::EKC_W;
            keys[CVirtualGimbalEvent::MoveBackward] = ui::E_KEY_CODE::EKC_S;
            keys[CVirtualGimbalEvent::MoveLeft] = ui::E_KEY_CODE::EKC_A;
            keys[CVirtualGimbalEvent::MoveRight] = ui::E_KEY_CODE::EKC_D;
            keys[CVirtualGimbalEvent::MoveUp] = ui::E_KEY_CODE::EKC_SPACE;
            keys[CVirtualGimbalEvent::MoveDown] = ui::E_KEY_CODE::EKC_LEFT_SHIFT;
        });
    }

    traits_t::gimbal_t m_gimbal;
};

}

#endif // _C_CAMERA_HPP_