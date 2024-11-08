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

    CFPSCamera(const vector<typename base_t::precision_t, 3u>& position, glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) {}
	~CFPSCamera() = default;

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual void manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) override
    {
        constexpr float MoveSpeedScale = 0.01f, RotateSpeedScale = 0.003f, MaxVerticalAngle = glm::radians(88.0f), MinVerticalAngle = -MaxVerticalAngle;

        if (!virtualEvents.size())
            return;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto& gForward = m_gimbal.getZAxis();
        const float currentPitch = atan2(glm::length(glm::vec2(gForward.x, gForward.z)), gForward.y) - glm::half_pi<float>(), currentYaw = atan2(gForward.x, gForward.z);
        const auto newPitch = std::clamp(currentPitch + impulse.dVirtualRotation.x * RotateSpeedScale, MinVerticalAngle, MaxVerticalAngle), newYaw = currentYaw + impulse.dVirtualRotation.y * RotateSpeedScale;

        m_gimbal.begin();
        {
            m_gimbal.setOrientation(glm::quat(glm::vec3(newPitch, newYaw, 0.0f)));
            m_gimbal.move(impulse.dVirtualTranslate * MoveSpeedScale);
            m_gimbal.updateView();
        }
        m_gimbal.end();
    }

    virtual const uint32_t getAllowedVirtualEvents() override
    {
        return AllowedVirtualEvents;
    }

private:
    typename base_t::CGimbal m_gimbal;
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::MoveForward | CVirtualGimbalEvent::MoveBackward | CVirtualGimbalEvent::MoveRight | CVirtualGimbalEvent::MoveLeft | CVirtualGimbalEvent::TiltUp | CVirtualGimbalEvent::TiltDown | CVirtualGimbalEvent::PanRight | CVirtualGimbalEvent::PanLeft;
};

}

#endif // _C_CAMERA_HPP_