// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FPS_CAMERA_HPP_
#define _C_FPS_CAMERA_HPP_

#include <cmath>

#include "ICamera.hpp"

namespace nbl::core
{

class CFPSCamera final : public ICamera
{ 
public:
    using base_t = ICamera;
    static inline constexpr float HalfPi = 1.57079632679489661923f;
    static inline constexpr float RadToDeg = 57.2957795130823208768f;

    CFPSCamera(const float64_t3& position, const camera_quaternion_t<float64_t>& orientation = makeIdentityQuaternion<float64_t>())
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) 
    {
        m_gimbal.begin();
        {
            const auto& gForward = m_gimbal.getZAxis();
            const float gForwardX = static_cast<float>(gForward.x);
            const float gForwardY = static_cast<float>(gForward.y);
            const float gForwardZ = static_cast<float>(gForward.z);
            const float gPitch = std::atan2(std::hypot(gForwardX, gForwardZ), gForwardY) - HalfPi;
            const float gYaw = std::atan2(gForwardX, gForwardZ);
            m_gimbal.setOrientation(makeQuaternionFromEulerRadians(float64_t3(gPitch, gYaw, 0.0f)));
        }
        m_gimbal.end();
    }
	~CFPSCamera() = default;

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        CReferenceTransform reference;
        if (not m_gimbal.extractReferenceTransform(&reference, referenceFrame))
            return false;

        auto validateReference = [&]()
        {
            if (referenceFrame)
            {
                const auto& q = reference.orientation;
                const float w = static_cast<float>(q.data.w);
                const float x = static_cast<float>(q.data.x);
                const float y = static_cast<float>(q.data.y);
                const float z = static_cast<float>(q.data.z);
                const float sinr_cosp = 2.f * (w * z + x * y);
                const float cosr_cosp = 1.f - 2.f * (y * y + z * z);
                const float roll = RadToDeg * std::atan2(sinr_cosp, cosr_cosp);
                const float absRoll = std::abs(roll);
                constexpr float epsilon = 1.e-4f;

                if (not ((absRoll <= epsilon) || (std::abs(absRoll - 180.f) <= epsilon)))
                    return false;
            }

            return true;
        };

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        m_gimbal.begin();
        {
            const float rForwardX = static_cast<float>(reference.frame[2].x);
            const float rForwardY = static_cast<float>(reference.frame[2].y);
            const float rForwardZ = static_cast<float>(reference.frame[2].z);
            const float rPitch = std::atan2(std::hypot(rForwardX, rForwardZ), rForwardY) - HalfPi;
            const float gYaw = std::atan2(rForwardX, rForwardZ);
            const float newPitch = std::clamp<float>(rPitch + impulse.dVirtualRotation.x * getRotationSpeedScale(), MinVerticalAngle, MaxVerticalAngle), newYaw = gYaw + impulse.dVirtualRotation.y * getRotationSpeedScale();

            if (validateReference())
                m_gimbal.setOrientation(makeQuaternionFromEulerRadians(float64_t3(newPitch, newYaw, 0.0f)));
            m_gimbal.setPosition(float64_t3(reference.frame[3]) + rotateVectorByQuaternion(reference.orientation, float64_t3(impulse.dVirtualTranslate)));
        }
        m_gimbal.end();

        manipulated &= bool(m_gimbal.getManipulationCounter());

        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::FPS;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "FPS Camera";
    }

private:

    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr float MaxVerticalAngle = 1.53588974175501f, MinVerticalAngle = -MaxVerticalAngle;
};

}

#endif // _C_FPS_CAMERA_HPP_
