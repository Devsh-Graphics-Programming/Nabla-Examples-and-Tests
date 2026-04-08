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

    CFPSCamera(const hlsl::float64_t3& position, const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>())
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) 
    {
        m_gimbal.begin();
        {
            const auto pitchYaw = hlsl::getPitchYawFromForwardVector(m_gimbal.getZAxis());
            m_gimbal.setOrientation(hlsl::makeQuaternionFromEulerRadians(hlsl::float64_t3(pitchYaw.x, pitchYaw.y, 0.0)));
        }
        m_gimbal.end();
    }
	~CFPSCamera() = default;

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
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
                const float roll = static_cast<float>(hlsl::degrees(hlsl::getQuaternionEulerRadiansYXZ(reference.orientation).z));
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
            const auto pitchYaw = hlsl::getPitchYawFromForwardVector(hlsl::float64_t3(reference.frame[2]));
            const float newPitch = std::clamp<float>(static_cast<float>(pitchYaw.x + scaleVirtualRotation(impulse.dVirtualRotation.x)), MinVerticalAngle, MaxVerticalAngle);
            const float newYaw = static_cast<float>(pitchYaw.y + scaleVirtualRotation(impulse.dVirtualRotation.y));

            if (validateReference())
                m_gimbal.setOrientation(hlsl::makeQuaternionFromEulerRadians(hlsl::float64_t3(newPitch, newYaw, 0.0f)));
            m_gimbal.setPosition(hlsl::float64_t3(reference.frame[3]) + hlsl::rotateVectorByQuaternion(reference.orientation, hlsl::float64_t3(impulse.dVirtualTranslate)));
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
    static inline constexpr float MaxVerticalAngle = static_cast<float>(hlsl::numbers::pi<double> * (88.0 / 180.0));
    static inline constexpr float MinVerticalAngle = -MaxVerticalAngle;
};

}

#endif // _C_FPS_CAMERA_HPP_
