// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::core
{
// Free Lock Camera
class CFreeCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFreeCamera(const float64_t3& position, const camera_quaternion_t<float64_t>& orientation = makeIdentityQuaternion<float64_t>())
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) {}
    ~CFreeCamera() = default;

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

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        m_gimbal.begin();
        {
            const auto pitch = makeQuaternionFromAxisAngle(normalize(float64_t3(reference.frame[0])), impulse.dVirtualRotation.x);
            const auto yaw = makeQuaternionFromAxisAngle(normalize(float64_t3(reference.frame[1])), impulse.dVirtualRotation.y);
            const auto roll = makeQuaternionFromAxisAngle(normalize(float64_t3(reference.frame[2])), impulse.dVirtualRotation.z);

            m_gimbal.setOrientation(normalizeQuaternion(yaw * pitch * roll * reference.orientation));
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
        return CameraKind::Free;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Free-Look Camera";
    }

private:
    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
};

}

#endif // _C_FREE_CAMERA_HPP_
