// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_TURNTABLE_CAMERA_HPP_
#define _C_TURNTABLE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CTurntableCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CTurntableCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_v = std::clamp(m_v, MinPitch, MaxPitch);
        applyPose();
    }
    ~CTurntableCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const double deltaYaw = impulse.dVirtualRotation.y * getRotationSpeedScale();
        const double deltaPitch = impulse.dVirtualRotation.x * getRotationSpeedScale();

        constexpr double translateScalar = 0.01;
        const double deltaDistance = impulse.dVirtualTranslate.z * translateScalar;

        m_u += deltaYaw;
        m_v = std::clamp(m_v + deltaPitch, MinPitch, MaxPitch);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Turntable; }
    virtual std::string_view getIdentifier() const override { return "Turntable Camera"; }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

private:

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = 1.5533430342749532;
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
