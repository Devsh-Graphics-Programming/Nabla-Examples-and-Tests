#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include <algorithm>
#include <cmath>
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class COrbitCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    COrbitCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_distance = std::clamp<float>(length(m_targetPosition - position), MinDistance, MaxDistance);
        applyPose();
    }
    ~COrbitCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        double deltaU = impulse.dVirtualTranslate.y, deltaV = impulse.dVirtualTranslate.x, deltaDistance = impulse.dVirtualTranslate.z;

        constexpr auto orbitMotionScalar = 0.01;
        deltaU *= orbitMotionScalar * getMoveSpeedScale();
        deltaV *= orbitMotionScalar * getMoveSpeedScale();

        m_u += deltaU;
        m_v += deltaV;
   
        m_distance = std::clamp<float>(m_distance += deltaDistance * orbitMotionScalar, MinDistance, MaxDistance);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Orbit;
    }

    virtual std::string_view getIdentifier() const override
    {
        return "Orbit Camera";
    }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
};

}

#endif // _C_ORBIT_CAMERA_HPP_
