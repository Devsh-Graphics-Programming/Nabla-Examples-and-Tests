#ifndef _C_DOLLY_CAMERA_HPP_
#define _C_DOLLY_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CDollyCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_v = std::clamp(m_v, MinPitch, MaxPitch);
        applyPose();
    }
    ~CDollyCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const double deltaYaw = impulse.dVirtualRotation.y * getRotationSpeedScale();
        const double deltaPitch = impulse.dVirtualRotation.x * getRotationSpeedScale();

        constexpr double translateScalar = 0.01;
        const double moveScalar = translateScalar * getMoveSpeedScale();

        const auto basis = computeBasis(m_u, m_v, m_distance);
        const auto delta = (basis.right * impulse.dVirtualTranslate.x + basis.up * impulse.dVirtualTranslate.y + basis.forward * impulse.dVirtualTranslate.z) * moveScalar;

        m_targetPosition += delta;
        m_u += deltaYaw;
        m_v = std::clamp(m_v + deltaPitch, MinPitch, MaxPitch);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Dolly; }
    virtual std::string_view getIdentifier() const override { return "Dolly Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = 1.4835298641951802;
    static inline constexpr double MinPitch = -1.4835298641951802;
};

}

#endif
