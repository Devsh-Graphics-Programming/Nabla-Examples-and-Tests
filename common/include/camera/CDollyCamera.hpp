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

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const double deltaYaw = scaleVirtualRotation(impulse.dVirtualRotation.y);
        const double deltaPitch = scaleVirtualRotation(impulse.dVirtualRotation.x);

        const auto basis = computeBasis(m_u, m_v, m_distance);
        const auto delta =
            basis.right * scaleVirtualTranslation(impulse.dVirtualTranslate.x) +
            basis.up * scaleVirtualTranslation(impulse.dVirtualTranslate.y) +
            basis.forward * scaleVirtualTranslation(impulse.dVirtualTranslate.z);

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
    static inline constexpr double MaxPitch = hlsl::numbers::pi<double> * (85.0 / 180.0);
    static inline constexpr double MinPitch = -MaxPitch;
};

}

#endif
