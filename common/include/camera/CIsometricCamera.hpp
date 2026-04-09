#ifndef _C_ISOMETRIC_CAMERA_HPP_
#define _C_ISOMETRIC_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CIsometricCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CIsometricCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv = hlsl::float64_t2(IsoYaw, IsoPitch);
        applyPose();
    }
    ~CIsometricCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv = hlsl::float64_t2(IsoYaw, IsoPitch);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        const auto basis = computeBasis(m_orbitUv, m_distance);
        applyPlanarTargetTranslation(deltaTranslation, basis);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Isometric; }
    virtual std::string_view getIdentifier() const override { return "Isometric Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double IsoYaw = SCameraTargetRelativeRigDefaults::IsometricYawRad;
    static inline constexpr double IsoPitch = SCameraTargetRelativeRigDefaults::IsometricPitchRad;
};

}

#endif
