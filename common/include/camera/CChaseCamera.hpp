#ifndef _C_CHASE_CAMERA_HPP_
#define _C_CHASE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CChaseCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CChaseCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_orbitUv.y = std::clamp(m_orbitUv.y, MinPitch, MaxPitch);
        applyPose();
    }
    ~CChaseCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const auto deltaRotation = scaleVirtualRotation(impulse.dVirtualRotation);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const auto deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.y);

        const auto basis = computeBasis(m_orbitUv, m_distance);

        const auto planarForward = hlsl::safeNormalizeVec3(
            hlsl::float64_t3(basis.forward.x, 0.0, basis.forward.z),
            hlsl::float64_t3(0.0, 0.0, 1.0));
        const auto planarRight = hlsl::safeNormalizeVec3(
            hlsl::float64_t3(basis.right.x, 0.0, basis.right.z),
            hlsl::float64_t3(1.0, 0.0, 0.0));

        m_targetPosition += planarRight * deltaTranslation.x + planarForward * deltaTranslation.z;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        m_orbitUv.x += deltaRotation.y;
        m_orbitUv.y = std::clamp(m_orbitUv.y + deltaRotation.x, MinPitch, MaxPitch);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Chase; }
    virtual std::string_view getIdentifier() const override { return "Chase Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = SCameraTargetRelativeRigDefaults::ChaseMaxPitchRad;
    static inline constexpr double MinPitch = SCameraTargetRelativeRigDefaults::ChaseMinPitchRad;
};

}

#endif
