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

    CIsometricCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_u = IsoYaw;
        m_v = IsoPitch;
        applyPose();
    }
    ~CIsometricCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        constexpr double translateScalar = 0.01;
        const double panScalar = translateScalar * getMoveSpeedScale();
        const double deltaPanX = impulse.dVirtualTranslate.x * panScalar;
        const double deltaPanY = impulse.dVirtualTranslate.y * panScalar;
        const double deltaDistance = impulse.dVirtualTranslate.z * translateScalar;

        m_u = IsoYaw;
        m_v = IsoPitch;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        const auto basis = computeBasis(m_u, m_v, m_distance);
        if (deltaPanX != 0.0 || deltaPanY != 0.0)
            m_targetPosition += basis.right * deltaPanX + basis.up * deltaPanY;

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Isometric; }
    virtual std::string_view getIdentifier() const override { return "Isometric Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double IsoYaw = 0.7853981633974483;
    static inline constexpr double IsoPitch = 0.6154797086703873;
};

}

#endif
