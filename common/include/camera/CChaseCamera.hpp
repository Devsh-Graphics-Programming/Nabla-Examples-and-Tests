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
        m_v = std::clamp(m_v, MinPitch, MaxPitch);
        applyPose();
    }
    ~CChaseCamera() = default;

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

        hlsl::float64_t3 planarForward = hlsl::float64_t3(basis.forward.x, 0.0, basis.forward.z);
        hlsl::float64_t3 planarRight = hlsl::float64_t3(basis.right.x, 0.0, basis.right.z);

        const double forwardLen = hlsl::length(planarForward);
        if (forwardLen > 0.0)
            planarForward /= forwardLen;
        else
            planarForward = hlsl::float64_t3(0.0, 0.0, 1.0);

        const double rightLen = hlsl::length(planarRight);
        if (rightLen > 0.0)
            planarRight /= rightLen;
        else
            planarRight = hlsl::float64_t3(1.0, 0.0, 0.0);

        m_targetPosition += (planarRight * impulse.dVirtualTranslate.x + planarForward * impulse.dVirtualTranslate.z) * moveScalar;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(impulse.dVirtualTranslate.y * translateScalar), MinDistance, MaxDistance);

        m_u += deltaYaw;
        m_v = std::clamp(m_v + deltaPitch, MinPitch, MaxPitch);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Chase; }
    virtual std::string_view getIdentifier() const override { return "Chase Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = 1.2217304763960306;
    static inline constexpr double MinPitch = -1.0471975511965976;
};

}

#endif
