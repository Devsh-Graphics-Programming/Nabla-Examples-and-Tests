#ifndef _C_DOLLY_ZOOM_CAMERA_HPP_
#define _C_DOLLY_ZOOM_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CDollyZoomCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyZoomCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, float baseFov = DefaultBaseFovDeg)
        : base_t(position, target), m_baseFov(baseFov), m_referenceDistance(m_distance)
    {
        applyPose();
    }
    ~CDollyZoomCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    float getBaseFov() const { return m_baseFov; }
    void setBaseFov(float fov) { m_baseFov = fov; }

    float getReferenceDistance() const { return m_referenceDistance; }
    void setReferenceDistance(float distance) { m_referenceDistance = distance; }

    float computeDollyFov() const
    {
        const double base = std::tan(hlsl::radians(static_cast<double>(m_baseFov)) * 0.5);
        const double ratio = static_cast<double>(m_referenceDistance) / std::max<double>(static_cast<double>(m_distance), static_cast<double>(MinDistance));
        const double fov = 2.0 * std::atan(base * ratio);
        const double fovDeg = hlsl::degrees(fov);
        return static_cast<float>(std::clamp(fovDeg, static_cast<double>(MinDynamicFovDeg), static_cast<double>(MaxDynamicFovDeg)));
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const auto deltaTranslation = scaleVirtualTranslation(impulse.dVirtualTranslate);
        const double deltaDistance = scaleUnscaledVirtualTranslation(impulse.dVirtualTranslate.z);

        m_orbitUv += hlsl::float64_t2(deltaTranslation.y, deltaTranslation.x);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::DollyZoom; }
    virtual uint32_t getCapabilities() const override { return base_t::getCapabilities() | base_t::DynamicPerspectiveFov; }
    virtual bool tryGetDynamicPerspectiveFov(float& outFov) const override
    {
        outFov = computeDollyFov();
        return true;
    }
    virtual bool tryGetDynamicPerspectiveState(DynamicPerspectiveState& out) const override
    {
        out.baseFov = m_baseFov;
        out.referenceDistance = m_referenceDistance;
        return true;
    }
    virtual bool trySetDynamicPerspectiveState(const DynamicPerspectiveState& state) override
    {
        if (!hlsl::isFiniteScalar(state.baseFov) || !hlsl::isFiniteScalar(state.referenceDistance) || state.referenceDistance <= 0.f)
            return false;

        m_baseFov = state.baseFov;
        m_referenceDistance = state.referenceDistance;
        return true;
    }
    virtual std::string_view getIdentifier() const override { return "Dolly Zoom Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr float DefaultBaseFovDeg = 40.0f;
    static inline constexpr float MinDynamicFovDeg = 10.0f;
    static inline constexpr float MaxDynamicFovDeg = 150.0f;

    float m_baseFov = DefaultBaseFovDeg;
    float m_referenceDistance = 1.0f;
};

}

#endif
