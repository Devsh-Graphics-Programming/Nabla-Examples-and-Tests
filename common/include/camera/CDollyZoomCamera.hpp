#ifndef _C_DOLLY_ZOOM_CAMERA_HPP_
#define _C_DOLLY_ZOOM_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::hlsl
{

class CDollyZoomCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CDollyZoomCamera(const float64_t3& position, const float64_t3& target, float baseFov = 40.0f)
        : base_t(position, target), m_baseFov(baseFov), m_referenceDistance(m_distance)
    {
        applyPose();
    }
    ~CDollyZoomCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

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
        return static_cast<float>(std::clamp(fovDeg, 10.0, 150.0));
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        double deltaU = impulse.dVirtualTranslate.y;
        double deltaV = impulse.dVirtualTranslate.x;
        double deltaDistance = impulse.dVirtualTranslate.z;

        constexpr auto scalar = 0.01;
        deltaU *= scalar * m_moveSpeedScale;
        deltaV *= scalar * m_moveSpeedScale;

        m_u += deltaU;
        m_v += deltaV;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance * scalar), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual const uint32_t getAllowedVirtualEvents() override { return AllowedVirtualEvents; }
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
        if (!std::isfinite(state.baseFov) || !std::isfinite(state.referenceDistance) || state.referenceDistance <= 0.f)
            return false;

        m_baseFov = state.baseFov;
        m_referenceDistance = state.referenceDistance;
        return true;
    }
    virtual const std::string_view getIdentifier() override { return "Dolly Zoom Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;

    float m_baseFov = 40.0f;
    float m_referenceDistance = 1.0f;

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::MoveRight;
        preset[ui::E_KEY_CODE::EKC_E] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_Q] = CVirtualGimbalEvent::MoveBackward;

        return preset;
    }();

    static inline const auto m_mouse_to_virtual_events_preset = []()
    {
        typename base_t::mouse_to_virtual_events_t preset;

        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X] = CVirtualGimbalEvent::MoveRight;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL] = CVirtualGimbalEvent::MoveBackward;

        return preset;
    }();

    static inline const auto m_imguizmo_to_virtual_events_preset = []()
    {
        typename base_t::imguizmo_to_virtual_events_t preset;

        preset[CVirtualGimbalEvent::MoveForward] = CVirtualGimbalEvent::MoveForward;
        preset[CVirtualGimbalEvent::MoveBackward] = CVirtualGimbalEvent::MoveBackward;
        preset[CVirtualGimbalEvent::MoveLeft] = CVirtualGimbalEvent::MoveLeft;
        preset[CVirtualGimbalEvent::MoveRight] = CVirtualGimbalEvent::MoveRight;
        preset[CVirtualGimbalEvent::MoveUp] = CVirtualGimbalEvent::MoveUp;
        preset[CVirtualGimbalEvent::MoveDown] = CVirtualGimbalEvent::MoveDown;

        return preset;
    }();
};

}

#endif
