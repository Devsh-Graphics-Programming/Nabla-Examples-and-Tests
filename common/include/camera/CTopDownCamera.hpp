#ifndef _C_TOPDOWN_CAMERA_HPP_
#define _C_TOPDOWN_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::hlsl
{

class CTopDownCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CTopDownCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_v = TopDownPitch;
        applyPose();
    }
    ~CTopDownCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const double deltaYaw = impulse.dVirtualRotation.y * m_rotationSpeedScale;
        constexpr double translateScalar = 0.01;
        const double panScalar = translateScalar * m_moveSpeedScale;
        const double deltaPanX = impulse.dVirtualTranslate.x * panScalar;
        const double deltaPanY = impulse.dVirtualTranslate.y * panScalar;
        const double deltaDistance = impulse.dVirtualTranslate.z * translateScalar;

        m_u += deltaYaw;
        m_v = TopDownPitch;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        const auto basis = computeBasis(m_u, m_v, m_distance);
        if (deltaPanX != 0.0 || deltaPanY != 0.0)
            m_targetPosition += basis.right * deltaPanX + basis.up * deltaPanY;

        return applyPose();
    }

    virtual const uint32_t getAllowedVirtualEvents() override { return AllowedVirtualEvents; }
    virtual const std::string_view getIdentifier() override { return "Top-Down Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double TopDownPitch = -1.5707963267948966;

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::MoveRight;
        preset[ui::E_KEY_CODE::EKC_Q] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_KEY_CODE::EKC_E] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_J] = CVirtualGimbalEvent::PanLeft;
        preset[ui::E_KEY_CODE::EKC_L] = CVirtualGimbalEvent::PanRight;

        return preset;
    }();

    static inline const auto m_mouse_to_virtual_events_preset = []()
    {
        typename base_t::mouse_to_virtual_events_t preset;

        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X] = CVirtualGimbalEvent::PanRight;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X] = CVirtualGimbalEvent::PanLeft;
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
        preset[CVirtualGimbalEvent::PanLeft] = CVirtualGimbalEvent::PanLeft;
        preset[CVirtualGimbalEvent::PanRight] = CVirtualGimbalEvent::PanRight;

        return preset;
    }();
};

}

#endif
