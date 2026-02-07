#ifndef _C_CHASE_CAMERA_HPP_
#define _C_CHASE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::hlsl
{

class CChaseCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CChaseCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_v = std::clamp(m_v, MinPitch, MaxPitch);
        applyPose();
    }
    ~CChaseCamera() = default;

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
        const double deltaPitch = impulse.dVirtualRotation.x * m_rotationSpeedScale;

        constexpr double translateScalar = 0.01;
        const double moveScalar = translateScalar * m_moveSpeedScale;

        const auto basis = computeBasis(m_u, m_v, m_distance);

        float64_t3 planarForward = float64_t3(basis.forward.x, 0.0, basis.forward.z);
        float64_t3 planarRight = float64_t3(basis.right.x, 0.0, basis.right.z);

        const double forwardLen = length(planarForward);
        if (forwardLen > 0.0)
            planarForward /= forwardLen;
        else
            planarForward = float64_t3(0.0, 0.0, 1.0);

        const double rightLen = length(planarRight);
        if (rightLen > 0.0)
            planarRight /= rightLen;
        else
            planarRight = float64_t3(1.0, 0.0, 0.0);

        m_targetPosition += (planarRight * impulse.dVirtualTranslate.x + planarForward * impulse.dVirtualTranslate.z) * moveScalar;
        m_distance = std::clamp<float>(m_distance + static_cast<float>(impulse.dVirtualTranslate.y * translateScalar), MinDistance, MaxDistance);

        m_u += deltaYaw;
        m_v = std::clamp(m_v + deltaPitch, MinPitch, MaxPitch);

        return applyPose();
    }

    virtual const uint32_t getAllowedVirtualEvents() override { return AllowedVirtualEvents; }
    virtual const std::string_view getIdentifier() override { return "Chase Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = glm::radians(70.0);
    static inline constexpr double MinPitch = glm::radians(-60.0);

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::MoveRight;
        preset[ui::E_KEY_CODE::EKC_Q] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_KEY_CODE::EKC_E] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_KEY_CODE::EKC_I] = CVirtualGimbalEvent::TiltDown;
        preset[ui::E_KEY_CODE::EKC_K] = CVirtualGimbalEvent::TiltUp;
        preset[ui::E_KEY_CODE::EKC_J] = CVirtualGimbalEvent::PanLeft;
        preset[ui::E_KEY_CODE::EKC_L] = CVirtualGimbalEvent::PanRight;

        return preset;
    }();

    static inline const auto m_mouse_to_virtual_events_preset = []()
    {
        typename base_t::mouse_to_virtual_events_t preset;

        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_X] = CVirtualGimbalEvent::PanRight;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_X] = CVirtualGimbalEvent::PanLeft;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_POSITIVE_MOVEMENT_Y] = CVirtualGimbalEvent::TiltUp;
        preset[ui::E_MOUSE_CODE::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y] = CVirtualGimbalEvent::TiltDown;
        preset[ui::E_MOUSE_CODE::EMC_VERTICAL_POSITIVE_SCROLL] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_MOUSE_CODE::EMC_HORIZONTAL_POSITIVE_SCROLL] = CVirtualGimbalEvent::MoveUp;
        preset[ui::E_MOUSE_CODE::EMC_VERTICAL_NEGATIVE_SCROLL] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_MOUSE_CODE::EMC_HORIZONTAL_NEGATIVE_SCROLL] = CVirtualGimbalEvent::MoveDown;

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
        preset[CVirtualGimbalEvent::TiltDown] = CVirtualGimbalEvent::TiltDown;
        preset[CVirtualGimbalEvent::TiltUp] = CVirtualGimbalEvent::TiltUp;
        preset[CVirtualGimbalEvent::PanLeft] = CVirtualGimbalEvent::PanLeft;
        preset[CVirtualGimbalEvent::PanRight] = CVirtualGimbalEvent::PanRight;

        return preset;
    }();
};

}

#endif
