#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include <algorithm>
#include <cmath>
#include "CSphericalTargetCamera.hpp"

namespace nbl::hlsl
{

class COrbitCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    COrbitCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        m_distance = std::clamp<float>(length(m_targetPosition - position), MinDistance, MaxDistance);
        applyPose();
    }
    ~COrbitCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        // TODO: it must work differently, we should take another gimbal to control target

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        double deltaU = impulse.dVirtualTranslate.y, deltaV = impulse.dVirtualTranslate.x, deltaDistance = impulse.dVirtualTranslate.z;

        // TODO!
        constexpr auto nastyScalar = 0.01;
        deltaU *= nastyScalar * m_moveSpeedScale;
        deltaV *= nastyScalar * m_moveSpeedScale;

        m_u += deltaU;
        m_v += deltaV;
   
        m_distance = std::clamp<float>(m_distance += deltaDistance * nastyScalar, MinDistance, MaxDistance);

        return applyPose();
    }

    virtual const uint32_t getAllowedVirtualEvents() override
    {
        return AllowedVirtualEvents;
    }

    virtual CameraKind getKind() const override
    {
        return CameraKind::Orbit;
    }

    virtual const std::string_view getIdentifier() override
    {
        return "Orbit Camera";
    }

    static inline constexpr float MinDistance = base_t::MinDistance;
    static inline constexpr float MaxDistance = base_t::MaxDistance;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;

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

#endif // _C_ORBIT_CAMERA_HPP_
