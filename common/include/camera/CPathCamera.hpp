#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::hlsl
{

class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CPathCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        const auto offset = position - target;
        m_pathRadius = std::sqrt(offset.x * offset.x + offset.z * offset.z);
        if (m_pathRadius < MinPathRadius)
            m_pathRadius = MinPathRadius;
        m_pathHeight = offset.y;
        m_pathAngle = std::atan2(offset.z, offset.x);
        updateFromPath();
    }
    ~CPathCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        constexpr double translateScalar = 0.01;
        const double moveScalar = translateScalar * m_moveSpeedScale;

        m_pathAngle += impulse.dVirtualTranslate.z * moveScalar;
        m_pathRadius = std::max(MinPathRadius, m_pathRadius + impulse.dVirtualTranslate.x * moveScalar);
        m_pathHeight += impulse.dVirtualTranslate.y * moveScalar;

        return updateFromPath();
    }

    virtual const uint32_t getAllowedVirtualEvents() override { return AllowedVirtualEvents; }
    virtual const std::string_view getIdentifier() override { return "Path Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double MinPathRadius = 0.1;

    double m_pathAngle = 0.0;
    double m_pathRadius = 1.0;
    double m_pathHeight = 0.0;

    bool updateFromPath()
    {
        const double x = std::cos(m_pathAngle) * m_pathRadius;
        const double z = std::sin(m_pathAngle) * m_pathRadius;
        const float64_t3 position = m_targetPosition + float64_t3(x, m_pathHeight, z);
        initFromPosition(position);
        return applyPose();
    }

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::MoveRight;
        preset[ui::E_KEY_CODE::EKC_Q] = CVirtualGimbalEvent::MoveDown;
        preset[ui::E_KEY_CODE::EKC_E] = CVirtualGimbalEvent::MoveUp;

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
