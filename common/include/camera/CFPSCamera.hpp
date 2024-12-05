// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_CAMERA_HPP_
#define _C_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

// FPS Camera
template<typename T = float64_t>
class CFPSCamera final : public ICamera<T>
{ 
public:
    using base_t = ICamera<T>;

    CFPSCamera(const vector<typename base_t::precision_t, 3u>& position, const glm::quat& orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) {}
	~CFPSCamera() = default;

    const base_t::keyboard_to_virtual_events_t& getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t& getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t& getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, base_t::ManipulationMode mode) override
    {
        constexpr float MoveSpeedScale = 0.01f, RotateSpeedScale = 0.003f, MaxVerticalAngle = glm::radians(88.0f), MinVerticalAngle = -MaxVerticalAngle;

        if (!virtualEvents.size())
            return false;

        const auto& gForward = m_gimbal.getZAxis();
        const float gPitch = atan2(glm::length(glm::vec2(gForward.x, gForward.z)), gForward.y) - glm::half_pi<float>(), gYaw = atan2(gForward.x, gForward.z);

        glm::quat newOrientation; vector<typename base_t::precision_t, 3u> newPosition;

        // TODO: I make assumption what world base is now (at least temporary), I need to think of this but maybe each ITransformObject should know what its world is
        // in ideal scenario we would define this crazy enum with all possible standard bases
        const auto impulse = mode == base_t::Local ? m_gimbal.accumulate<AllowedLocalVirtualEvents>(virtualEvents)
            : m_gimbal.accumulate<AllowedWorldVirtualEvents>(virtualEvents, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 });

        const auto newPitch = std::clamp(gPitch + impulse.dVirtualRotation.x * RotateSpeedScale, MinVerticalAngle, MaxVerticalAngle), newYaw = gYaw + impulse.dVirtualRotation.y * RotateSpeedScale;
        newOrientation = glm::quat(glm::vec3(newPitch, newYaw, 0.0f)); newPosition = m_gimbal.getPosition() + impulse.dVirtualTranslate * MoveSpeedScale;

        bool manipulated = true;

        m_gimbal.begin();
        {
            m_gimbal.setOrientation(newOrientation);
            m_gimbal.setPosition(newPosition);
        }
        m_gimbal.end();

        manipulated &= bool(m_gimbal.getManipulationCounter());

        if(manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    virtual const uint32_t getAllowedVirtualEvents(base_t::ManipulationMode mode) override
    {
        switch (mode)
        {
            case base_t::Local: return AllowedLocalVirtualEvents;
            case base_t::World: return AllowedWorldVirtualEvents;
            default: return CVirtualGimbalEvent::None;
        }
    }

    virtual const std::string_view getIdentifier() override
    {
        return "FPS Camera";
    }

private:
    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedLocalVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::TiltUp | CVirtualGimbalEvent::TiltDown | CVirtualGimbalEvent::PanRight | CVirtualGimbalEvent::PanLeft;
    static inline constexpr auto AllowedWorldVirtualEvents = AllowedLocalVirtualEvents;

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::MoveLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::MoveRight;
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

#endif // _C_CAMERA_HPP_