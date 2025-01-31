// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FPS_CAMERA_HPP_
#define _C_FPS_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

// FPS Camera
class CFPSCamera final : public ICamera
{ 
public:
    using base_t = ICamera;

    CFPSCamera(const float64_t3& position, const glm::quat& orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) 
    {
        m_gimbal.begin();
        {
            const auto& gForward = m_gimbal.getZAxis();
            const float gPitch = atan2(glm::length(glm::vec2(gForward.x, gForward.z)), gForward.y) - glm::half_pi<float>(), gYaw = atan2(gForward.x, gForward.z);




            auto test = glm::quat(glm::vec3(gPitch, gYaw, 0.0f));

            glm::vec3 euler = glm::eulerAngles(test);

            m_gimbal.setOrientation(test);
        }
        m_gimbal.end();
    }
	~CFPSCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4 const* referenceFrame = nullptr) override
    {
        if (!virtualEvents.size())
            return false;

        //! true if virtual events are with respect to the moving gimbal frame, otherwise it is assumed deltas are generated for fixed reference frame
        const bool isMovingReference = not referenceFrame;

        CReferenceTransform reference;
        if (not m_gimbal.extractReferenceTransform(&reference, referenceFrame))
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        m_gimbal.begin();
        {
            if (isMovingReference)
            {
                const auto& gForward = m_gimbal.getZAxis();
                const float gPitch = atan2(glm::length(glm::vec2(gForward.x, gForward.z)), gForward.y) - glm::half_pi<float>(), gYaw = atan2(gForward.x, gForward.z);
                const float newPitch = std::clamp<float>(gPitch + impulse.dVirtualRotation.x * m_rotationSpeedScale, MinVerticalAngle, MaxVerticalAngle), newYaw = gYaw + impulse.dVirtualRotation.y * m_rotationSpeedScale;
                
                m_gimbal.setOrientation(glm::quat(glm::vec3(newPitch, newYaw, 0.0f)));
                m_gimbal.setPosition(m_gimbal.getPosition() + mul(impulse.dVirtualTranslate * m_moveSpeedScale, m_gimbal.getOrthonornalMatrix()));
            }
            else
            {
                // TODO: I need to think of this, the problem is that since reference frame may not be aligned with world nicely it means events must be transformed
                // what FPS camera really does is:
                // tilt around FIXED world (0,1,0) vector
                // pitch around REFERENCE right vector
                m_gimbal.transform(reference, impulse);                
            }
        }
        m_gimbal.end();

        manipulated &= bool(m_gimbal.getManipulationCounter());

        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    virtual const uint32_t getAllowedVirtualEvents() override
    {
        return AllowedVirtualEvents;
    }

    virtual const std::string_view getIdentifier() override
    {
        return "FPS Camera";
    }

private:

    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr float MaxVerticalAngle = glm::radians(88.0f), MinVerticalAngle = -MaxVerticalAngle;

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

#endif // _C_FPS_CAMERA_HPP_