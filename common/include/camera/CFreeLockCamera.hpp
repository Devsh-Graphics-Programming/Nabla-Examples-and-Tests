// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_FREE_CAMERA_HPP_
#define _C_FREE_CAMERA_HPP_

#include "ICamera.hpp"

#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{
    static inline IGimbal<float64_t>::VirtualImpulse sVirtualImpulse = {};
    static inline glm::mat4 sReferenceFrame = glm::mat4(1.0f);
    static inline glm::quat sReferenceOrientation = {};

    // TODO: DEBUG AND TEMPORARY
    void ShowDebugWindow()
    {
        ImGui::Begin("Debug Window");

        ImGui::Text("Translate deltas:");
        ImGui::Text("  x: %.3f", sVirtualImpulse.dVirtualTranslate.x);
        ImGui::Text("  y: %.3f", sVirtualImpulse.dVirtualTranslate.y);
        ImGui::Text("  z: %.3f", sVirtualImpulse.dVirtualTranslate.z);

        ImGui::Separator();

        ImGui::Text("Rotation deltas:");
        ImGui::Text("  x: %.3f", sVirtualImpulse.dVirtualRotation.x);
        ImGui::Text("  y: %.3f", sVirtualImpulse.dVirtualRotation.y);
        ImGui::Text("  z: %.3f", sVirtualImpulse.dVirtualRotation.z);

        ImGui::Separator();

        ImGui::Text("Scale deltas:");
        ImGui::Text("  x: %.3f", sVirtualImpulse.dVirtualScale.x);
        ImGui::Text("  y: %.3f", sVirtualImpulse.dVirtualScale.y);
        ImGui::Text("  z: %.3f", sVirtualImpulse.dVirtualScale.z);

        ImGui::Separator();

        ImGui::Text("Reference frame:");

        for (int row = 0; row < 4; ++row)
        {
            ImGui::Text("%.3f  %.3f  %.3f  %.3f",
                sReferenceFrame[0][row],
                sReferenceFrame[1][row],
                sReferenceFrame[2][row],
                sReferenceFrame[3][row]);
        }

        ImGui::Text("Reference orientation:");

        ImGui::Text("%.3f  %.3f  %.3f  %.3f",
            sReferenceOrientation.x,
            sReferenceOrientation.y,
            sReferenceOrientation.z,
            sReferenceOrientation.w);

        ImGui::End();
    }

// Free Lock Camera
class CFreeCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CFreeCamera(const float64_t3& position, const glm::quat& orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
        : base_t(), m_gimbal({ .position = position, .orientation = orientation }) {}
    ~CFreeCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override
    {
        return m_gimbal;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4 const* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        CReferenceTransform reference;
        if (not m_gimbal.extractReferenceTransform(&reference, referenceFrame))
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        bool manipulated = true;

        // TODO: DEBUG AND TEMPORARY
        {
            sVirtualImpulse = impulse;
            auto cast = getCastedMatrix<float32_t>(reference.frame);;
            memcpy(&sReferenceFrame, &cast, sizeof(sReferenceFrame));
            sReferenceOrientation = reference.orientation;
        }

        m_gimbal.begin();
        {
            glm::quat pitch = glm::angleAxis<float>(impulse.dVirtualRotation.x, glm::vec3(reference.frame[0]));
            glm::quat yaw = glm::angleAxis<float>(impulse.dVirtualRotation.y, glm::vec3(reference.frame[1]));
            glm::quat roll = glm::angleAxis<float>(impulse.dVirtualRotation.z, glm::vec3(reference.frame[2]));

            m_gimbal.setOrientation(yaw * pitch * roll * reference.orientation);
            m_gimbal.setPosition(glm::vec3(reference.frame[3]) + reference.orientation * glm::vec3(impulse.dVirtualTranslate));
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
        return "Free-Look Camera";
    }

private:
    typename base_t::CGimbal m_gimbal;

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;

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
        preset[ui::E_KEY_CODE::EKC_Q] = CVirtualGimbalEvent::RollLeft;
        preset[ui::E_KEY_CODE::EKC_E] = CVirtualGimbalEvent::RollRight;

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
        preset[CVirtualGimbalEvent::RollLeft] = CVirtualGimbalEvent::RollLeft;
        preset[CVirtualGimbalEvent::RollRight] = CVirtualGimbalEvent::RollRight;

        return preset;
    }();
};

}

#endif // _C_FREE_CAMERA_HPP_
