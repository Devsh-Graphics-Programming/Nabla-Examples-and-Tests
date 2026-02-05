// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _C_TURNTABLE_CAMERA_HPP_
#define _C_TURNTABLE_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "ICamera.hpp"

namespace nbl::hlsl
{

class CTurntableCamera final : public ICamera
{
public:
    using base_t = ICamera;

    CTurntableCamera(const float64_t3& position, const float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(1.0f), m_gimbal({ .position = position, .orientation = glm::quat(glm::vec3(0.0f)) })
    {
        initFromPosition(position);
        applyPose();
    }
    ~CTurntableCamera() = default;

    const base_t::keyboard_to_virtual_events_t getKeyboardMappingPreset() const override { return m_keyboard_to_virtual_events_preset; }
    const base_t::mouse_to_virtual_events_t getMouseMappingPreset() const override { return m_mouse_to_virtual_events_preset; }
    const base_t::imguizmo_to_virtual_events_t getImguizmoMappingPreset() const override { return m_imguizmo_to_virtual_events_preset; }

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    inline bool setDistance(float d)
    {
        const auto clamped = std::clamp<float>(d, MinDistance, MaxDistance);
        const bool ok = clamped == d;
        m_distance = clamped;
        return ok;
    }

    inline void target(const float64_t3& p) { m_targetPosition = p; }
    inline float64_t3 getTarget() const { return m_targetPosition; }

    inline float getDistance() { return m_distance; }
    inline double getU() { return u; }
    inline double getV() { return v; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4 const* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        const double deltaYaw = impulse.dVirtualRotation.y * m_rotationSpeedScale;
        const double deltaPitch = impulse.dVirtualRotation.x * m_rotationSpeedScale;

        constexpr double translateScalar = 0.01;
        const double deltaDistance = impulse.dVirtualTranslate.z * translateScalar;

        u += deltaYaw;
        v = std::clamp(v + deltaPitch, MinPitch, MaxPitch);
        m_distance = std::clamp<float>(m_distance + static_cast<float>(deltaDistance), MinDistance, MaxDistance);

        return applyPose();
    }

    virtual const uint32_t getAllowedVirtualEvents() override { return AllowedVirtualEvents; }
    virtual const std::string_view getIdentifier() override { return "Turntable Camera"; }

    static inline constexpr float MinDistance = 0.1f;
    static inline constexpr float MaxDistance = 10000.f;

private:
    float64_t3 m_targetPosition;
    float m_distance;
    typename base_t::CGimbal m_gimbal;
    double u = {};
    double v = {};

    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::Rotate;
    static inline constexpr double MaxPitch = glm::radians(89.0);
    static inline constexpr double MinPitch = -MaxPitch;

    inline float64_t3 S(double su, double sv) const
    {
        return float64_t3
        {
            std::cos(sv) * std::cos(su),
            std::cos(sv) * std::sin(su),
            std::sin(sv)
        };
    }

    inline float64_t3 Sdv(double su, double sv) const
    {
        return float64_t3
        {
            -std::sin(sv) * std::cos(su),
            -std::sin(sv) * std::sin(su),
            std::cos(sv)
        };
    }

    inline void initFromPosition(const float64_t3& position)
    {
        const auto offset = position - m_targetPosition;
        const double dist = length(offset);
        const double safeDist = std::isfinite(dist) && dist > 0.0 ? dist : static_cast<double>(MinDistance);
        m_distance = std::clamp<float>(static_cast<float>(safeDist), MinDistance, MaxDistance);
        const auto local = offset / static_cast<double>(m_distance);
        u = std::atan2(local.y, local.x);
        v = std::asin(std::clamp(local.z, -1.0, 1.0));
        v = std::clamp(v, MinPitch, MaxPitch);
    }

    inline bool applyPose()
    {
        const auto localSpherePosition = S(u, v) * static_cast<double>(m_distance);
        const auto newPosition = localSpherePosition + m_targetPosition;
        const auto newForward = normalize(-localSpherePosition);
        const auto newUp = normalize(Sdv(u, v));
        const auto newRight = normalize(cross(newUp, newForward));
        const auto newOrientation = glm::quat_cast(glm::dmat3{ newRight, newUp, newForward });

        m_gimbal.begin();
        {
            m_gimbal.setPosition(newPosition);
            m_gimbal.setOrientation(newOrientation);
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());
        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    static inline const auto m_keyboard_to_virtual_events_preset = []()
    {
        typename base_t::keyboard_to_virtual_events_t preset;

        preset[ui::E_KEY_CODE::EKC_W] = CVirtualGimbalEvent::MoveForward;
        preset[ui::E_KEY_CODE::EKC_S] = CVirtualGimbalEvent::MoveBackward;
        preset[ui::E_KEY_CODE::EKC_A] = CVirtualGimbalEvent::PanLeft;
        preset[ui::E_KEY_CODE::EKC_D] = CVirtualGimbalEvent::PanRight;
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
        preset[CVirtualGimbalEvent::TiltDown] = CVirtualGimbalEvent::TiltDown;
        preset[CVirtualGimbalEvent::TiltUp] = CVirtualGimbalEvent::TiltUp;
        preset[CVirtualGimbalEvent::PanLeft] = CVirtualGimbalEvent::PanLeft;
        preset[CVirtualGimbalEvent::PanRight] = CVirtualGimbalEvent::PanRight;

        return preset;
    }();
};

}

#endif
