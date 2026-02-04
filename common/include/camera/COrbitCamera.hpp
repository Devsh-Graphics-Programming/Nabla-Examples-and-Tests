#ifndef _C_ORBIT_CAMERA_HPP_
#define _C_ORBIT_CAMERA_HPP_

#include "ICamera.hpp"

namespace nbl::hlsl
{

class COrbitCamera final : public ICamera
{
public:
    using base_t = ICamera;

    COrbitCamera(const float64_t3& position, const float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(length(m_targetPosition - position)), m_gimbal({ .position = position, .orientation = glm::quat(glm::vec3(0, 0, 0)) }) {}
    ~COrbitCamera() = default;

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

    inline void target(const float64_t3& p)
    {
        m_targetPosition = p;
    }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4 const* referenceFrame = nullptr) override
    {
        // TODO: it must work differently, we should take another gimbal to control target

        // position on the sphere
        auto S = [&](double u, double v) -> float64_t3
        {
            return float64_t3
            {
                std::cos(v) * std::cos(u),
                std::cos(v) * std::sin(u),
                std::sin(v)
            } * (double) m_distance;
        };

        /*
        // partial derivative of S with respect to u
        auto Sdu = [&](double u, double v) -> float64_t3
        {
            return float64_t3
            {
                -std::cos(v) * std::sin(u),
                std::cos(v)* std::cos(u),
                0
            } * (double) m_distance;
        };
        */

        // partial derivative of S with respect to v
        auto Sdv = [&](double u, double v) -> float64_t3
        {
            return float64_t3
            {
                -std::sin(v) * std::cos(u),
                -std::sin(v) * std::sin(u),
                std::cos(v)
            } *(double)m_distance;
        };

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        double deltaU = impulse.dVirtualTranslate.y, deltaV = impulse.dVirtualTranslate.x, deltaDistance = impulse.dVirtualTranslate.z;

        // TODO!
        constexpr auto nastyScalar = 0.01;
        deltaU *= nastyScalar * m_moveSpeedScale;
        deltaV *= nastyScalar * m_moveSpeedScale;

        u += deltaU;
        v += deltaV;
   
        m_distance = std::clamp<float>(m_distance += deltaDistance * nastyScalar, MinDistance, MaxDistance);

        const auto localSpherePostion = S(u, v);
        const auto newPosition = localSpherePostion + m_targetPosition;

        // note we are not using Sdu (though we could!)
        // instead we benefit from forward we have for free when moving on sphere surface
        // and given up vector obtained from partial derivative we can easily get right vector, this way
        // we don't have frenet frame flip we would have with Sdu, however it could be adjusted anyway, less code

        const auto newUp = normalize(Sdv(u, v));
        const auto newForward = normalize(-localSpherePostion);
        const auto newRight = normalize(cross(newUp, newForward));

        const auto newOrientation = glm::quat_cast
        (
            glm::dmat3
            {
                newRight,
                newUp,
                newForward
            }
        );
        
        m_gimbal.begin();
        {
            m_gimbal.setPosition(newPosition);
            m_gimbal.setOrientation(newOrientation);
        }
        m_gimbal.end();

        bool manipulated = bool(m_gimbal.getManipulationCounter());

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
        return "Orbit Camera";
    }

    inline float getDistance() { return m_distance; }
    inline double getU() { return u; }
    inline double getV() { return v; }

    static inline constexpr float MinDistance = 0.1f;
    static inline constexpr float MaxDistance = 10000.f;

private:
    float64_t3 m_targetPosition;
    float m_distance;
    typename base_t::CGimbal m_gimbal;

    double u = {}, v = {};

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
