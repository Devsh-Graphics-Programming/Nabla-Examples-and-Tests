// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <optional>
#include <utility>

#include "camera/IGimbalBindingLayout.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

class ICamera : virtual public core::IReferenceCounted
{ 
public:
    using binding_layout_t = IGimbalBindingLayout;
    using gimbal_event_t = binding_layout_t::gimbal_event_t;
    using encode_keyboard_code_t = binding_layout_t::encode_keyboard_code_t;
    using encode_mouse_code_t = binding_layout_t::encode_mouse_code_t;
    using encode_imguizmo_code_t = binding_layout_t::encode_imguizmo_code_t;
    using BindingDomain = binding_layout_t::BindingDomain;
    using CKeyInfo = binding_layout_t::CKeyInfo;
    using CHashInfo = binding_layout_t::CHashInfo;
    using keyboard_to_virtual_events_t = binding_layout_t::keyboard_to_virtual_events_t;
    using mouse_to_virtual_events_t = binding_layout_t::mouse_to_virtual_events_t;
    using imguizmo_to_virtual_events_t = binding_layout_t::imguizmo_to_virtual_events_t;

    struct SMotionConfig
    {
        double moveSpeedScale = 0.01;
        double rotationSpeedScale = 0.003;
    };

    struct SInputBindingConfig
    {
        CGimbalBindingLayoutStorage defaultBindingLayout;
    };

    enum class CameraKind : uint8_t
    {
        Unknown,
        FPS,
        Free,
        Orbit,
        Arcball,
        Turntable,
        TopDown,
        Isometric,
        Chase,
        Dolly,
        DollyZoom,
        Path
    };

    enum CameraCapability : uint32_t
    {
        None = 0u,
        SphericalTarget = core::createBitmask({ 0 }),
        DynamicPerspectiveFov = core::createBitmask({ 1 })
    };

    enum GoalStateMask : uint32_t
    {
        GoalStateNone = 0u,
        GoalStateSphericalTarget = core::createBitmask({ 0 }),
        GoalStateDynamicPerspective = core::createBitmask({ 1 }),
        GoalStatePath = core::createBitmask({ 2 })
    };

    struct SphericalTargetState
    {
        float64_t3 target = float64_t3(0.0);
        float distance = 0.f;
        double u = 0.0;
        double v = 0.0;
        float minDistance = 0.f;
        float maxDistance = 0.f;
    };

    struct DynamicPerspectiveState
    {
        float baseFov = 0.f;
        float referenceDistance = 0.f;
    };

    struct PathState
    {
        double angle = 0.0;
        double radius = 0.0;
        double height = 0.0;
    };

    // Gimbal with view parameters representing a camera in world space
    class CGimbal : public IGimbal<float64_t>
    {
    public:
        using base_t = IGimbal<float64_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) { updateView(); }
        ~CGimbal() = default;

        inline void updateView()
        {            
            const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

            assert(isOrthoBase(gRight, gUp, gForward));

            const auto& position = base_t::getPosition();

            m_viewMatrix[0u] = float64_t4(gRight, -hlsl::dot(gRight, position));
            m_viewMatrix[1u] = float64_t4(gUp, -hlsl::dot(gUp, position));
            m_viewMatrix[2u] = float64_t4(gForward, -hlsl::dot(gForward, position));
        }

        inline const float64_t3x4& getViewMatrix() const { return m_viewMatrix; }

    private:
        float64_t3x4 m_viewMatrix;
    };

    class SScopedMotionScaleOverride
    {
    public:
        SScopedMotionScaleOverride(ICamera* camera, const double moveScale, const double rotationScale)
            : m_camera(camera)
        {
            if (!m_camera)
                return;

            m_prevMoveScale = m_camera->getMoveSpeedScale();
            m_prevRotationScale = m_camera->getRotationSpeedScale();
            m_camera->setMotionScales(moveScale, rotationScale);
        }

        SScopedMotionScaleOverride(const SScopedMotionScaleOverride&) = delete;
        SScopedMotionScaleOverride& operator=(const SScopedMotionScaleOverride&) = delete;

        SScopedMotionScaleOverride(SScopedMotionScaleOverride&& other) noexcept
            : m_camera(std::exchange(other.m_camera, nullptr)),
            m_prevMoveScale(other.m_prevMoveScale),
            m_prevRotationScale(other.m_prevRotationScale)
        {
        }

        SScopedMotionScaleOverride& operator=(SScopedMotionScaleOverride&& other) = delete;

        ~SScopedMotionScaleOverride()
        {
            if (m_camera)
                m_camera->setMotionScales(m_prevMoveScale, m_prevRotationScale);
        }

    private:
        ICamera* m_camera = nullptr;
        double m_prevMoveScale = 0.0;
        double m_prevRotationScale = 0.0;
    };

    ICamera() {}
	virtual ~ICamera() = default;

    virtual const keyboard_to_virtual_events_t getKeyboardMappingPreset() const { return {}; }
    virtual const mouse_to_virtual_events_t getMouseMappingPreset() const { return {}; }
    virtual const imguizmo_to_virtual_events_t getImguizmoMappingPreset() const { return {}; }

    inline const IGimbalBindingLayout& getDefaultInputBindingLayout() const { return m_inputBindingConfig.defaultBindingLayout; }
    inline IGimbalBindingLayout& getDefaultInputBindingLayout() { return m_inputBindingConfig.defaultBindingLayout; }
    inline void copyDefaultInputBindingPresetTo(IGimbalBindingLayout& layout) const
    {
        layout.updateKeyboardMapping([&](auto& map) { map = getKeyboardMappingPreset(); });
        layout.updateMouseMapping([&](auto& map) { map = getMouseMappingPreset(); });
        layout.updateImguizmoMapping([&](auto& map) { map = getImguizmoMappingPreset(); });
    }

	// Returns a gimbal which *models the camera view*
	virtual const CGimbal& getGimbal() = 0u;

    // Camera core contract: consume virtual events only. Raw input binding and absolute goal solving live outside ICamera.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) = 0;

    // VirtualEventType bitmask for a camera view gimbal manipulation requests filtering
	virtual const uint32_t getAllowedVirtualEvents() = 0u;

    virtual CameraKind getKind() const = 0;
    virtual uint32_t getCapabilities() const { return None; }
    virtual uint32_t getGoalStateMask() const
    {
        uint32_t mask = GoalStateNone;
        if (hasCapability(SphericalTarget))
            mask |= GoalStateSphericalTarget;
        if (hasCapability(DynamicPerspectiveFov))
            mask |= GoalStateDynamicPerspective;
        return mask;
    }

    // Identifier of a camera type
    virtual const std::string_view getIdentifier() = 0u;

    inline bool hasCapability(CameraCapability capability) const
    {
        return (getCapabilities() & capability) == capability;
    }

    inline bool supportsGoalState(GoalStateMask goalState) const
    {
        return (getGoalStateMask() & goalState) == goalState;
    }

    virtual bool tryGetSphericalTargetState(SphericalTargetState& out) const
    {
        return false;
    }

    virtual bool trySetSphericalTarget(const float64_t3& target)
    {
        return false;
    }

    virtual bool trySetSphericalDistance(float distance)
    {
        return false;
    }

    virtual bool tryGetDynamicPerspectiveFov(float& outFov) const
    {
        return false;
    }

    virtual bool tryGetDynamicPerspectiveState(DynamicPerspectiveState& out) const
    {
        return false;
    }

    virtual bool trySetDynamicPerspectiveState(const DynamicPerspectiveState& state)
    {
        return false;
    }

    virtual bool tryGetPathState(PathState& out) const
    {
        return false;
    }

    virtual bool trySetPathState(const PathState& state)
    {
        return false;
    }

    // (***)
    inline void setMoveSpeedScale(double scalar)
    {
        m_motionConfig.moveSpeedScale = scalar;
    }

    // (***)
    inline void setRotationSpeedScale(double scalar)
    {
        m_motionConfig.rotationSpeedScale = scalar;
    }

    inline void setMotionScales(const double moveScale, const double rotationScale)
    {
        setMoveSpeedScale(moveScale);
        setRotationSpeedScale(rotationScale);
    }

    inline double getMoveSpeedScale() const { return m_motionConfig.moveSpeedScale; }
    inline double getRotationSpeedScale() const { return m_motionConfig.rotationSpeedScale; }
    inline const SMotionConfig& getMotionConfig() const { return m_motionConfig; }
    inline const SInputBindingConfig& getInputBindingConfig() const { return m_inputBindingConfig; }
    inline SScopedMotionScaleOverride overrideMotionScales(const double moveScale, const double rotationScale)
    {
        return SScopedMotionScaleOverride(this, moveScale, rotationScale);
    }
    inline void resetDefaultInputBindingToPreset()
    {
        copyDefaultInputBindingPresetTo(m_inputBindingConfig.defaultBindingLayout);
    }

protected:
    SMotionConfig m_motionConfig;
    SInputBindingConfig m_inputBindingConfig;
};

}

#endif // _I_CAMERA_HPP_
