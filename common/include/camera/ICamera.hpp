// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <optional>

#include "camera/CGimbalInputBinder.hpp"

namespace nbl::hlsl // TODO: DIFFERENT NAMESPACE
{

class ICamera : public IGimbalBindingLayout, virtual public core::IReferenceCounted
{ 
public:
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

    ICamera() {}
	virtual ~ICamera() = default;

    virtual const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() const override { return m_defaultInputBinding.getKeyboardVirtualEventMap(); }
    virtual const mouse_to_virtual_events_t& getMouseVirtualEventMap() const override { return m_defaultInputBinding.getMouseVirtualEventMap(); }
    virtual const imguizmo_to_virtual_events_t& getImguizmoVirtualEventMap() const override { return m_defaultInputBinding.getImguizmoVirtualEventMap(); }

    virtual void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys) override { m_defaultInputBinding.updateKeyboardMapping(mapKeys); }
    virtual void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys) override { m_defaultInputBinding.updateMouseMapping(mapKeys); }
    virtual void updateImguizmoMapping(const std::function<void(imguizmo_to_virtual_events_t&)>& mapKeys) override { m_defaultInputBinding.updateImguizmoMapping(mapKeys); }

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
        m_moveSpeedScale = scalar;
    }

    // (***)
    inline void setRotationSpeedScale(double scalar)
    {
        m_rotationSpeedScale = scalar;
    }

    inline double getMoveSpeedScale() const { return m_moveSpeedScale; }
    inline double getRotationSpeedScale() const { return m_rotationSpeedScale; }

protected:
    
    // (***) TODO: I need to think whether a camera should own this or controllers should be able 
    // to set sensitivity to scale magnitudes of generated events we put into manipulate method
    double m_moveSpeedScale = 0.01, m_rotationSpeedScale = 0.003;
    CGimbalInputBinder m_defaultInputBinding;
};

}

#endif // _I_CAMERA_HPP_
