// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _I_CAMERA_HPP_
#define _I_CAMERA_HPP_

#include <optional>
#include <utility>

#include "IGimbal.hpp"

namespace nbl::core
{

/**
* Shared camera contract.
*
* The hot runtime path is event-only: cameras consume `CVirtualGimbalEvent`
* streams through `manipulate(...)`. Optional typed state hooks exist only for
* tooling features such as capture, compatibility analysis, presets, and playback.
*/
class ICamera : virtual public core::IReferenceCounted
{ 
public:
    static inline constexpr float SphericalMinDistance = 0.1f;
    static inline constexpr float SphericalMaxDistance = 10000.f;
    static inline constexpr double VirtualTranslationStep = 0.01;
    static inline constexpr double DefaultMoveSpeedScale = VirtualTranslationStep;
    static inline constexpr double DefaultRotationSpeedScale = 0.003;
    static inline constexpr double ScalarTolerance = 1e-6;
    static inline constexpr double TinyScalarEpsilon = 1e-9;
    static inline constexpr double DefaultPositionTolerance = 2.0 * ScalarTolerance;
    static inline constexpr double DefaultAngularToleranceDeg = 0.1;

    struct SMotionConfig
    {
        //! Camera-local scales applied by implementations to virtual motion magnitude.
        double moveSpeedScale = DefaultMoveSpeedScale;
        double rotationSpeedScale = DefaultRotationSpeedScale;
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
        hlsl::float64_t3 target = hlsl::float64_t3(0.0);
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

    //! Gimbal that models the camera pose and cached view matrix in world space.
    class CGimbal : public IGimbal<hlsl::float64_t>
    {
    public:
        using base_t = IGimbal<hlsl::float64_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) { updateView(); }
        ~CGimbal() = default;

        inline void updateView()
        {            
            const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

            assert(hlsl::isOrthoBase(gRight, gUp, gForward));

            const auto& position = base_t::getPosition();

            m_viewMatrix[0u] = hlsl::float64_t4(gRight, -hlsl::dot(gRight, position));
            m_viewMatrix[1u] = hlsl::float64_t4(gUp, -hlsl::dot(gUp, position));
            m_viewMatrix[2u] = hlsl::float64_t4(gForward, -hlsl::dot(gForward, position));
        }

        inline const hlsl::float64_t3x4& getViewMatrix() const { return m_viewMatrix; }

    private:
        hlsl::float64_t3x4 m_viewMatrix;
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

	virtual const CGimbal& getGimbal() = 0u;

    // Camera core contract: consume virtual events only. Raw input binding and absolute goal solving live outside ICamera.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) = 0;
    inline bool manipulateWithMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame, const double moveScale, const double rotationScale)
    {
        auto scopedOverride = overrideMotionScales(moveScale, rotationScale);
        return manipulate(virtualEvents, referenceFrame);
    }
    inline bool manipulateWithUnitMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr)
    {
        return manipulateWithMotionScales(virtualEvents, referenceFrame, 1.0, 1.0);
    }

	virtual uint32_t getAllowedVirtualEvents() const = 0u;

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

    virtual std::string_view getIdentifier() const = 0u;

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

    virtual bool trySetSphericalTarget(const hlsl::float64_t3& target)
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

    inline void setMoveSpeedScale(double scalar)
    {
        m_motionConfig.moveSpeedScale = scalar;
    }

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
    inline double getScaledVirtualTranslationMagnitude() const
    {
        return VirtualTranslationStep * getMoveSpeedScale();
    }
    inline double getUnscaledVirtualTranslationMagnitude() const
    {
        return VirtualTranslationStep;
    }
    inline double scaleVirtualTranslation(const double magnitude) const
    {
        return magnitude * getScaledVirtualTranslationMagnitude();
    }
    inline double scaleUnscaledVirtualTranslation(const double magnitude) const
    {
        return magnitude * getUnscaledVirtualTranslationMagnitude();
    }
    inline double scaleVirtualRotation(const double magnitude) const
    {
        return magnitude * getRotationSpeedScale();
    }
    inline SScopedMotionScaleOverride overrideMotionScales(const double moveScale, const double rotationScale)
    {
        return SScopedMotionScaleOverride(this, moveScale, rotationScale);
    }

protected:
    SMotionConfig m_motionConfig;
};

}

#endif // _I_CAMERA_HPP_
