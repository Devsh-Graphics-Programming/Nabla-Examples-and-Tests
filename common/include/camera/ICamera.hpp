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

/// @brief Shared camera interface.
///
/// The hot runtime path is event-only: cameras consume `CVirtualGimbalEvent`
/// streams through `manipulate(...)`. Optional typed state hooks exist only for
/// tooling features such as capture, compatibility analysis, presets, and playback.
class ICamera : virtual public core::IReferenceCounted
{ 
public:
    /// @brief Shared lower bound used by spherical and path rigs for valid camera distance.
    static inline constexpr float SphericalMinDistance = 0.1f;
    /// @brief Shared upper bound used by spherical and path rigs for valid camera distance.
    static inline constexpr float SphericalMaxDistance = 10000.f;
    /// @brief Base runtime translation magnitude represented by a unit virtual move event.
    static inline constexpr double VirtualTranslationStep = 0.01;
    /// @brief Default multiplier applied to virtual translation magnitudes.
    static inline constexpr double DefaultMoveSpeedScale = VirtualTranslationStep;
    /// @brief Default multiplier applied to virtual rotation magnitudes.
    static inline constexpr double DefaultRotationSpeedScale = 0.003;
    /// @brief Shared scalar epsilon used by typed tooling comparisons.
    static inline constexpr double ScalarTolerance = 1e-6;
    /// @brief Very small epsilon used when exact replay helpers need stricter comparisons.
    static inline constexpr double TinyScalarEpsilon = 1e-9;
    /// @brief Default world-space position tolerance used by pose comparisons.
    static inline constexpr double DefaultPositionTolerance = 2.0 * ScalarTolerance;
    /// @brief Default angular tolerance in degrees used by pose and state comparisons.
    static inline constexpr double DefaultAngularToleranceDeg = 0.1;

    /// @brief Camera-local multipliers applied when translating virtual events into motion.
    struct SMotionConfig
    {
        /// @brief Camera-local scales applied by implementations to virtual motion magnitude.
        double moveSpeedScale = DefaultMoveSpeedScale;
        double rotationSpeedScale = DefaultRotationSpeedScale;
    };

    /// @brief Stable runtime camera-family identifier used by tooling, metadata, and default presets.
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

    /// @brief Optional typed capabilities exposed by a concrete runtime camera implementation.
    enum CameraCapability : uint32_t
    {
        None = 0u,
        SphericalTarget = core::createBitmask({ 0 }),
        DynamicPerspectiveFov = core::createBitmask({ 1 })
    };

    /// @brief Typed goal-state fragments that tooling may capture from or apply to a camera.
    enum GoalStateMask : uint32_t
    {
        GoalStateNone = 0u,
        GoalStateSphericalTarget = core::createBitmask({ 0 }),
        GoalStateDynamicPerspective = core::createBitmask({ 1 }),
        GoalStatePath = core::createBitmask({ 2 })
    };

    /// @brief Canonical spherical-target state shared by orbit-like cameras.
    ///
    /// The state stores the tracked target position, orbit angles in `orbitUv`,
    /// and distance limits needed by tooling that wants to capture or reapply a
    /// target-relative camera pose without going through free-form setters.
    struct SphericalTargetState
    {
        /// @brief Tracked target position in world space.
        hlsl::float64_t3 target = hlsl::float64_t3(0.0);
        /// @brief Orbit yaw and pitch around the target, expressed in radians.
        hlsl::float64_t2 orbitUv = hlsl::float64_t2(0.0);
        /// @brief Current camera-to-target distance.
        float distance = 0.f;
        /// @brief Lowest distance that remains valid for the current camera.
        float minDistance = 0.f;
        /// @brief Highest distance that remains valid for the current camera.
        float maxDistance = 0.f;
    };

    /// @brief Typed authored state used by cameras with derived perspective behavior.
    struct DynamicPerspectiveState
    {
        /// @brief Authored reference FOV in degrees.
        float baseFov = 0.f;
        /// @brief Distance at which `baseFov` should be preserved.
        float referenceDistance = 0.f;
    };

    /// @brief Limits constraining reusable `PathState` coordinates for `Path Rig` cameras.
    struct PathStateLimits
    {
        /// @brief Minimal valid `u` coordinate after path-state sanitization.
        double minU = static_cast<double>(SphericalMinDistance);
        /// @brief Minimal valid radial distance derived from the `(u, v)` pair.
        hlsl::float64_t minDistance = static_cast<hlsl::float64_t>(SphericalMinDistance);
        /// @brief Maximal valid radial distance derived from the `(u, v)` pair.
        hlsl::float64_t maxDistance = static_cast<hlsl::float64_t>(SphericalMaxDistance);
    };

    /// @brief Parametric path-rig state used by the `Path Rig` camera kind.
    ///
    /// The default shared model interprets `(s, u, v, roll)` as angular progress,
    /// radial component, vertical component, and view-axis roll around a target.
    /// Concrete path models may reuse the same coordinates differently, while the
    /// hot runtime path still stays event-only through `manipulate(...)`.
    struct PathState
    {
        /// @brief Primary path-progress coordinate interpreted by the active path model.
        double s = 0.0;
        /// @brief First lateral/shape coordinate interpreted by the active path model.
        double u = 0.0;
        /// @brief Second lateral/shape coordinate interpreted by the active path model.
        double v = 0.0;
        /// @brief Roll around the path-model forward axis, expressed in radians.
        double roll = 0.0;

        /// @brief Pack the state into one four-component vector for math helpers and persistence tooling.
        inline hlsl::float64_t4 asVector() const
        {
            return hlsl::float64_t4(s, u, v, roll);
        }

        /// @brief Project the state onto the shared translation-style view used by replay helpers.
        inline hlsl::float64_t3 asTranslationVector() const
        {
            return hlsl::float64_t3(u, v, s);
        }

        /// @brief Rebuild one path state from the packed vector representation.
        static inline PathState fromVector(const hlsl::float64_t4& value)
        {
            return {
                .s = value.x,
                .u = value.y,
                .v = value.z,
                .roll = value.w
            };
        }

        /// @brief Rebuild one path state from the translation-style helper representation.
        static inline PathState fromTranslationVector(const hlsl::float64_t3& value, const double pathRoll = 0.0)
        {
            return {
                .s = value.z,
                .u = value.x,
                .v = value.y,
                .roll = pathRoll
            };
        }
    };

    /// @brief Gimbal that models the camera pose and cached view matrix in world space.
    class CGimbal : public IGimbal<hlsl::float64_t>
    {
    public:
        using base_t = IGimbal<hlsl::float64_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) { updateView(); }
        ~CGimbal() = default;

        /// @brief Rebuild the cached world-to-view matrix from the current gimbal pose.
        inline void updateView()
        {            
            const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

            assert(hlsl::isOrthoBase(gRight, gUp, gForward));

            const auto& position = base_t::getPosition();

            m_viewMatrix[0u] = hlsl::float64_t4(gRight, -hlsl::dot(gRight, position));
            m_viewMatrix[1u] = hlsl::float64_t4(gUp, -hlsl::dot(gUp, position));
            m_viewMatrix[2u] = hlsl::float64_t4(gForward, -hlsl::dot(gForward, position));
        }

        /// @brief Return the cached world-to-view matrix derived from the current pose.
        inline const hlsl::float64_t3x4& getViewMatrix() const { return m_viewMatrix; }

    private:
        hlsl::float64_t3x4 m_viewMatrix;
    };

    class SScopedMotionScaleOverride
    {
    public:
        /// @brief Temporarily override both motion scales and restore the previous values on destruction.
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

    /// @brief Return the mutable gimbal backing the runtime camera pose.
	virtual const CGimbal& getGimbal() = 0u;

    /// @brief Consume virtual events only.
    ///
    /// Raw input binding and absolute goal solving live outside `ICamera`.
    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) = 0;
    /// @brief Apply one frame of virtual events while temporarily overriding the camera-local motion scales.
    inline bool manipulateWithMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame, const double moveScale, const double rotationScale)
    {
        auto scopedOverride = overrideMotionScales(moveScale, rotationScale);
        return manipulate(virtualEvents, referenceFrame);
    }
    /// @brief Apply one frame of virtual events with unit translation and rotation scales.
    inline bool manipulateWithUnitMotionScales(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr)
    {
        return manipulateWithMotionScales(virtualEvents, referenceFrame, 1.0, 1.0);
    }

    /// @brief Return the semantic virtual-event mask accepted by this camera kind.
	virtual uint32_t getAllowedVirtualEvents() const = 0u;

    /// @brief Return the stable camera-family identifier for this concrete runtime camera.
    virtual CameraKind getKind() const = 0;
    /// @brief Return the optional typed capabilities exposed by this camera implementation.
    virtual uint32_t getCapabilities() const { return None; }
    /// @brief Return the typed goal-state fragments that tooling may safely use with this camera.
    virtual uint32_t getGoalStateMask() const
    {
        uint32_t mask = GoalStateNone;
        if (hasCapability(SphericalTarget))
            mask |= GoalStateSphericalTarget;
        if (hasCapability(DynamicPerspectiveFov))
            mask |= GoalStateDynamicPerspective;
        return mask;
    }

    /// @brief Return the stable human-readable identifier for this concrete camera instance.
    virtual std::string_view getIdentifier() const = 0u;

    /// @brief Check whether the camera exposes the requested optional capability.
    inline bool hasCapability(CameraCapability capability) const
    {
        return (getCapabilities() & capability) == capability;
    }

    /// @brief Check whether the camera can exchange the requested typed goal-state fragment.
    inline bool supportsGoalState(GoalStateMask goalState) const
    {
        return (getGoalStateMask() & goalState) == goalState;
    }

    /// @brief Query the current spherical-target state when the camera exposes it.
    virtual bool tryGetSphericalTargetState(SphericalTargetState& out) const
    {
        return false;
    }

    /// @brief Replace only the tracked target position for spherical-target cameras.
    virtual bool trySetSphericalTarget(const hlsl::float64_t3& target)
    {
        return false;
    }

    /// @brief Replace only the tracked target distance for spherical-target cameras.
    virtual bool trySetSphericalDistance(float distance)
    {
        return false;
    }

    /// @brief Query the current derived dynamic perspective FOV when the camera exposes it.
    virtual bool tryGetDynamicPerspectiveFov(float& outFov) const
    {
        return false;
    }

    /// @brief Query the current authored dynamic perspective state when the camera exposes it.
    virtual bool tryGetDynamicPerspectiveState(DynamicPerspectiveState& out) const
    {
        return false;
    }

    /// @brief Replace the authored dynamic perspective state when the camera exposes it.
    virtual bool trySetDynamicPerspectiveState(const DynamicPerspectiveState& state)
    {
        return false;
    }

    /// @brief Query the current typed path state when the camera exposes it.
    virtual bool tryGetPathState(PathState& out) const
    {
        return false;
    }

    /// @brief Query the active typed limits constraining the current path state.
    virtual bool tryGetPathStateLimits(PathStateLimits& out) const
    {
        return false;
    }

    /// @brief Replace the current typed path state when the camera exposes it.
    virtual bool trySetPathState(const PathState& state)
    {
        return false;
    }

    /// @brief Update only the translation motion scale used by the camera runtime.
    inline void setMoveSpeedScale(double scalar)
    {
        m_motionConfig.moveSpeedScale = scalar;
    }

    /// @brief Update only the rotation motion scale used by the camera runtime.
    inline void setRotationSpeedScale(double scalar)
    {
        m_motionConfig.rotationSpeedScale = scalar;
    }

    /// @brief Update both translation and rotation motion scales at once.
    inline void setMotionScales(const double moveScale, const double rotationScale)
    {
        setMoveSpeedScale(moveScale);
        setRotationSpeedScale(rotationScale);
    }

    /// @brief Return the current translation motion scale.
    inline double getMoveSpeedScale() const { return m_motionConfig.moveSpeedScale; }
    /// @brief Return the current rotation motion scale.
    inline double getRotationSpeedScale() const { return m_motionConfig.rotationSpeedScale; }
    /// @brief Return the full motion-scale bundle.
    inline const SMotionConfig& getMotionConfig() const { return m_motionConfig; }
    /// @brief Return the effective world-space translation represented by a unit virtual move event.
    inline double getScaledVirtualTranslationMagnitude() const
    {
        return VirtualTranslationStep * getMoveSpeedScale();
    }
    /// @brief Return the raw translation magnitude before applying the camera-local move scale.
    inline double getUnscaledVirtualTranslationMagnitude() const
    {
        return VirtualTranslationStep;
    }
    /// @brief Scale one scalar translation magnitude through the active move scale.
    inline double scaleVirtualTranslation(const double magnitude) const
    {
        return magnitude * getScaledVirtualTranslationMagnitude();
    }
    /// @brief Scale one translation vector through the active move scale.
    template<typename T, uint32_t N>
    inline hlsl::camera_vector_t<T, N> scaleVirtualTranslation(const hlsl::camera_vector_t<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getScaledVirtualTranslationMagnitude());
    }
    /// @brief Scale one scalar translation magnitude without applying the camera-local move scale.
    inline double scaleUnscaledVirtualTranslation(const double magnitude) const
    {
        return magnitude * getUnscaledVirtualTranslationMagnitude();
    }
    /// @brief Scale one translation vector without applying the camera-local move scale.
    template<typename T, uint32_t N>
    inline hlsl::camera_vector_t<T, N> scaleUnscaledVirtualTranslation(const hlsl::camera_vector_t<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getUnscaledVirtualTranslationMagnitude());
    }
    /// @brief Scale one scalar rotation magnitude through the active rotation scale.
    inline double scaleVirtualRotation(const double magnitude) const
    {
        return magnitude * getRotationSpeedScale();
    }
    /// @brief Scale one rotation vector through the active rotation scale.
    template<typename T, uint32_t N>
    inline hlsl::camera_vector_t<T, N> scaleVirtualRotation(const hlsl::camera_vector_t<T, N>& magnitude) const
    {
        return magnitude * static_cast<T>(getRotationSpeedScale());
    }
    /// @brief Create a scoped helper that restores the previous motion scales on destruction.
    inline SScopedMotionScaleOverride overrideMotionScales(const double moveScale, const double rotationScale)
    {
        return SScopedMotionScaleOverride(this, moveScale, rotationScale);
    }

protected:
    SMotionConfig m_motionConfig;
};

}

#endif // _I_CAMERA_HPP_
