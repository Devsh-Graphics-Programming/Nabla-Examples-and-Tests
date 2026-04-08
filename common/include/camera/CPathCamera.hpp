#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        if (!hlsl::tryBuildPathStateFromPosition(
                target,
                position,
                MinPathRadius,
                m_pathState.angle,
                m_pathState.radius,
                m_pathState.height))
        {
            m_pathState = {
                .angle = 0.0,
                .radius = MinPathRadius,
                .height = 0.0
            };
        }
        updateFromPath();
    }
    ~CPathCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        m_pathState.angle += scaleVirtualTranslation(impulse.dVirtualTranslate.z);
        m_pathState.radius += scaleVirtualTranslation(impulse.dVirtualTranslate.x);
        m_pathState.height += scaleVirtualTranslation(impulse.dVirtualTranslate.y);
        if (!hlsl::sanitizePathState(m_pathState.angle, m_pathState.radius, m_pathState.height, MinPathRadius))
            return false;

        return updateFromPath();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Path; }
    virtual uint32_t getGoalStateMask() const override { return base_t::getGoalStateMask() | base_t::GoalStatePath; }
    virtual bool tryGetPathState(PathState& out) const override
    {
        out = m_pathState;
        return true;
    }
    virtual bool trySetPathState(const PathState& state) override
    {
        auto sanitized = state;
        if (!hlsl::sanitizePathState(sanitized.angle, sanitized.radius, sanitized.height, MinPathRadius))
            return false;

        const bool exact = hlsl::nearlyEqualScalar(
            static_cast<hlsl::float64_t>(sanitized.radius),
            static_cast<hlsl::float64_t>(state.radius),
            static_cast<hlsl::float64_t>(ICamera::TinyScalarEpsilon));
        m_pathState = sanitized;
        updateFromPath();
        return exact;
    }
    virtual bool trySetSphericalDistance(float distance) override
    {
        const auto clamped = std::clamp<float>(distance, MinDistance, MaxDistance);
        const bool inRange = clamped == distance;

        if (!hlsl::tryScalePathStateDistance(
                static_cast<double>(clamped),
                MinPathRadius,
                m_pathState.radius,
                m_pathState.height))
            return false;

        updateFromPath();

        const double appliedDistance = hlsl::getPathDistance(m_pathState.radius, m_pathState.height);
        return inRange && std::abs(appliedDistance - static_cast<double>(clamped)) <= ICamera::ScalarTolerance;
    }
    virtual std::string_view getIdentifier() const override { return "Path Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double MinPathRadius = ICamera::SphericalMinDistance;

    PathState m_pathState = { .angle = 0.0, .radius = 1.0, .height = 0.0 };

    bool updateFromPath()
    {
        hlsl::float64_t3 position = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
        hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(m_distance);
        double orbitU = m_u;
        double orbitV = m_v;
        if (!hlsl::tryBuildPathPoseFromState(
                m_targetPosition,
                m_pathState.angle,
                m_pathState.radius,
                m_pathState.height,
                MinPathRadius,
                static_cast<hlsl::float64_t>(MinDistance),
                static_cast<hlsl::float64_t>(MaxDistance),
                position,
                orientation,
                &appliedDistance,
                &orbitU,
                &orbitV))
        {
            return false;
        }

        m_distance = static_cast<float>(appliedDistance);
        m_u = orbitU;
        m_v = orbitV;

        m_gimbal.begin();
        {
            m_gimbal.setPosition(position);
            m_gimbal.setOrientation(orientation);
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());
        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }
};

}

#endif
