#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CPathCamera(const float64_t3& position, const float64_t3& target)
        : base_t(position, target)
    {
        const auto offset = position - target;
        m_pathRadius = std::sqrt(offset.x * offset.x + offset.z * offset.z);
        if (m_pathRadius < MinPathRadius)
            m_pathRadius = MinPathRadius;
        m_pathHeight = offset.y;
        m_pathAngle = std::atan2(offset.z, offset.x);
        updateFromPath();
    }
    ~CPathCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);

        constexpr double translateScalar = 0.01;
        const double moveScalar = translateScalar * getMoveSpeedScale();

        m_pathAngle += impulse.dVirtualTranslate.z * moveScalar;
        m_pathRadius = std::max(MinPathRadius, m_pathRadius + impulse.dVirtualTranslate.x * moveScalar);
        m_pathHeight += impulse.dVirtualTranslate.y * moveScalar;

        return updateFromPath();
    }

    virtual uint32_t getAllowedVirtualEvents() const override { return AllowedVirtualEvents; }
    virtual CameraKind getKind() const override { return CameraKind::Path; }
    virtual uint32_t getGoalStateMask() const override { return base_t::getGoalStateMask() | base_t::GoalStatePath; }
    virtual bool tryGetPathState(PathState& out) const override
    {
        out.angle = m_pathAngle;
        out.radius = m_pathRadius;
        out.height = m_pathHeight;
        return true;
    }
    virtual bool trySetPathState(const PathState& state) override
    {
        if (!std::isfinite(state.angle) || !std::isfinite(state.radius) || !std::isfinite(state.height))
            return false;

        m_pathAngle = state.angle;
        m_pathRadius = std::max(MinPathRadius, state.radius);
        m_pathHeight = state.height;
        const bool exact = std::abs(m_pathRadius - state.radius) <= 1e-9;
        updateFromPath();
        return exact;
    }
    virtual bool trySetSphericalDistance(float distance) override
    {
        const auto clamped = std::clamp<float>(distance, MinDistance, MaxDistance);
        const bool inRange = clamped == distance;

        const double currentDistance = std::sqrt(m_pathRadius * m_pathRadius + m_pathHeight * m_pathHeight);
        if (currentDistance > 1e-9)
        {
            const double scale = static_cast<double>(clamped) / currentDistance;
            m_pathRadius = std::max(MinPathRadius, m_pathRadius * scale);
            m_pathHeight *= scale;
        }
        else
        {
            m_pathRadius = std::max(MinPathRadius, static_cast<double>(clamped));
            m_pathHeight = 0.0;
        }

        updateFromPath();

        const double appliedDistance = std::sqrt(m_pathRadius * m_pathRadius + m_pathHeight * m_pathHeight);
        return inRange && std::abs(appliedDistance - static_cast<double>(clamped)) <= 1e-6;
    }
    virtual std::string_view getIdentifier() const override { return "Path Camera"; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;
    static inline constexpr double MinPathRadius = 0.1;

    double m_pathAngle = 0.0;
    double m_pathRadius = 1.0;
    double m_pathHeight = 0.0;

    bool updateFromPath()
    {
        const double x = std::cos(m_pathAngle) * m_pathRadius;
        const double z = std::sin(m_pathAngle) * m_pathRadius;
        const float64_t3 position = m_targetPosition + float64_t3(x, m_pathHeight, z);
        initFromPosition(position);
        return applyPose();
    }
};

}

#endif
