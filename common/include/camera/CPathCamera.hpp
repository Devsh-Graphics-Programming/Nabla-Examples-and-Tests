#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include "CCameraPathUtilities.hpp"
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

//! Target-relative cylindrical path rig camera driven by `PathState`.
//!
//! The authored surface exposes this kind as `PathRig`. It stores a
//! cylindrical target-relative state `(angle, radius, height)` and rebuilds the
//! camera pose from that state through the shared path utilities.
class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;

    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        tryResolvePathState(target, position, SCameraPathDefaults::Limits, nullptr, m_pathState);
        updateFromPathState();
    }
    ~CPathCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (not virtualEvents.size() and not referenceFrame)
            return false;

        PathState nextPathState = m_pathState;
        if (referenceFrame)
        {
            CReferenceTransform reference = {};
            if (!m_gimbal.extractReferenceTransform(&reference, referenceFrame))
                return false;
            if (!tryResolvePathState(m_targetPosition, hlsl::float64_t3(reference.frame[3]), SCameraPathDefaults::Limits, nullptr, nextPathState))
                return false;
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const auto stateDelta = makePathDeltaFromVirtualPathTranslate(scaleVirtualTranslation(impulse.dVirtualTranslate));
        if (!tryApplyPathStateDelta(nextPathState, stateDelta, SCameraPathDefaults::Limits, nextPathState))
            return false;

        m_pathState = nextPathState;
        return updateFromPathState();
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
        if (!sanitizePathState(sanitized, SCameraPathDefaults::Limits.minRadius))
            return false;

        const bool exact = pathStatesNearlyEqual(sanitized, state, SCameraPathDefaults::ExactComparisonThresholds);
        m_pathState = sanitized;
        updateFromPathState();
        return exact;
    }
    virtual bool trySetSphericalDistance(float distance) override
    {
        SCameraPathDistanceUpdateResult distanceUpdate = {};
        if (!tryUpdatePathStateDistance(distance, SCameraPathDefaults::Limits, m_pathState, &distanceUpdate))
            return false;

        updateFromPathState();
        return distanceUpdate.exact;
    }
    virtual std::string_view getIdentifier() const override { return SCameraPathDefaults::Identifier; }

private:
    static inline constexpr auto AllowedVirtualEvents = CVirtualGimbalEvent::Translate;

    PathState m_pathState = makeDefaultPathState(SCameraPathDefaults::Limits.minRadius);

    bool updateFromPathState()
    {
        SCameraCanonicalPathState canonicalPathState = {};
        if (!tryBuildCanonicalPathState(m_targetPosition, m_pathState, SCameraPathDefaults::Limits, canonicalPathState))
        {
            return false;
        }

        m_distance = canonicalPathState.targetRelative.distance;
        m_u = canonicalPathState.targetRelative.orbitU;
        m_v = canonicalPathState.targetRelative.orbitV;

        m_gimbal.begin();
        {
            m_gimbal.setPosition(canonicalPathState.pose.position);
            m_gimbal.setOrientation(canonicalPathState.pose.orientation);
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
