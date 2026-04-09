#ifndef _C_PATH_CAMERA_HPP_
#define _C_PATH_CAMERA_HPP_

#include <algorithm>
#include <utility>

#include "CCameraPathUtilities.hpp"
#include "CSphericalTargetCamera.hpp"

namespace nbl::core
{

//! Path-rig camera driven by typed `PathState` plus an injected path model.
//!
//! The public runtime contract stays event-only through `manipulate(...)`.
//! `CPathCamera` only interprets the accumulated impulse through `m_pathModel`
//! instead of hardcoding one default target-relative mapping in the method body.
class CPathCamera final : public CSphericalTargetCamera
{
public:
    using base_t = CSphericalTargetCamera;
    using path_model_t = SCameraPathModel;

    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(position, target)
    {
        m_pathModel = CCameraPathUtilities::makeDefaultPathModel();
        m_pathModel.resolveState(target, position, SCameraPathDefaults::Limits, nullptr, m_pathState);
        updateFromPathState();
    }

    CPathCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target, path_model_t pathModel)
        : base_t(position, target)
        , m_pathModel(std::move(pathModel))
    {
        if (!m_pathModel.resolveState)
            m_pathModel = CCameraPathUtilities::makeDefaultPathModel();
        m_pathModel.resolveState(target, position, SCameraPathDefaults::Limits, nullptr, m_pathState);
        updateFromPathState();
    }

    ~CPathCamera() = default;

    const typename base_t::CGimbal& getGimbal() override { return m_gimbal; }

    virtual bool manipulate(std::span<const CVirtualGimbalEvent> virtualEvents, const hlsl::float64_t4x4* referenceFrame = nullptr) override
    {
        if (virtualEvents.empty() && !referenceFrame)
            return false;

        PathState nextPathState = m_pathState;
        CReferenceTransform reference = {};
        const CReferenceTransform* resolvedReference = nullptr;
        if (referenceFrame)
        {
            if (!m_gimbal.extractReferenceTransform(&reference, referenceFrame))
                return false;
            resolvedReference = &reference;
            if (!m_pathModel.resolveState ||
                !m_pathModel.resolveState(
                    m_targetPosition,
                    hlsl::float64_t3(reference.frame[3]),
                    SCameraPathDefaults::Limits,
                    nullptr,
                    nextPathState))
            {
                return false;
            }
        }

        const auto impulse = m_gimbal.accumulate<AllowedVirtualEvents>(virtualEvents);
        const SCameraPathControlContext context = {
            .currentState = nextPathState,
            .translation = scaleVirtualTranslation(impulse.dVirtualTranslate),
            .rotation = scaleVirtualRotation(impulse.dVirtualRotation),
            .targetPosition = m_targetPosition,
            .reference = resolvedReference,
            .limits = SCameraPathDefaults::Limits
        };

        if (!m_pathModel.controlLaw || !m_pathModel.integrate)
            return false;

        const auto stateDelta = m_pathModel.controlLaw(context);
        if (!m_pathModel.integrate(nextPathState, stateDelta, SCameraPathDefaults::Limits, nextPathState))
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
        if (!m_pathModel.resolveState)
            return false;

        PathState sanitized = {};
        if (!m_pathModel.resolveState(m_targetPosition, m_gimbal.getPosition(), SCameraPathDefaults::Limits, &state, sanitized))
            return false;

        const bool exact = CCameraPathUtilities::pathStatesNearlyEqual(sanitized, state, SCameraPathDefaults::ExactComparisonThresholds);
        m_pathState = sanitized;
        updateFromPathState();
        return exact;
    }

    virtual bool trySetSphericalDistance(float distance) override
    {
        SCameraPathDistanceUpdateResult distanceUpdate = {};
        if (!m_pathModel.updateDistance ||
            !m_pathModel.updateDistance(distance, SCameraPathDefaults::Limits, m_pathState, &distanceUpdate))
        {
            return false;
        }

        updateFromPathState();
        return distanceUpdate.exact;
    }

    virtual std::string_view getIdentifier() const override { return SCameraPathDefaults::Identifier; }

    inline const path_model_t& getPathModel() const
    {
        return m_pathModel;
    }

    inline bool setPathModel(path_model_t pathModel)
    {
        if (!pathModel.resolveState || !pathModel.controlLaw || !pathModel.integrate || !pathModel.evaluate || !pathModel.updateDistance)
            return false;

        PathState sanitized = {};
        if (!pathModel.resolveState(m_targetPosition, m_gimbal.getPosition(), SCameraPathDefaults::Limits, &m_pathState, sanitized))
            return false;

        m_pathModel = std::move(pathModel);
        m_pathState = sanitized;
        return updateFromPathState();
    }

private:
    static inline constexpr auto AllowedVirtualEvents =
        CVirtualGimbalEvent::Translate | CVirtualGimbalEvent::RollLeft | CVirtualGimbalEvent::RollRight;

    path_model_t m_pathModel = CCameraPathUtilities::makeDefaultPathModel();
    PathState m_pathState = CCameraPathUtilities::makeDefaultPathState(SCameraPathDefaults::Limits.minU);

    bool updateFromPathState()
    {
        if (!m_pathModel.evaluate)
            return false;

        SCameraCanonicalPathState canonicalPathState = {};
        if (!m_pathModel.evaluate(m_targetPosition, m_pathState, SCameraPathDefaults::Limits, canonicalPathState))
            return false;

        m_distance = canonicalPathState.targetRelative.distance;
        m_orbitUv = canonicalPathState.targetRelative.orbitUv;

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
