#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include "CCameraTargetRelativeUtilities.hpp"

namespace nbl::core
{

/**
* Common base for cameras orbiting or tracking a target with spherical coordinates.
*
* The shared state is target position, distance, and orbit angles `u/v`.
*/
class CSphericalTargetCamera : public ICamera
{
public:
    using base_t = ICamera;

    CSphericalTargetCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(SCameraTargetRelativeRigDefaults::InitialDistance),
          m_gimbal({ .position = position, .orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>() })
    {
        initFromPosition(position);
    }
    ~CSphericalTargetCamera() = default;

    inline bool setDistance(float d)
    {
        const auto clamped = std::clamp<float>(d, MinDistance, MaxDistance);
        const bool ok = clamped == d;
        if (m_distance == clamped)
            return ok;
        m_distance = clamped;
        applyPose();
        return ok;
    }

    inline void target(const hlsl::float64_t3& p)
    {
        if (m_targetPosition == p)
            return;
        m_targetPosition = p;
        applyPose();
    }
    inline hlsl::float64_t3 getTarget() const { return m_targetPosition; }

    inline float getDistance() const { return m_distance; }
    inline double getU() const { return m_u; }
    inline double getV() const { return m_v; }

    static inline constexpr float MinDistance = base_t::SphericalMinDistance;
    static inline constexpr float MaxDistance = base_t::SphericalMaxDistance;

    virtual uint32_t getCapabilities() const override
    {
        return base_t::SphericalTarget;
    }

    virtual bool tryGetSphericalTargetState(typename base_t::SphericalTargetState& out) const override
    {
        out.target = m_targetPosition;
        out.distance = m_distance;
        out.u = m_u;
        out.v = m_v;
        out.minDistance = MinDistance;
        out.maxDistance = MaxDistance;
        return true;
    }

    virtual bool trySetSphericalTarget(const hlsl::float64_t3& targetPosition) override
    {
        target(targetPosition);
        return true;
    }

    virtual bool trySetSphericalDistance(float distance) override
    {
        return setDistance(distance);
    }

protected:
    using SphericalBasis = SCameraTargetRelativeBasis;

    inline SphericalBasis computeBasis(double orbitU, double orbitV, float distance) const
    {
        SphericalBasis basis;
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitU = orbitU,
            .orbitV = orbitV,
            .distance = distance
        };
        if (!tryBuildTargetRelativeBasis(state, MinDistance, MaxDistance, basis))
            return basis;
        return basis;
    }

    inline void initFromPosition(const hlsl::float64_t3& position)
    {
        SCameraTargetRelativeState state = {};
        if (!tryBuildTargetRelativeStateFromPosition(m_targetPosition, position, MinDistance, MaxDistance, state))
        {
            m_distance = MinDistance;
            m_u = 0.0;
            m_v = 0.0;
            return;
        }

        m_distance = state.distance;
        m_u = state.orbitU;
        m_v = state.orbitV;
    }

    inline void applyPlanarTargetTranslation(const hlsl::float64_t3& deltaTranslation, const SphericalBasis& basis)
    {
        if (!hlsl::hasPlanarDeltaXY(deltaTranslation, static_cast<hlsl::float64_t>(base_t::TinyScalarEpsilon)))
            return;

        m_targetPosition += hlsl::transformLocalVectorToWorldBasis(
            hlsl::float64_t3(deltaTranslation.x, deltaTranslation.y, 0.0),
            basis.right,
            basis.up,
            basis.forward);
    }

    inline bool applyPose()
    {
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitU = m_u,
            .orbitV = m_v,
            .distance = m_distance
        };
        SCameraTargetRelativePose pose = {};
        if (!tryBuildTargetRelativePoseFromState(state, MinDistance, MaxDistance, pose))
            return false;
        m_distance = static_cast<float>(pose.appliedDistance);

        m_gimbal.begin();
        {
            m_gimbal.setPosition(pose.position);
            m_gimbal.setOrientation(pose.orientation);
        }
        m_gimbal.end();

        const bool manipulated = bool(m_gimbal.getManipulationCounter());
        if (manipulated)
            m_gimbal.updateView();

        return manipulated;
    }

    hlsl::float64_t3 m_targetPosition;
    float m_distance;
    typename base_t::CGimbal m_gimbal;
    double m_u = {};
    double m_v = {};
};

}

#endif
