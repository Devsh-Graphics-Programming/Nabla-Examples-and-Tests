#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include "CCameraTargetRelativeUtilities.hpp"

namespace nbl::core
{

/// @brief Common base for cameras orbiting or tracking a target with spherical coordinates.
///
/// The shared state is target position, distance, and orbit angles stored in `orbitUv`.
class CSphericalTargetCamera : public ICamera
{
public:
    using base_t = ICamera;

    CSphericalTargetCamera(const hlsl::float64_t3& position, const hlsl::float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(SCameraTargetRelativeRigDefaults::InitialDistance),
          m_gimbal({ .position = position, .orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float64_t>() })
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
    inline const hlsl::float64_t2& getOrbitUv() const { return m_orbitUv; }

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
        out.orbitUv = m_orbitUv;
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

    inline SphericalBasis computeBasis(const hlsl::float64_t2& orbitUv, float distance) const
    {
        SphericalBasis basis;
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitUv = orbitUv,
            .distance = distance
        };
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeBasis(state, MinDistance, MaxDistance, basis))
            return basis;
        return basis;
    }

    inline void initFromPosition(const hlsl::float64_t3& position)
    {
        SCameraTargetRelativeState state = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativeStateFromPosition(m_targetPosition, position, MinDistance, MaxDistance, state))
        {
            m_distance = MinDistance;
            m_orbitUv = hlsl::float64_t2(0.0);
            return;
        }

        m_distance = state.distance;
        m_orbitUv = state.orbitUv;
    }

    inline void applyPlanarTargetTranslation(const hlsl::float64_t3& deltaTranslation, const SphericalBasis& basis)
    {
        if (!hlsl::CCameraMathUtilities::hasPlanarDeltaXY(deltaTranslation, static_cast<hlsl::float64_t>(base_t::TinyScalarEpsilon)))
            return;

        m_targetPosition += hlsl::CCameraMathUtilities::transformLocalVectorToWorldBasis(
            hlsl::float64_t3(deltaTranslation.x, deltaTranslation.y, 0.0),
            basis.right,
            basis.up,
            basis.forward);
    }

    inline bool applyPose()
    {
        const SCameraTargetRelativeState state = {
            .target = m_targetPosition,
            .orbitUv = m_orbitUv,
            .distance = m_distance
        };
        SCameraTargetRelativePose pose = {};
        if (!CCameraTargetRelativeUtilities::tryBuildTargetRelativePoseFromState(state, MinDistance, MaxDistance, pose))
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
    hlsl::float64_t2 m_orbitUv = hlsl::float64_t2(0.0);
};

}

#endif

