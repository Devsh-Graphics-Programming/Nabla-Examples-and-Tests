#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include "ICamera.hpp"

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
        : base_t(), m_targetPosition(target), m_distance(1.0f),
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
    struct SphericalBasis
    {
        hlsl::float64_t3 localSpherePosition = hlsl::float64_t3(0.0);
        hlsl::float64_t3 right = hlsl::float64_t3(1.0, 0.0, 0.0);
        hlsl::float64_t3 up = hlsl::float64_t3(0.0, 0.0, 1.0);
        hlsl::float64_t3 forward = hlsl::float64_t3(0.0, 1.0, 0.0);
    };

    inline SphericalBasis computeBasis(double orbitU, double orbitV, float distance) const
    {
        SphericalBasis basis;
        hlsl::float64_t3 position = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> orientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
        if (!hlsl::tryBuildSphericalPoseFromOrbit(
                m_targetPosition,
                orbitU,
                orbitV,
                static_cast<hlsl::float64_t>(distance),
                static_cast<hlsl::float64_t>(MinDistance),
                static_cast<hlsl::float64_t>(MaxDistance),
                position,
                orientation))
        {
            return basis;
        }

        basis.localSpherePosition = position - m_targetPosition;
        basis.right = orientation.transformVector(hlsl::float64_t3(1.0, 0.0, 0.0), true);
        basis.up = orientation.transformVector(hlsl::float64_t3(0.0, 1.0, 0.0), true);
        basis.forward = orientation.transformVector(hlsl::float64_t3(0.0, 0.0, 1.0), true);
        return basis;
    }

    inline void initFromPosition(const hlsl::float64_t3& position)
    {
        double orbitU = 0.0;
        double orbitV = 0.0;
        hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(MinDistance);
        if (!hlsl::tryBuildOrbitFromPosition(
                m_targetPosition,
                position,
                static_cast<hlsl::float64_t>(MinDistance),
                static_cast<hlsl::float64_t>(MaxDistance),
                orbitU,
                orbitV,
                appliedDistance))
        {
            m_distance = MinDistance;
            m_u = 0.0;
            m_v = 0.0;
            return;
        }

        m_distance = static_cast<float>(appliedDistance);
        m_u = orbitU;
        m_v = orbitV;
    }

    inline bool applyPose()
    {
        hlsl::float64_t3 newPosition = hlsl::float64_t3(0.0);
        hlsl::camera_quaternion_t<hlsl::float64_t> newOrientation = hlsl::makeIdentityQuaternion<hlsl::float64_t>();
        hlsl::float64_t appliedDistance = static_cast<hlsl::float64_t>(m_distance);
        if (!hlsl::tryBuildSphericalPoseFromOrbit(
                m_targetPosition,
                m_u,
                m_v,
                static_cast<hlsl::float64_t>(m_distance),
                static_cast<hlsl::float64_t>(MinDistance),
                static_cast<hlsl::float64_t>(MaxDistance),
                newPosition,
                newOrientation,
                &appliedDistance))
        {
            return false;
        }
        m_distance = static_cast<float>(appliedDistance);

        m_gimbal.begin();
        {
            m_gimbal.setPosition(newPosition);
            m_gimbal.setOrientation(newOrientation);
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
