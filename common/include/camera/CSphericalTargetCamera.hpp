#ifndef _C_SPHERICAL_TARGET_CAMERA_HPP_
#define _C_SPHERICAL_TARGET_CAMERA_HPP_

#include <algorithm>
#include <cmath>

#include "ICamera.hpp"

namespace nbl::hlsl
{

class CSphericalTargetCamera : public ICamera
{
public:
    using base_t = ICamera;

    CSphericalTargetCamera(const float64_t3& position, const float64_t3& target)
        : base_t(), m_targetPosition(target), m_distance(1.0f),
          m_gimbal({ .position = position, .orientation = glm::quat(glm::vec3(0.0f)) })
    {
        initFromPosition(position);
    }
    ~CSphericalTargetCamera() = default;

    inline bool setDistance(float d)
    {
        const auto clamped = std::clamp<float>(d, MinDistance, MaxDistance);
        const bool ok = clamped == d;
        m_distance = clamped;
        return ok;
    }

    inline void target(const float64_t3& p) { m_targetPosition = p; }
    inline float64_t3 getTarget() const { return m_targetPosition; }

    inline float getDistance() const { return m_distance; }
    inline double getU() const { return m_u; }
    inline double getV() const { return m_v; }

    static inline constexpr float MinDistance = 0.1f;
    static inline constexpr float MaxDistance = 10000.f;

protected:
    struct SphericalBasis
    {
        float64_t3 localSpherePosition;
        float64_t3 right;
        float64_t3 up;
        float64_t3 forward;
    };

    inline float64_t3 S(double su, double sv) const
    {
        return float64_t3
        {
            std::cos(sv) * std::cos(su),
            std::cos(sv) * std::sin(su),
            std::sin(sv)
        };
    }

    inline float64_t3 Sdv(double su, double sv) const
    {
        return float64_t3
        {
            -std::sin(sv) * std::cos(su),
            -std::sin(sv) * std::sin(su),
            std::cos(sv)
        };
    }

    inline SphericalBasis computeBasis(double su, double sv, float distance) const
    {
        const auto localSpherePosition = S(su, sv) * static_cast<double>(distance);
        const auto forward = normalize(-localSpherePosition);
        const auto up = normalize(Sdv(su, sv));
        const auto right = normalize(cross(up, forward));

        return { localSpherePosition, right, up, forward };
    }

    inline void initFromPosition(const float64_t3& position)
    {
        const auto offset = position - m_targetPosition;
        const double dist = length(offset);
        const double safeDist = std::isfinite(dist) && dist > 0.0 ? dist : static_cast<double>(MinDistance);
        m_distance = std::clamp<float>(static_cast<float>(safeDist), MinDistance, MaxDistance);
        const auto local = offset / static_cast<double>(m_distance);
        m_u = std::atan2(local.y, local.x);
        m_v = std::asin(std::clamp(local.z, -1.0, 1.0));
    }

    inline bool applyPose()
    {
        const auto basis = computeBasis(m_u, m_v, m_distance);
        const auto newPosition = basis.localSpherePosition + m_targetPosition;
        const auto newOrientation = glm::quat_cast(glm::dmat3{ basis.right, basis.up, basis.forward });

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

    float64_t3 m_targetPosition;
    float m_distance;
    typename base_t::CGimbal m_gimbal;
    double m_u = {};
    double m_v = {};
};

}

#endif
