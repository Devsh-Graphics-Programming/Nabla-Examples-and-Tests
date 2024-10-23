#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include "IProjection.hpp"
#include "CVirtualCameraEvent.hpp"
#include "glm/glm/ext/matrix_transform.hpp" // TODO: TEMPORARY!!! whatever used will be moved to cpp
#include "glm/glm/gtc/quaternion.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl 
{

template<ProjectionMatrix T = float64_t4x4>
class ICameraController : virtual public core::IReferenceCounted
{
public:
    using projection_t = typename IProjection<typename T>;

    class CGimbal : virtual public core::IReferenceCounted
    {
    public:
        CGimbal(const core::smart_refctd_ptr<projection_t>&& projection, const float32_t3& position, const float32_t3& lookat, const float32_t3& upVec = float32_t3(0.0f, 1.0f, 0.0f), const float32_t3& backupUpVec = float32_t3(0.5f, 1.0f, 0.0f))
            : m_projection(projection), m_position(position), m_target(lookat), m_upVec(upVec), m_backupUpVec(backupUpVec), m_initialPosition(position), m_initialTarget(lookat), m_orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), m_viewMatrix({}), m_isLeftHanded(isLeftHanded(m_projection->getProjectionMatrix()))
        {
            recomputeViewMatrix();
        }

        //! Start a gimbal manipulation session
        inline void begin()
        {
            needsToRecomputeViewMatrix = false;
        }

        //! End the gimbal manipulation session, recompute matrices and check projection
        inline void end()
        {
            m_isLeftHanded = isLeftHanded(m_projection->getProjectionMatrix());

            // Recompute the view matrix
            if(needsToRecomputeViewMatrix)
                recomputeViewMatrix();

            needsToRecomputeViewMatrix = false;
        }

        inline float32_t3 getLocalTarget() const
        {
            return m_target - m_position;
        }

        inline float32_t3 getForwardDirection() const
        {
            return glm::normalize(getLocalTarget());
        }

        //! Reset the gimbal to its initial position, target, and orientation
        inline void reset()
        {
            m_position = m_initialPosition; 
            m_target = m_initialTarget;
            m_orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

            recomputeViewMatrix();  // Recompute the view matrix after resetting
        }

        //! Move the camera in the direction of strafe (mostly left/right, up/down)
        void strafe(const glm::vec3& direction, float distance)
        {
            if (distance != 0.0f)
            {
                const auto strafeVector = glm::normalize(direction) * distance;
                m_position += strafeVector;
                m_target += strafeVector;

                needsToRecomputeViewMatrix = true;
            }
        }

        //! Update orientation of camera by rotating around all XYZ axes - delta rotations in radians
        void rotate(float dPitchRadians, float dYawDeltaRadians, float dRollDeltaRadians)
        {
            // Rotate around X (pitch)
            glm::quat qPitch = glm::angleAxis(dPitchRadians, glm::vec3(1.0f, 0.0f, 0.0f));

            // Rotate around Y (yaw)
            glm::quat qYaw = glm::angleAxis(dYawDeltaRadians, glm::vec3(0.0f, 1.0f, 0.0f));

            // Rotate around Z (roll) // TODO: handness!!
            glm::quat qRoll = glm::angleAxis(dRollDeltaRadians, glm::vec3(0.0f, 0.0f, 1.0f));

            // Combine the new rotations with the current orientation
            m_orientation = glm::normalize(qYaw * qPitch * qRoll * m_orientation);

            // Now we have rotation transformation as 3x3 matrix
            auto rotate = glm::mat3_cast(m_orientation);

            // We do not change magnitude of the vector
            auto localTargetRotated = rotate * getLocalTarget();

            // And we can simply update target vector
            m_target = m_position + localTargetRotated;

            // TODO: std::any + nice checks for deltas (radians - periodic!)
            needsToRecomputeViewMatrix = true;
        }

        // TODO: ctor with core::path to json config file to load defaults

        const projection_t* getProjection() { return m_projection.get(); }

    private:
        inline void recomputeViewMatrix()
        {
            auto up = getPatchedUpVector();

            if (m_isLeftHanded)
                m_viewMatrix = glm::lookAtLH(m_position, m_target, up);
            else
                m_viewMatrix = glm::lookAtRH(m_position, m_target, up);
        }

        inline float32_t3 getPatchedUpVector()
        {
            // if up vector and vector to the target are the same we patch the up vector
            auto up = glm::normalize(m_upVec);

            const auto localTarget = getForwardDirection();
            const auto cross = glm::cross(localTarget, up);

            // we compute squared length but for checking if the len is 0 it doesnt matter 
            const bool upVectorZeroLength = glm::dot(cross, cross) == 0.f;

            if (upVectorZeroLength)
                up = glm::normalize(m_backupUpVec);

            return up;
        }

        inline bool isLeftHanded(const auto& projectionMatrix)
        {
            return nbl::hlsl::determinant(projectionMatrix) < 0.f;
        }

        const core::smart_refctd_ptr<projection_t> m_projection;
        float32_t3 m_position, m_target, m_upVec, m_backupUpVec;
        const float32_t3 m_initialPosition, m_initialTarget;

        glm::quat m_orientation;
        float64_t4x4 m_viewMatrix;
        bool m_isLeftHanded;

        bool needsToRecomputeViewMatrix = false;
    };

    ICameraController(core::smart_refctd_ptr<CGimbal>&& gimbal)
        : m_gimbal(std::move(gimbal)) {}

    void processVirtualEvent(const CVirtualCameraEvent& virtualEvent)
    {
        const auto manipulation = virtualEvent.getManipulation();

        case VirtualEventType::Strafe:
        {
            m_gimbal->strafe(manipulation.strafe.direction, manipulation.strafe.distance);
        } break;
    
        case VirtualEventType::Rotate:
        {
            m_gimbal->rotate(manipulation.rotation.pitch, manipulation.rotation.yaw, manipulation.rotation.roll);
        } break;

        case VirtualEventType::State:
        {
            if (manipulation.state.reset)
                m_gimbal->reset();
        } break;

        default:
            break;
    }

private:
    core::smart_refctd_ptr<CGimbal> m_gimbal;
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_