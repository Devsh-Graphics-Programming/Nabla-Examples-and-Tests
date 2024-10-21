#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include "IProjection.hpp"
#include "CVirtualCameraEvent.hpp"
#include "glm/glm/ext/matrix_transform.hpp" // TODO: TEMPORARY!!! whatever used will be moved to cpp

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
            : m_projection(std::move(projection)), m_position(position), m_target(lookat), m_upVec(upVec), m_backupUpVec(backupUpVec), m_initialPosition(position), m_initialTarget(lookat), m_viewMatrix({}), m_isLeftHanded(nbl::hlsl::determinant(m_projection->getProjectionMatrix()) < 0.f)
        {
            //! I make an assumption that we take ownership of projection (just to make less calculations on gimbal update iterations [but maybe I shouldnt, think of it!])
            recomputeViewMatrix();
        }

        // TODO: ctor with core::path to json config file to load defaults

        const projection_t* getProjection() { return m_projection.get(); }

        // TODO: gimbal methods (to handle VirtualEventType requests)

    private:
        inline void recomputeViewMatrix()
        {
            auto localTarget = glm::normalize(m_target - m_position);

            // If up vector and vector to the target are the same, adjust the up vector
            auto up = glm::normalize(m_upVec);
            auto cross = glm::cross(localTarget, up);
            bool upVectorNeedsChange = glm::dot(cross, cross) == 0;

            if (upVectorNeedsChange)
                up = glm::normalize(m_backupUpVec);

            if (m_isLeftHanded)
                m_viewMatrix = glm::lookAtLH(m_position, m_target, up);
            else
                m_viewMatrix = glm::lookAtRH(m_position, m_target, up);
        }

        const core::smart_refctd_ptr<projection_t> m_projection;
        float32_t3 m_position, m_target, m_upVec, m_backupUpVec;
        const float32_t3 m_initialPosition, m_initialTarget;

        float64_t4x4 m_viewMatrix;
        const bool m_isLeftHanded;
    };

    ICameraController(core::smart_refctd_ptr<CGimbal>&& gimbal)
        : m_gimbal(std::move(gimbal)) {}

    void processVirtualEvent(const CVirtualCameraEvent& virtualEvent)
    {
        // we will treat all manipulation event values as NDC, also for non manipulation events
        // we will define how to handle it, all values are encoded onto vec4 (manipulation_value_t)
        manipulation_value_t value = virtualEvent.getValue();

        // TODO: this will use gimbal to handle a virtual event registered by a class (wip) which maps physical keys to virtual events

        case VirtualEventType::MoveForward:
        {
            // TODO
        } break;
    
        case VirtualEventType::Strafe:
        {
            // TODO
        } break;

        case VirtualEventType::Zoom:
        {
            // TODO
        } break;

        case VirtualEventType::Pan:
        {
            // TODO
        } break;

        case VirtualEventType::Tilt:
        {
            // TODO
        } break;

        case VirtualEventType::Roll:
        {
            // TODO
        } break;

        case VirtualEventType::Reset:
        {
            // TODO
        } break;

        default:
            break;
    }

private:
    core::smart_refctd_ptr<CGimbal> m_gimbal;
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_