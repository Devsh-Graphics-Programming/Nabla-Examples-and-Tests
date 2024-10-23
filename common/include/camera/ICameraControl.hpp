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

    enum VirtualEventType : uint8_t
    {
        // Strafe forward
        MoveForward = 0,

        // Strafe backward
        MoveBackward,

        // Strafe left
        MoveLeft,

        // Strafe right
        MoveRight,

        // Strafe up
        MoveUp,

        // Strafe down
        MoveDown,

        // Tilt the camera upward (pitch)
        TiltUp,

        // Tilt the camera downward (pitch)
        TiltDown,

        // Rotate the camera left around the vertical axis (yaw)
        PanLeft,

        // Rotate the camera right around the vertical axis (yaw)
        PanRight,

        // Roll the camera counterclockwise around the forward axis (roll)
        RollLeft,

        // Roll the camera clockwise around the forward axis (roll)
        RollRight,

        // Reset the camera to the default state
        Reset,

        EventsCount
    };

    class CGimbal : virtual public core::IReferenceCounted
    {
    public:
        //! Virtual event representing a manipulation
        enum VirtualEventType
        {
            //! Move the camera in the direction of strafe vector
            Strafe,

            //! Update orientation of camera by rotating around X, Y, Z axes
            Rotate,

            //! Signals boolean state, for example "reset"
            State
        };

        class CVirtualEvent
        {
        public:
            using manipulation_encode_t = float32_t4;

            struct StrafeManipulation
            {
                float32_t3 direction = {};
                float distance = {};
            };

            struct RotateManipulation
            {
                float pitch = {}, roll = {}, yaw = {};
            };

            struct StateManipulation
            {
                uint32_t reset : 1;
                uint32_t reserved : 31;

                StateManipulation() { memset(this, 0, sizeof(StateManipulation)); }
                ~StateManipulation() {}
            };

            union ManipulationValue
            {
                StrafeManipulation strafe;
                RotateManipulation rotation;
                StateManipulation state;

                ManipulationValue() { memset(this, 0, sizeof(ManipulationValue)); }
                ~ManipulationValue() {}
            };

            CVirtualEvent(VirtualEventType type, const ManipulationValue manipulation)
                : m_type(type), m_manipulation(manipulation)
            {
                static_assert(sizeof(manipulation_encode_t) == sizeof(ManipulationValue));
            }

            // Returns the type of manipulation value
            VirtualEventType getType() const
            {
                return m_type;
            }

            // Returns manipulation value
            ManipulationValue getManipulation() const
            {
                return m_manipulation;
            }

        private:
            VirtualEventType m_type;
            ManipulationValue m_manipulation;
        };

        CGimbal(const core::smart_refctd_ptr<projection_t>&& projection, const float32_t3& position, const float32_t3& lookat, const float32_t3& upVec = float32_t3(0.0f, 1.0f, 0.0f), const float32_t3& backupUpVec = float32_t3(0.5f, 1.0f, 0.0f))
            : m_projection(projection), m_position(position), m_target(lookat), m_upVec(upVec), m_backupUpVec(backupUpVec), m_initialPosition(position), m_initialTarget(lookat), m_orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), m_viewMatrix({}), m_isLeftHanded(isLeftHanded(m_projection->getProjectionMatrix()))
        {
            recomputeViewMatrix();
        }

        // TODO: ctor with core::path to json config file to load defaults

        //! Start a gimbal manipulation session
        inline void begin()
        {
            m_needsToRecomputeViewMatrix = false;
            m_recordingManipulation = true;
        }

        //! Record manipulation of the gimbal, note those events are delta manipulations
        void manipulate(const CVirtualEvent& virtualEvent)
        {
            if (!m_recordingManipulation)
                return; // TODO: log it

            const auto manipulation = virtualEvent.getManipulation();

            case VirtualEventType::Strafe:
            {
                strafe(manipulation.strafe.direction, manipulation.strafe.distance);
            } break;

            case VirtualEventType::Rotate:
            {
                rotate(manipulation.rotation.pitch, manipulation.rotation.yaw, manipulation.rotation.roll);
            } break;

            case VirtualEventType::State:
            {
                if (manipulation.state.reset)
                    reset();
            } break;

            default:
                break;
        }

        // Record change of position vector, global update
        inline void setPosition(const float32_t3& position)
        {
            if (!m_recordingManipulation)
                return; // TODO: log it

            m_position = position;
        }

        // Record change of target vector, global update
        inline void setTarget(const float32_t3& target)
        {
            if (!m_recordingManipulation)
                return; // TODO: log it

            m_target = target;
        }

        // Change up vector, global update
        inline void setUpVector(const float32_t3& up) { m_upVec = up; }

        // Change backupUp vector, global update
        inline void setBackupUpVector(const float32_t3& backupUp) { m_backupUpVec = backupUp; }

        //! End the gimbal manipulation session, recompute view matrix if required and update handedness state from projection
        inline void end()
        {
            m_isLeftHanded = isLeftHanded(m_projection->getProjectionMatrix());

            if (m_needsToRecomputeViewMatrix)
                recomputeViewMatrix();

            m_needsToRecomputeViewMatrix = false;
            m_recordingManipulation = false;
        }

        inline const float32_t3& getPosition() const { return m_position; }
        inline const float32_t3& getTarget() const { return m_target; }
        inline const float32_t3& getUpVector() const { return m_upVec; }
        inline const float32_t3& getBackupUpVector() const { return m_backupUpVec; }
        inline const float32_t3 getLocalTarget() const { return m_target - m_position; }
        inline const float32_t3 getForwardDirection() const { return glm::normalize(getLocalTarget()); }
        inline const projection_t* getProjection() { return m_projection.get(); }

        // TODO: getConcatenatedMatrix()
        // TODO: getViewMatrix()

    private:
        //! Reset the gimbal to its initial position, target, and orientation
        inline void reset()
        {
            m_position = m_initialPosition;
            m_target = m_initialTarget;
            m_orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

            recomputeViewMatrix();  // Recompute the view matrix after resetting
        }

        //! Move in the direction of strafe (mostly left/right, up/down)
        inline void strafe(const glm::vec3& direction, float distance)
        {
            if (distance != 0.0f)
            {
                const auto strafeVector = glm::normalize(direction) * distance;
                m_position += strafeVector;
                m_target += strafeVector;

                m_needsToRecomputeViewMatrix = true;
            }
        }

        //! Update orientation by rotating around all XYZ axes - delta rotations in radians
        inline void rotate(float dPitchRadians, float dYawDeltaRadians, float dRollDeltaRadians)
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
            m_needsToRecomputeViewMatrix = true;
        }

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
            return hlsl::determinant(projectionMatrix) < 0.f;
        }

        core::smart_refctd_ptr<projection_t> m_projection;
        float32_t3 m_position, m_target, m_upVec, m_backupUpVec;
        const float32_t3 m_initialPosition, m_initialTarget;

        glm::quat m_orientation;
        float64_t4x4 m_viewMatrix;

        bool m_isLeftHanded, 
        m_needsToRecomputeViewMatrix = false,
        m_recordingManipulation = false;
    };

    ICameraController() : {}

    // controller overrides how a manipulation is done for a given camera event with a gimbal
    virtual void manipulate(CGimbal* gimbal, VirtualEventType event) = 0;

    // controller can override default set of event map
    virtual void initKeysToEvent()
    {
        m_keysToEvent[MoveForward] = { ui::E_KEY_CODE::EKC_W };
        m_keysToEvent[MoveBackward] = { ui::E_KEY_CODE::EKC_S };
        m_keysToEvent[MoveLeft] = { ui::E_KEY_CODE::EKC_A };
        m_keysToEvent[MoveRight] = { ui::E_KEY_CODE::EKC_D };
        m_keysToEvent[MoveUp] = { ui::E_KEY_CODE::EKC_SPACE };
        m_keysToEvent[MoveDown] = { ui::E_KEY_CODE::EKC_LEFT_SHIFT };
        m_keysToEvent[TiltUp] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[TiltDown] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[PanLeft] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[PanRight] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[RollLeft] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[RollRight] = { ui::E_KEY_CODE::EKC_NONE };
        m_keysToEvent[Reset] = { ui::E_KEY_CODE::EKC_R };
    }

    // controller can override which keys correspond to which event
    void updateKeysToEvent(const std::vector<ui::E_KEY_CODE>& codes, VirtualEventType event)
    {
        m_keysToEvent[event] = std::move(codes);
    }

protected:

    inline void setMoveSpeed(const float moveSpeed) { moveSpeed = m_moveSpeed; }
    inline void setRotateSpeed(const float rotateSpeed) { rotateSpeed = m_rotateSpeed; }

    inline const float getMoveSpeed() const { return m_moveSpeed; }
    inline const float getRotateSpeed() const { return m_rotateSpeed; }

    std::array<std::vector<ui::E_KEY_CODE>, EventsCount> m_keysToEvent = {};

    // speed factors
    float m_moveSpeed = 1.f, m_rotateSpeed = 1.f;

    // states signaling if keys are pressed down or not
    bool m_keysDown[EventsCount] = {};

    // durations for which the key was being held down from lastVirtualUpTimeStamp(=last "guessed" presentation time) to nextPresentationTimeStamp
    double m_perActionDt[EventsCount] = {};

    std::chrono::microseconds nextPresentationTimeStamp, lastVirtualUpTimeStamp;
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_