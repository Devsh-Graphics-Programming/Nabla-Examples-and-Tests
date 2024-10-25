#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include "IProjection.hpp"
#include "../ICamera.hpp"
#include "CVirtualCameraEvent.hpp"
#include "glm/glm/ext/matrix_transform.hpp" // TODO: TEMPORARY!!! whatever used will be moved to cpp
#include "glm/glm/gtc/quaternion.hpp"
#include "nbl/builtin/hlsl/matrix_utils/transformation_matrix_utils.hlsl"

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

    //! Virtual event representing a manipulation
    struct CVirtualEvent
    {
        using manipulation_encode_t = float64_t;

        VirtualEventType type;
        manipulation_encode_t value;
    };

    class CGimbal : virtual public core::IReferenceCounted
    {
    public:
        //! Virtual event representing a combined gimbal manipulation
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

            CVirtualEvent() {}

            CVirtualEvent(VirtualEventType _type, const ManipulationValue _manipulation)
                : type(_type), manipulation(_manipulation)
            {
                static_assert(sizeof(manipulation_encode_t) == sizeof(ManipulationValue));
            }

            VirtualEventType type;
            ManipulationValue manipulation;
        };

        CGimbal(const core::smart_refctd_ptr<projection_t>&& projection, const float32_t3& position, const float32_t3& lookat, const float32_t3& upVec = float32_t3(0.0f, 1.0f, 0.0f), const float32_t3& backupUpVec = float32_t3(0.5f, 1.0f, 0.0f))
            : m_projection(projection), m_position(position), m_target(lookat), m_upVec(upVec), m_backupUpVec(backupUpVec), m_initialPosition(position), m_initialTarget(lookat), m_orientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f)), m_viewMatrix({})
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

            const auto& manipulation = virtualEvent.manipulation;

            switch (virtualEvent.type)
            {
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
            if (m_needsToRecomputeViewMatrix)
                recomputeViewMatrix();

            m_needsToRecomputeViewMatrix = false;
            m_recordingManipulation = false;
        }

        inline bool isRecordingManipulation() { return m_recordingManipulation; }
        inline projection_t* getProjection() { return m_projection.get(); }
        inline const float32_t3& getPosition() const { return m_position; }
        inline const float32_t3& getTarget() const { return m_target; }
        inline const float32_t3& getUpVector() const { return m_upVec; }
        inline const float32_t3& getBackupUpVector() const { return m_backupUpVec; }
        inline const float32_t3 getLocalTarget() const { return m_target - m_position; }
        inline const float32_t3 getForwardDirection() const { return glm::normalize(getLocalTarget()); }
        inline const float32_t3x4& getViewMatrix() const { return m_viewMatrix; }

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

            // Rotate around Z (roll)
            glm::quat qRoll = glm::angleAxis(dRollDeltaRadians, glm::vec3(0.0f, 0.0f, 1.0f));

            // Combine the new rotations with the current orientation
            m_orientation = glm::normalize(qYaw * qPitch * qRoll * m_orientation);

            // Now we have rotation transformation as 3x3 matrix
            auto rotate = glm::mat3_cast(m_orientation);

            // We do not change magnitude of the vector
            auto localTargetRotated = rotate * getLocalTarget();

            // And we can simply update target vector
            m_target = m_position + localTargetRotated;

            m_needsToRecomputeViewMatrix = true;
        }

        inline void recomputeViewMatrix()
        {
            auto up = getPatchedUpVector();

            if (m_projection->isLeftHanded())
                m_viewMatrix = buildCameraLookAtMatrixLH<float>(m_position, m_target, up);
            else
                m_viewMatrix = buildCameraLookAtMatrixRH<float>(m_position, m_target, up);
        }

        core::smart_refctd_ptr<projection_t> m_projection;
        float32_t3 m_position, m_target, m_upVec, m_backupUpVec;
        const float32_t3 m_initialPosition, m_initialTarget;

        glm::quat m_orientation;
        float32_t3x4 m_viewMatrix;

        bool m_needsToRecomputeViewMatrix = true,
        m_recordingManipulation = false;
    };

    ICameraController() {}

    // override controller keys map, it binds a key code to a virtual event
    void updateKeysToEvent(const std::vector<ui::E_KEY_CODE>& codes, VirtualEventType event)
    {
        m_keysToEvent[event] = std::move(codes);
    }

    // start controller manipulation session
    virtual void begin(std::chrono::microseconds nextPresentationTimeStamp)
    {
        m_nextPresentationTimeStamp = nextPresentationTimeStamp;
        return;
    }

    // manipulate camera with gimbal & virtual events, begin must be called before that!
    virtual void manipulate(CGimbal* gimbal, std::span<const CVirtualEvent> virtualEvents) = 0;

    // finish controller manipulation session, call after last manipulate in the hot loop
    void end(std::chrono::microseconds nextPresentationTimeStamp)
    {
        m_lastVirtualUpTimeStamp = nextPresentationTimeStamp;
    }

    /*
    // process keyboard to generate virtual manipulation events
    // note that:
    // - it doesn't make the manipulation itself!
    */
    std::vector<CVirtualEvent> processKeyboard(std::span<const ui::SKeyboardEvent> events)
    {
        std::vector<CVirtualEvent> virtualEvents;

        for (const auto& ev : events)
        {
            constexpr auto NblVirtualKeys = std::to_array({ MoveForward, MoveBackward, MoveLeft, MoveRight, MoveUp, MoveDown, TiltUp, TiltDown, PanLeft, PanRight, RollLeft, RollRight, Reset });
            static_assert(NblVirtualKeys.size() == EventsCount);

            for (const auto virtualKey : NblVirtualKeys)
            {
                const auto code = m_keysToEvent[virtualKey];

                if (ev.keyCode == code)
                {
                    if (code == ui::EKC_NONE)
                        continue;

                    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(m_nextPresentationTimeStamp - ev.timeStamp).count();
                    assert(dt >= 0);

                    if (ev.action == nbl::ui::SKeyboardEvent::ECA_PRESSED && !m_keysDown[virtualKey])
                    {
                        m_keysDown[virtualKey] = true;
                        virtualEvents.emplace_back(CVirtualEvent{ virtualKey, static_cast<float64_t>(dt) });
                    }
                    else if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
                    {
                        m_keysDown[virtualKey] = false;
                    }
                }
            }
        }

        return virtualEvents;
    }

    /*
    // [OPTIONAL]: process mouse to generate virtual manipulation events
    // note that:
    // - all manipulations *may* be done with keyboard keys (if you have a touchpad or sth an ui:: event could be a code!)
    // - it doesn't make the manipulation itself!
    */
    std::vector<CVirtualEvent> processMouse(std::span<const ui::SMouseEvent> events) const
    {
        // accumulate total pitch & yaw delta from mouse movement events
        const auto [dTotalPitch, dTotalYaw] = [&]()
        {
            double dPitch = {}, dYaw = {};

            for (const auto& ev : events)
                if (ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT)
                {
                    dYaw += ev.movementEvent.relativeMovementX;  // (yaw)
                    dPitch -= ev.movementEvent.relativeMovementY; // (pitch)
                }

            return std::make_tuple(dPitch, dYaw);
        }();

        CVirtualEvent pitch;
        pitch.type = (pitch.value = dTotalPitch) > 0.f ? TiltUp : TiltDown;

        CVirtualEvent yaw;
        yaw.type = (yaw.value = dTotalYaw) > 0.f ? PanRight : PanLeft;

        return { pitch, yaw };
    }

    inline void setMoveSpeed(const float moveSpeed) { m_moveSpeed = moveSpeed; }
    inline void setRotateSpeed(const float rotateSpeed) { m_rotateSpeed = rotateSpeed; }

    inline const float getMoveSpeed() const { return m_moveSpeed; }
    inline const float getRotateSpeed() const { return m_rotateSpeed; }

protected:
    // controller can override default set of event map
    virtual void initKeysToEvent()
    {
        m_keysToEvent[MoveForward] = ui::E_KEY_CODE::EKC_W ;
        m_keysToEvent[MoveBackward] =  ui::E_KEY_CODE::EKC_S ;
        m_keysToEvent[MoveLeft] =  ui::E_KEY_CODE::EKC_A ;
        m_keysToEvent[MoveRight] =  ui::E_KEY_CODE::EKC_D ;
        m_keysToEvent[MoveUp] =  ui::E_KEY_CODE::EKC_SPACE ;
        m_keysToEvent[MoveDown] =  ui::E_KEY_CODE::EKC_LEFT_SHIFT ;
        m_keysToEvent[TiltUp] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[TiltDown] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[PanLeft] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[PanRight] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[RollLeft] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[RollRight] =  ui::E_KEY_CODE::EKC_NONE ;
        m_keysToEvent[Reset] =  ui::E_KEY_CODE::EKC_R ;
    }

    std::array<ui::E_KEY_CODE, EventsCount> m_keysToEvent = {};
    float m_moveSpeed = 1.f, m_rotateSpeed = 1.f;
    bool m_keysDown[EventsCount] = {};

    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_