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
        CGimbal(const float32_t3& position, glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
            : m_position(position), m_orientation(orientation) { updateOrthonormalMatrix(); }

        inline void setPosition(const float32_t3& position)
        {
            m_position = position;
        }

        inline void rotate(const float32_t3& axis, float dRadians)
        {
            glm::quat dRotation = glm::angleAxis(dRadians, axis);
            m_orientation = glm::normalize(dRotation * m_orientation);
            updateOrthonormalMatrix();
        }

        inline void strafe(float distance)
        {
            move({ 0.f, distance, 0.f });
        }

        inline void climb(float distance)
        {
            move({ 0.f, 0.f, distance });
        }

        inline void advance(float distance)
        {
            move({ distance, 0.f, 0.f });
        }

        inline void move(float32_t3 delta)
        {
            m_position += mul(m_orthonormal, delta);
        }

        inline void reset()
        {
            // TODO
        }

        // Position of gimbal
        inline const float32_t3& getPosition() const { return m_position; }

        // Orientation of gimbal
        inline const glm::quat& getOrientation() const { return m_orientation; }

        // Orthonormal [getXAxis(), getYAxis(), getZAxis()] matrix
        inline const float32_t3x3& getOrthonornalMatrix() const { return m_orthonormal; }

        // Base right vector in orthonormal basis, base "right" vector (X-axis)
        inline const float32_t3& getXAxis() const { return m_orthonormal[0u]; }

        // Base up vector in orthonormal basis, base "up" vector (Y-axis)
        inline const float32_t3& getYAxis() const { return m_orthonormal[1u]; }

        // Base forward vector in orthonormal basis, base "forward" vector (Z-axis)
        inline const float32_t3& getZAxis() const { return m_orthonormal[2u]; }

    private:
        inline void updateOrthonormalMatrix() 
        { 
            m_orthonormal = float32_t3x3(glm::mat3_cast(glm::normalize(m_orientation))); 

            // DEBUG
            const auto [xaxis, yaxis, zaxis] = std::make_tuple(getXAxis(),getYAxis(), getZAxis());

            auto isNormalized = [](const auto& v, float epsilon) -> bool
            {
                return glm::epsilonEqual(glm::length(v), 1.0f, epsilon);
            };

            auto isOrthogonal = [](const auto& a, const auto& b, float epsilon) -> bool
            {
                return glm::epsilonEqual(glm::dot(a, b), 0.0f, epsilon);
            };

            auto isOrthoBase = [&](const auto& x, const auto& y, const auto& z, float epsilon = 1e-6f) -> bool
            {
                return isNormalized(x, epsilon) && isNormalized(y, epsilon) && isNormalized(z, epsilon) &&
                    isOrthogonal(x, y, epsilon) && isOrthogonal(x, z, epsilon) && isOrthogonal(y, z, epsilon);
            };

            assert(isOrthoBase(xaxis, yaxis, zaxis));
        }

        float32_t3 m_position;
        glm::quat m_orientation;

        // Represents the camera's orthonormal basis
        // https://en.wikipedia.org/wiki/Orthonormal_basis
        float32_t3x3 m_orthonormal;
    };

    ICameraController(core::smart_refctd_ptr<CGimbal>&& gimbal) : m_gimbal(core::smart_refctd_ptr(gimbal)) {}

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
    virtual void manipulate(std::span<const CVirtualEvent> virtualEvents) = 0;

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
        std::vector<CVirtualEvent> output;

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
                        output.emplace_back(CVirtualEvent{ virtualKey, static_cast<float64_t>(dt) });
                    }
                    else if (ev.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
                    {
                        m_keysDown[virtualKey] = false;
                    }
                }
            }
        }

        return output;
    }

    /*
    // [OPTIONAL]: process mouse to generate virtual manipulation events
    // note that:
    // - all manipulations *may* be done with keyboard keys (if you have a touchpad or sth an ui:: event could be a code!)
    // - it doesn't make the manipulation itself!
    */
    std::vector<CVirtualEvent> processMouse(std::span<const ui::SMouseEvent> events) const
    {
        double dPitch = {}, dYaw = {};

        for (const auto& ev : events)
            if (ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT)
            {
                dYaw += ev.movementEvent.relativeMovementX;
                dPitch += ev.movementEvent.relativeMovementY;
            }

        std::vector<CVirtualEvent> output;

        if (dPitch)
        {
            auto& pitch = output.emplace_back();
            pitch.type = dPitch > 0.f ? TiltUp : TiltDown;
            pitch.value = std::abs(dPitch);
        }

        if (dYaw)
        {
            auto& yaw = output.emplace_back();
            yaw.type = dYaw > 0.f ? PanRight : PanLeft;
            yaw.value = std::abs(dYaw);
        }

        return output;
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

    core::smart_refctd_ptr<CGimbal> m_gimbal;
    std::array<ui::E_KEY_CODE, EventsCount> m_keysToEvent = {};
    float m_moveSpeed = 1.f, m_rotateSpeed = 1.f;
    bool m_keysDown[EventsCount] = {};

    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_