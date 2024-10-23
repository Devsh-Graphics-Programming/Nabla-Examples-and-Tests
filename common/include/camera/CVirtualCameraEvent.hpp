#ifndef _NBL_VIRTUAL_CAMERA_EVENT_HPP_
#define _NBL_VIRTUAL_CAMERA_EVENT_HPP_

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl
{

//! Virtual camera event representing a manipulation
enum class VirtualEventType 
{
    //! Move the camera in the direction of strafe vector
    Strafe,

    //! Update orientation of camera by rotating around X, Y, Z axes
    Rotate,

    //! Signals boolean state, for example "reset"
    State
};

class CVirtualCameraEvent 
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

    CVirtualCameraEvent(VirtualEventType type, const ManipulationValue manipulation)
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

}

#endif // _NBL_VIRTUAL_CAMERA_EVENT_HPP_