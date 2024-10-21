#ifndef _NBL_VIRTUAL_CAMERA_EVENT_HPP_
#define _NBL_VIRTUAL_CAMERA_EVENT_HPP_

#include "nbl/builtin/hlsl/cpp_compat/matrix.hlsl"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl
{

//! Virtual camera event representing a manipulation
enum class VirtualEventType 
{
    //! Moves the camera forward or backward
    MoveForward,

    //! Zooms in or out
    Zoom,

    //! Moves the camera left/right and up/down
    Strafe,

    //! Rotates the camera horizontally
    Pan,

    //! Rotates the camera vertically
    Tilt,

    //! Rolls the camera around the Z axis
    Roll,

    //! Resets the camera to a default position and orientation
    Reset
};

//! We encode all manipulation type values into a vec4 & assume they are written in NDC coordinate system with respect to camera
using manipulation_value_t = float64_t4x4;

class CVirtualCameraEvent 
{
public:
    CVirtualCameraEvent(VirtualEventType type, const manipulation_value_t value)
        : type(type), value(value) {}

    // Returns the type of virtual event
    VirtualEventType getType() const 
    {
        return type;
    }

    // Returns the associated value for manipulation
    manipulation_value_t getValue() const 
    {
        return value;
    }

private:
    VirtualEventType type;
    manipulation_value_t value;
};

}

#endif // _NBL_VIRTUAL_CAMERA_EVENT_HPP_