#ifndef _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_
#define _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_

#include <array>
#include <span>
#include <vector>

#include "CCameraMathUtilities.hpp"
#include "ICamera.hpp"

namespace nbl::core
{

struct SCameraVirtualEventAxisBinding final
{
    CVirtualGimbalEvent::VirtualEventType positive = CVirtualGimbalEvent::None;
    CVirtualGimbalEvent::VirtualEventType negative = CVirtualGimbalEvent::None;
};

struct SCameraVirtualEventBindings final
{
    static inline constexpr std::array<SCameraVirtualEventAxisBinding, 3u> LocalTranslation = {{
        { CVirtualGimbalEvent::MoveRight, CVirtualGimbalEvent::MoveLeft },
        { CVirtualGimbalEvent::MoveUp, CVirtualGimbalEvent::MoveDown },
        { CVirtualGimbalEvent::MoveForward, CVirtualGimbalEvent::MoveBackward }
    }};
};

inline void appendSignedVirtualEvent(
    std::vector<CVirtualGimbalEvent>& events,
    const double value,
    const CVirtualGimbalEvent::VirtualEventType positive,
    const CVirtualGimbalEvent::VirtualEventType negative,
    const double tolerance = static_cast<double>(ICamera::TinyScalarEpsilon))
{
    if (!hlsl::isFiniteScalar(value) || hlsl::isNearlyZeroScalar(value, tolerance))
        return;

    auto& ev = events.emplace_back();
    ev.type = (value > 0.0) ? positive : negative;
    ev.magnitude = hlsl::abs(value);
}

inline void appendScaledVirtualEvent(
    std::vector<CVirtualGimbalEvent>& events,
    const double value,
    const double denominator,
    const double tolerance,
    const CVirtualGimbalEvent::VirtualEventType positive,
    const CVirtualGimbalEvent::VirtualEventType negative)
{
    if (!hlsl::isFiniteScalar(denominator) || hlsl::isNearlyZeroScalar(denominator, static_cast<double>(ICamera::TinyScalarEpsilon)))
        return;

    appendSignedVirtualEvent(events, value / denominator, positive, negative, tolerance);
}

inline void appendAngularDeltaEvent(
    std::vector<CVirtualGimbalEvent>& events,
    const double deltaRadians,
    const double denominator,
    const double toleranceDeg,
    const CVirtualGimbalEvent::VirtualEventType positive,
    const CVirtualGimbalEvent::VirtualEventType negative)
{
    if (!hlsl::isFiniteScalar(deltaRadians) ||
        hlsl::isNearlyZeroScalar(hlsl::degrees(deltaRadians), toleranceDeg))
    {
        return;
    }

    appendScaledVirtualEvent(
        events,
        deltaRadians,
        denominator,
        hlsl::radians(toleranceDeg),
        positive,
        negative);
}

inline void appendScaledVirtualAxisEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const hlsl::float64_t3& values,
    const hlsl::float64_t3& denominators,
    const hlsl::float64_t3& tolerances,
    const std::array<SCameraVirtualEventAxisBinding, 3u>& axisBindings)
{
    for (size_t axisIx = 0u; axisIx < axisBindings.size(); ++axisIx)
    {
        appendScaledVirtualEvent(
            events,
            values[axisIx],
            denominators[axisIx],
            tolerances[axisIx],
            axisBindings[axisIx].positive,
            axisBindings[axisIx].negative);
    }
}

inline void appendLocalTranslationEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const hlsl::float64_t3& localDelta,
    const hlsl::float64_t3& denominators = hlsl::float64_t3(1.0),
    const hlsl::float64_t3& tolerances = hlsl::float64_t3(ICamera::TinyScalarEpsilon))
{
    appendScaledVirtualAxisEvents(
        events,
        localDelta,
        denominators,
        tolerances,
        SCameraVirtualEventBindings::LocalTranslation);
}

inline void appendWorldTranslationAsLocalEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const hlsl::camera_quaternion_t<hlsl::float64_t>& orientation,
    const hlsl::float64_t3& worldDelta,
    const hlsl::float64_t3& denominators = hlsl::float64_t3(1.0),
    const hlsl::float64_t3& tolerances = hlsl::float64_t3(ICamera::TinyScalarEpsilon))
{
    appendLocalTranslationEvents(
        events,
        hlsl::projectWorldVectorToLocalQuaternionFrame(orientation, worldDelta),
        denominators,
        tolerances);
}

inline void appendAngularAxisEvents(
    std::vector<CVirtualGimbalEvent>& events,
    const hlsl::float64_t3& deltaRadians,
    const hlsl::float64_t3& denominators,
    const hlsl::float64_t3& toleranceDeg,
    const std::array<SCameraVirtualEventAxisBinding, 3u>& axisBindings)
{
    for (size_t axisIx = 0u; axisIx < axisBindings.size(); ++axisIx)
    {
        appendAngularDeltaEvent(
            events,
            deltaRadians[axisIx],
            denominators[axisIx],
            toleranceDeg[axisIx],
            axisBindings[axisIx].positive,
            axisBindings[axisIx].negative);
    }
}

inline hlsl::float64_t3 collectSignedTranslationDelta(std::span<const CVirtualGimbalEvent> events)
{
    hlsl::float64_t3 delta = hlsl::float64_t3(0.0);
    for (const auto& ev : events)
    {
        switch (ev.type)
        {
            case CVirtualGimbalEvent::MoveRight: delta.x += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveLeft: delta.x -= ev.magnitude; break;
            case CVirtualGimbalEvent::MoveUp: delta.y += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveDown: delta.y -= ev.magnitude; break;
            case CVirtualGimbalEvent::MoveForward: delta.z += ev.magnitude; break;
            case CVirtualGimbalEvent::MoveBackward: delta.z -= ev.magnitude; break;
            default: break;
        }
    }
    return delta;
}

} // namespace nbl::core

#endif // _C_CAMERA_VIRTUAL_EVENT_UTILITIES_HPP_
