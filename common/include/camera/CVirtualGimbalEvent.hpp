#ifndef _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_
#define _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_

#include <array>
#include <cstdint>
#include <string_view>

#include "nbl/builtin/hlsl/cpp_compat/vector.hlsl"
#include "nbl/core/math/intutil.h"

namespace nbl::core
{

/// @brief Shared semantic camera command.
///
/// Input processors and scripted tools emit these events.
/// Camera implementations consume them through `ICamera::manipulate(...)`.
struct CVirtualGimbalEvent
{
    enum VirtualEventType : uint32_t
    {
        None = 0,

        MoveForward = core::createBitmask({ 0 }),
        MoveBackward = core::createBitmask({ 1 }),
        MoveLeft = core::createBitmask({ 2 }),
        MoveRight = core::createBitmask({ 3 }),
        MoveUp = core::createBitmask({ 4 }),
        MoveDown = core::createBitmask({ 5 }),
        TiltUp = core::createBitmask({ 6 }),
        TiltDown = core::createBitmask({ 7 }),
        PanLeft = core::createBitmask({ 8 }),
        PanRight = core::createBitmask({ 9 }),
        RollLeft = core::createBitmask({ 10 }),
        RollRight = core::createBitmask({ 11 }),
        ScaleXInc = core::createBitmask({ 12 }),
        ScaleXDec = core::createBitmask({ 13 }),
        ScaleYInc = core::createBitmask({ 14 }),
        ScaleYDec = core::createBitmask({ 15 }),
        ScaleZInc = core::createBitmask({ 16 }),
        ScaleZDec = core::createBitmask({ 17 }),

        EventsCount = 18,

        Translate = MoveForward | MoveBackward | MoveLeft | MoveRight | MoveUp | MoveDown,
        Rotate = TiltUp | TiltDown | PanLeft | PanRight | RollLeft | RollRight,
        Scale = ScaleXInc | ScaleXDec | ScaleYInc | ScaleYDec | ScaleZInc | ScaleZDec,

        All = Translate | Rotate | Scale
    };

    using manipulation_encode_t = hlsl::float64_t;

    VirtualEventType type = None;
    manipulation_encode_t magnitude = {};

    static constexpr std::string_view virtualEventToString(VirtualEventType event)
    {
        switch (event)
        {
            case MoveForward: return "MoveForward";
            case MoveBackward: return "MoveBackward";
            case MoveLeft: return "MoveLeft";
            case MoveRight: return "MoveRight";
            case MoveUp: return "MoveUp";
            case MoveDown: return "MoveDown";
            case TiltUp: return "TiltUp";
            case TiltDown: return "TiltDown";
            case PanLeft: return "PanLeft";
            case PanRight: return "PanRight";
            case RollLeft: return "RollLeft";
            case RollRight: return "RollRight";
            case ScaleXInc: return "ScaleXInc";
            case ScaleXDec: return "ScaleXDec";
            case ScaleYInc: return "ScaleYInc";
            case ScaleYDec: return "ScaleYDec";
            case ScaleZInc: return "ScaleZInc";
            case ScaleZDec: return "ScaleZDec";
            case Translate: return "Translate";
            case Rotate: return "Rotate";
            case Scale: return "Scale";
            case None: return "None";
            default: return "Unknown";
        }
    }

    static constexpr VirtualEventType stringToVirtualEvent(std::string_view event)
    {
        if (event == "MoveForward") return MoveForward;
        if (event == "MoveBackward") return MoveBackward;
        if (event == "MoveLeft") return MoveLeft;
        if (event == "MoveRight") return MoveRight;
        if (event == "MoveUp") return MoveUp;
        if (event == "MoveDown") return MoveDown;
        if (event == "TiltUp") return TiltUp;
        if (event == "TiltDown") return TiltDown;
        if (event == "PanLeft") return PanLeft;
        if (event == "PanRight") return PanRight;
        if (event == "RollLeft") return RollLeft;
        if (event == "RollRight") return RollRight;
        if (event == "ScaleXInc") return ScaleXInc;
        if (event == "ScaleXDec") return ScaleXDec;
        if (event == "ScaleYInc") return ScaleYInc;
        if (event == "ScaleYDec") return ScaleYDec;
        if (event == "ScaleZInc") return ScaleZInc;
        if (event == "ScaleZDec") return ScaleZDec;
        if (event == "Translate") return Translate;
        if (event == "Rotate") return Rotate;
        if (event == "Scale") return Scale;
        if (event == "None") return None;
        return None;
    }

    static constexpr bool isTranslationEvent(const VirtualEventType event)
    {
        return event != None && (event & Translate) == event;
    }

    static constexpr bool isRotationEvent(const VirtualEventType event)
    {
        return event != None && (event & Rotate) == event;
    }

    static constexpr bool isScaleEvent(const VirtualEventType event)
    {
        return event != None && (event & Scale) == event;
    }

    static inline constexpr auto VirtualEventsTypeTable = []()
    {
        std::array<VirtualEventType, EventsCount> output;

        for (uint16_t i = 0u; i < EventsCount; ++i)
            output[i] = static_cast<VirtualEventType>(core::createBitmask({ i }));

        return output;
    }();
};

} // namespace nbl::core

#endif // _NBL_C_VIRTUAL_GIMBAL_EVENT_HPP_
