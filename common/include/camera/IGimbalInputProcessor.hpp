#ifndef _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
#define _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_

#include <algorithm>
#include <array>

#include "nbl/ui/KeyCodes.h"
#include "nbl/ui/SInputEvent.h"

#include "IGimbalBindingLayout.hpp"

namespace nbl::ui
{

/**
* Runtime processor that turns keyboard, mouse, and ImGuizmo input into virtual events.
*/
class IGimbalInputProcessor : public CGimbalBindingLayoutStorage
{
public:
    struct SInputProcessorDefaults final
    {
        static inline constexpr double MaxFrameDeltaMs = 200.0;
        static inline constexpr float ZeroPivot = 0.0f;
        static inline constexpr float UnitPivot = 1.0f;
    };
    static inline constexpr double MaxFrameDeltaMs = SInputProcessorDefaults::MaxFrameDeltaMs;
    static inline constexpr float ZeroPivot = SInputProcessorDefaults::ZeroPivot;
    static inline constexpr float UnitPivot = SInputProcessorDefaults::UnitPivot;

    using CGimbalBindingLayoutStorage::CGimbalBindingLayoutStorage;

    IGimbalInputProcessor() = default;
    virtual ~IGimbalInputProcessor() = default;

    //! Keyboard events consumed by the processor.
    using input_keyboard_event_t = ui::SKeyboardEvent;

    //! Mouse events consumed by the processor.
    using input_mouse_event_t = ui::SMouseEvent;

    //! ImGuizmo world-space delta transforms consumed by the processor.
    using input_imguizmo_event_t = hlsl::float32_t4x4;

    void beginInputProcessing(const std::chrono::microseconds nextPresentationTimeStamp)
    {
        m_nextPresentationTimeStamp = nextPresentationTimeStamp;
        m_frameDeltaTime = clampFrameDeltaTimeMs(m_nextPresentationTimeStamp, m_lastVirtualUpTimeStamp);
    }

    void endInputProcessing()
    {
        m_lastVirtualUpTimeStamp = m_nextPresentationTimeStamp;
    }

    struct SUpdateParameters
    {
        std::span<const input_keyboard_event_t> keyboardEvents = {};
        std::span<const input_mouse_event_t> mouseEvents = {};
        std::span<const input_imguizmo_event_t> imguizmoEvents = {};
    };

    /**
    * @brief Processes combined events from SUpdateParameters to generate virtual manipulation events.
    *
    * @note This function combines the processing of events from keyboards, mouse and ImGuizmo.
    * It delegates the actual processing to the respective functions:
    * - @ref processKeyboard for keyboard events
    * - @ref processMouse for mouse events
    * - @ref processImguizmo for ImGuizmo events
    * The results are accumulated into the output array and the total count.
    *
    * @param "output" is a pointer to the array where all generated gimbal events will be stored.
    * If nullptr, the function will only calculate the total count of potential
    * output events without processing.
    *
    * @param "count" is a uint32_t reference to store the total count of generated gimbal events.
    *
    * @param "parameters" is an SUpdateParameters structure containing the individual event arrays
    * for keyboard, mouse, and ImGuizmo inputs.
    *
    * @return void. If "count" > 0 and "output" is a valid pointer, use it to dereference your "output"
    * containing "count" events. If "output" is nullptr, "count" tells you the total size of "output"
    * you must guarantee to be valid.
    */
    void process(gimbal_event_t* output, uint32_t& count, const SUpdateParameters parameters = {})
    {
        count = 0u;
        uint32_t vKeyboardEventsCount = {}, vMouseEventsCount = {}, vImguizmoEventsCount = {};

        if (output)
        {
            processKeyboard(output, vKeyboardEventsCount, parameters.keyboardEvents); output += vKeyboardEventsCount;
            processMouse(output, vMouseEventsCount, parameters.mouseEvents); output += vMouseEventsCount;
            processImguizmo(output, vImguizmoEventsCount, parameters.imguizmoEvents);
        }
        else
        {
            processKeyboard(nullptr, vKeyboardEventsCount, {});
            processMouse(nullptr, vMouseEventsCount, {});
            processImguizmo(nullptr, vImguizmoEventsCount, {});
        }

        count = vKeyboardEventsCount + vMouseEventsCount + vImguizmoEventsCount;
    }

    /**
    * @brief Processes keyboard events to generate virtual manipulation events.
    *
    * @note This function maps keyboard events into virtual gimbal manipulation events
    * based on predefined mappings. It supports event types such as key press and key release
    * to trigger corresponding actions.
    *
    * @param "output" is a pointer to the array where generated gimbal events will be stored.
    * If nullptr, the function will only calculate the count of potential
    * output events without processing.
    *
    * @param "count" is a uint32_t reference to store the count of generated gimbal events.
    *
    * @param "events" is a span of input_keyboard_event_t. Each such event contains a key code and action,
    * such as key press or release.
    *
    * @return void. If "count" > 0 and "output" is a valid pointer, use it to dereference your "output"
    * containing "count" events. If "output" is nullptr, "count" tells you the size of "output" you must guarantee to be valid.
    */
    void processKeyboard(gimbal_event_t* output, uint32_t& count, std::span<const input_keyboard_event_t> events)
    {
        processBindingMap(
            m_keyboardVirtualEventMap,
            output,
            count,
            [&](auto& map)
            {
                for (const auto& keyboardEvent : events)
                {
                    if (keyboardEvent.action == input_keyboard_event_t::ECA_PRESSED)
                        setBindingActiveState(map, keyboardEvent.keyCode, true);
                    else if (keyboardEvent.action == input_keyboard_event_t::ECA_RELEASED)
                        setBindingActiveState(map, keyboardEvent.keyCode, false);
                }
            });
    }

    /**
    * @brief Processes mouse events to generate virtual manipulation events.
    *
    * @note This function processes mouse input, including clicks, scrolls, and movements,
    * and maps them into virtual gimbal manipulation events. Mouse actions are processed
    * using predefined mappings to determine corresponding gimbal manipulations.
    *
    * @param "output" is a pointer to the array where generated gimbal events will be stored.
    * If nullptr, the function will only calculate the count of potential
    * output events without processing.
    *
    * @param "count" is a uint32_t reference to store the count of generated gimbal events.
    *
    * @param "events" is a span of input_mouse_event_t. Each such event represents a mouse action,
    * including clicks, scrolls, or movements.
    *
    * @return void. If "count" > 0 and "output" is a valid pointer, use it to dereference your "output"
    * containing "count" events. If "output" is nullptr, "count" tells you the size of "output" you must guarantee to be valid.
    */
    void processMouse(gimbal_event_t* output, uint32_t& count, std::span<const input_mouse_event_t> events)
    {
        processBindingMap(
            m_mouseVirtualEventMap,
            output,
            count,
            [&](auto& map)
            {
                for (const auto& mouseEvent : events)
                {
                    switch (mouseEvent.type)
                    {
                        case input_mouse_event_t::EET_CLICK:
                            updateMouseButtonState(map, mouseEvent.clickEvent);
                            break;

                        case input_mouse_event_t::EET_SCROLL:
                            requestMagnitudeUpdateWithSignedComponents(
                                ZeroPivot,
                                hlsl::float32_t2(
                                    static_cast<float>(mouseEvent.scrollEvent.verticalScroll),
                                    mouseEvent.scrollEvent.horizontalScroll),
                                SInputProcessorBindingGroups::MouseScroll,
                                map);
                            break;

                        case input_mouse_event_t::EET_MOVEMENT:
                            requestMagnitudeUpdateWithSignedComponents(
                                ZeroPivot,
                                hlsl::float32_t2(
                                    mouseEvent.movementEvent.relativeMovementX,
                                    mouseEvent.movementEvent.relativeMovementY),
                                SInputProcessorBindingGroups::MouseRelativeMovement,
                                map);
                            break;

                        default:
                            break;
                    }
                }
            });
    }

    /**
    * @brief Processes input events from ImGuizmo and generates virtual gimbal events.
    *
    * @note This function processes world-space delta transforms authored by ImGuizmo and converts
    * them into virtual gimbal events for world-space camera manipulation.
    * The function computes translation, rotation, and scale deltas from each transform matrix,
    * which are then mapped to corresponding virtual events using a predefined mapping.
    *
    * @param "output" is pointer to the array where generated gimbal events will be stored.
    * If nullptr, the function will only calculate the count of potential
    * output events without processing.
    * 
    * @param "count" is uint32_t reference to store the count of generated gimbal events.
    * 
    * @param "events" is a span of input_imguizmo_event_t. Each such event contains a delta
    * transformation matrix that represents changes in world space.
    * 
    * @return void. If "count" > 0 & "output" was valid pointer then use it to dereference your "output" containing "count" events. 
    * If "output" is nullptr then "count" tells you about size of "output" you must guarantee to be valid.
    */
    void processImguizmo(gimbal_event_t* output, uint32_t& count, std::span<const input_imguizmo_event_t> events)
    {
        processBindingMap(
            m_imguizmoVirtualEventMap,
            output,
            count,
            [&](auto& map)
            {
                for (const auto& ev : events)
                {
                    const auto& deltaWorldTRS = ev;

                    hlsl::SRigidTransformComponents<hlsl::float32_t> world = {};
                    if (!hlsl::tryExtractRigidTransformComponents(deltaWorldTRS, world))
                        continue;

                    requestMagnitudeUpdateWithSignedComponents(
                        ZeroPivot,
                        world.translation,
                        SInputProcessorBindingGroups::ImguizmoTranslation,
                        map);

                    const auto dRotationRad = hlsl::getCameraOrientationEulerRadians(world.orientation);
                    requestMagnitudeUpdateWithSignedComponents(
                        ZeroPivot,
                        dRotationRad,
                        SInputProcessorBindingGroups::ImguizmoRotation,
                        map);

                    requestMagnitudeUpdateWithSignedComponents(
                        UnitPivot,
                        world.scale,
                        SInputProcessorBindingGroups::ImguizmoScale,
                        map);
                }
            });
    }

private:
    template<typename EncodeType, uint32_t N>
    struct SEncodedAxisBindingGroup final
    {
        std::array<EncodeType, N> positive = {};
        std::array<EncodeType, N> negative = {};
    };

    struct SInputProcessorBindingGroups final
    {
        static inline constexpr SEncodedAxisBindingGroup<ui::E_MOUSE_CODE, 2u> MouseScroll = {
            .positive = {
                ui::EMC_VERTICAL_POSITIVE_SCROLL,
                ui::EMC_HORIZONTAL_POSITIVE_SCROLL
            },
            .negative = {
                ui::EMC_VERTICAL_NEGATIVE_SCROLL,
                ui::EMC_HORIZONTAL_NEGATIVE_SCROLL
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<ui::E_MOUSE_CODE, 2u> MouseRelativeMovement = {
            .positive = {
                ui::EMC_RELATIVE_POSITIVE_MOVEMENT_X,
                ui::EMC_RELATIVE_POSITIVE_MOVEMENT_Y
            },
            .negative = {
                ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_X,
                ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoTranslation = {
            .positive = {
                gimbal_event_t::MoveRight,
                gimbal_event_t::MoveUp,
                gimbal_event_t::MoveForward
            },
            .negative = {
                gimbal_event_t::MoveLeft,
                gimbal_event_t::MoveDown,
                gimbal_event_t::MoveBackward
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoRotation = {
            .positive = {
                gimbal_event_t::TiltUp,
                gimbal_event_t::PanRight,
                gimbal_event_t::RollRight
            },
            .negative = {
                gimbal_event_t::TiltDown,
                gimbal_event_t::PanLeft,
                gimbal_event_t::RollLeft
            }
        };

        static inline constexpr SEncodedAxisBindingGroup<gimbal_event_t::VirtualEventType, 3u> ImguizmoScale = {
            .positive = {
                gimbal_event_t::ScaleXInc,
                gimbal_event_t::ScaleYInc,
                gimbal_event_t::ScaleZInc
            },
            .negative = {
                gimbal_event_t::ScaleXDec,
                gimbal_event_t::ScaleYDec,
                gimbal_event_t::ScaleZDec
            }
        };
    };

    static double clampFrameDeltaTimeMs(
        const std::chrono::microseconds nextPresentationTimeStamp,
        const std::chrono::microseconds lastVirtualUpTimeStamp)
    {
        const auto deltaMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
        if (deltaMs < 0)
            return 0.0;
        return std::min(static_cast<double>(deltaMs), MaxFrameDeltaMs);
    }

    template<typename Map, typename ConsumeFn>
    void processBindingMap(Map& map, gimbal_event_t* output, uint32_t& count, ConsumeFn&& consume)
    {
        count = 0u;
        const auto mappedVirtualEventsCount = static_cast<uint32_t>(map.size());
        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }
        if (!mappedVirtualEventsCount)
            return;

        preprocess(map);
        consume(map);
        postprocess(map, output, count);
    }

    static bool tryGetMouseButtonCode(
        const ui::E_MOUSE_BUTTON button,
        ui::E_MOUSE_CODE& outCode)
    {
        switch (button)
        {
            case ui::EMB_LEFT_BUTTON:    outCode = ui::EMC_LEFT_BUTTON; return true;
            case ui::EMB_RIGHT_BUTTON:   outCode = ui::EMC_RIGHT_BUTTON; return true;
            case ui::EMB_MIDDLE_BUTTON:  outCode = ui::EMC_MIDDLE_BUTTON; return true;
            case ui::EMB_BUTTON_4:       outCode = ui::EMC_BUTTON_4; return true;
            case ui::EMB_BUTTON_5:       outCode = ui::EMC_BUTTON_5; return true;
            default:
                return false;
        }
    }

    template<typename Map>
    void updateMouseButtonState(Map& map, const input_mouse_event_t::SClickEvent& clickEvent)
    {
        ui::E_MOUSE_CODE mouseCode = ui::EMC_NONE;
        if (!tryGetMouseButtonCode(clickEvent.mouseButton, mouseCode))
            return;

        if (clickEvent.action == input_mouse_event_t::SClickEvent::EA_PRESSED)
            setBindingActiveState(map, mouseCode, true);
        else if (clickEvent.action == input_mouse_event_t::SClickEvent::EA_RELEASED)
            setBindingActiveState(map, mouseCode, false);
    }

    template<typename Code, typename Map>
    void setBindingActiveState(Map& map, const Code code, const bool active)
    {
        const auto request = map.find(code);
        if (request == map.end())
            return;

        request->second.active = active;
    }

    void preprocess(auto& map)
    {
        for (auto& [key, hash] : map)
        {
            hash.event.magnitude = 0.0f;

            if (hash.active)
                hash.event.magnitude = m_frameDeltaTime;
        }
    }

    void postprocess(const auto& map, gimbal_event_t* output, uint32_t& count)
    {
        for (const auto& [key, hash] : map)
            if (hash.event.magnitude)
            {
                auto* virtualEvent = output + count;
                virtualEvent->type = hash.event.type;
                virtualEvent->magnitude = hash.event.magnitude;
                ++count;
            }
    }

    template <typename EncodeType, typename Map>
    void requestMagnitudeUpdateWithScalar(float signPivot, float dScalar, EncodeType positive, EncodeType negative, Map& map)
    {
        if (dScalar != signPivot)
        {
            const auto dMagnitude = hlsl::abs(dScalar);
            auto code = (dScalar > signPivot) ? positive : negative;
            auto request = map.find(code);
            if (request != map.end())
                request->second.event.magnitude += dMagnitude;
        }
    }

    template <typename EncodeType, typename Map, uint32_t N>
    void requestMagnitudeUpdateWithSignedComponents(
        float signPivot,
        const hlsl::vector<float, N>& components,
        const std::array<EncodeType, N>& positive,
        const std::array<EncodeType, N>& negative,
        Map& map)
    {
        for (uint32_t i = 0u; i < N; ++i)
            requestMagnitudeUpdateWithScalar(signPivot, components[i], positive[i], negative[i], map);
    }

    template <typename EncodeType, typename Map, uint32_t N>
    void requestMagnitudeUpdateWithSignedComponents(
        float signPivot,
        const hlsl::vector<float, N>& components,
        const SEncodedAxisBindingGroup<EncodeType, N>& bindings,
        Map& map)
    {
        requestMagnitudeUpdateWithSignedComponents(
            signPivot,
            components,
            bindings.positive,
            bindings.negative,
            map);
    }

    double m_frameDeltaTime = {};
    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // namespace nbl::ui

#endif // _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
