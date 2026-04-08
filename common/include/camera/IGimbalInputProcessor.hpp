#ifndef _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
#define _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_

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
    static inline constexpr double MaxFrameDeltaMs = 200.0;

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
        const auto deltaMs = std::chrono::duration_cast<std::chrono::milliseconds>(m_nextPresentationTimeStamp - m_lastVirtualUpTimeStamp).count();
        if (deltaMs < 0)
            m_frameDeltaTime = 0.0;
        else if (static_cast<double>(deltaMs) > MaxFrameDeltaMs)
            m_frameDeltaTime = MaxFrameDeltaMs;
        else
            m_frameDeltaTime = static_cast<double>(deltaMs);
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
        count = 0u;
        const auto mappedVirtualEventsCount = m_keyboardVirtualEventMap.size();

        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }

        if (mappedVirtualEventsCount)
        {
            preprocess(m_keyboardVirtualEventMap);

            for (const auto& keyboardEvent : events)
            {
                auto request = m_keyboardVirtualEventMap.find(keyboardEvent.keyCode);
                if (request != std::end(m_keyboardVirtualEventMap))
                {
                    auto& hash = request->second;

                    if (keyboardEvent.action == input_keyboard_event_t::ECA_PRESSED)
                    {
                        if (!hash.active)
                            hash.active = true;
                    }
                    else if (keyboardEvent.action == input_keyboard_event_t::ECA_RELEASED)
                        hash.active = false;
                }
            }

            postprocess(m_keyboardVirtualEventMap, output, count);
        }
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
        count = 0u;
        const auto mappedVirtualEventsCount = m_mouseVirtualEventMap.size();

        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }

        if (mappedVirtualEventsCount)
        {
            preprocess(m_mouseVirtualEventMap);

            for (const auto& mouseEvent : events)
            {
                ui::E_MOUSE_CODE mouseCode = ui::EMC_NONE;

                switch (mouseEvent.type)
                {
                    case input_mouse_event_t::EET_CLICK:
                    {
                        switch (mouseEvent.clickEvent.mouseButton)
                        {
                            case ui::EMB_LEFT_BUTTON:    mouseCode = ui::EMC_LEFT_BUTTON; break;
                            case ui::EMB_RIGHT_BUTTON:   mouseCode = ui::EMC_RIGHT_BUTTON; break;
                            case ui::EMB_MIDDLE_BUTTON:  mouseCode = ui::EMC_MIDDLE_BUTTON; break;
                            case ui::EMB_BUTTON_4:       mouseCode = ui::EMC_BUTTON_4; break;
                            case ui::EMB_BUTTON_5:       mouseCode = ui::EMC_BUTTON_5; break;
                            default: continue;
                        }

                        auto request = m_mouseVirtualEventMap.find(mouseCode);
                        if (request != std::end(m_mouseVirtualEventMap))
                        {
                            auto& hash = request->second;

                            if (mouseEvent.clickEvent.action == input_mouse_event_t::SClickEvent::EA_PRESSED)
                            {
                                if (!hash.active)
                                    hash.active = true;
                            }
                            else if (mouseEvent.clickEvent.action == input_mouse_event_t::SClickEvent::EA_RELEASED)
                                hash.active = false;
                        }
                    } break;

                    case input_mouse_event_t::EET_SCROLL:
                    {
                        requestMagnitudeUpdateWithScalar(0.f, float(mouseEvent.scrollEvent.verticalScroll), float(std::abs(mouseEvent.scrollEvent.verticalScroll)), ui::EMC_VERTICAL_POSITIVE_SCROLL, ui::EMC_VERTICAL_NEGATIVE_SCROLL, m_mouseVirtualEventMap);
                        requestMagnitudeUpdateWithScalar(0.f, mouseEvent.scrollEvent.horizontalScroll, std::abs(mouseEvent.scrollEvent.horizontalScroll), ui::EMC_HORIZONTAL_POSITIVE_SCROLL, ui::EMC_HORIZONTAL_NEGATIVE_SCROLL, m_mouseVirtualEventMap);
                    } break;

                    case input_mouse_event_t::EET_MOVEMENT:
                    {
                        requestMagnitudeUpdateWithScalar(0.f, mouseEvent.movementEvent.relativeMovementX, std::abs(mouseEvent.movementEvent.relativeMovementX), ui::EMC_RELATIVE_POSITIVE_MOVEMENT_X, ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_X, m_mouseVirtualEventMap);
                        requestMagnitudeUpdateWithScalar(0.f, mouseEvent.movementEvent.relativeMovementY, std::abs(mouseEvent.movementEvent.relativeMovementY), ui::EMC_RELATIVE_POSITIVE_MOVEMENT_Y, ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y, m_mouseVirtualEventMap);
                    } break;

                    default:
                        break;
                }
            }

            postprocess(m_mouseVirtualEventMap, output, count);
        }
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
        count = 0u;
        const auto mappedVirtualEventsCount = m_imguizmoVirtualEventMap.size();

        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }

        if (mappedVirtualEventsCount)
        {
            preprocess(m_imguizmoVirtualEventMap);

            for (const auto& ev : events)
            {
                const auto& deltaWorldTRS = ev;

                hlsl::SRigidTransformComponents<hlsl::float32_t> world = {};
                if (!hlsl::tryExtractRigidTransformComponents(deltaWorldTRS, world))
                    continue;

                // Delta translation impulse
                requestMagnitudeUpdateWithScalar(0.f, world.translation[0], std::abs(world.translation[0]), gimbal_event_t::MoveRight, gimbal_event_t::MoveLeft, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, world.translation[1], std::abs(world.translation[1]), gimbal_event_t::MoveUp, gimbal_event_t::MoveDown, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, world.translation[2], std::abs(world.translation[2]), gimbal_event_t::MoveForward, gimbal_event_t::MoveBackward, m_imguizmoVirtualEventMap);

                // Delta rotation impulse
                const auto dRotationRad = hlsl::getQuaternionEulerRadians(world.orientation);
                requestMagnitudeUpdateWithScalar(0.f, dRotationRad[0], std::abs(dRotationRad[0]), gimbal_event_t::TiltUp , gimbal_event_t::TiltDown, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, dRotationRad[1], std::abs(dRotationRad[1]), gimbal_event_t::PanRight, gimbal_event_t::PanLeft, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, dRotationRad[2], std::abs(dRotationRad[2]), gimbal_event_t::RollRight, gimbal_event_t::RollLeft, m_imguizmoVirtualEventMap);

                // Delta scale impulse
                requestMagnitudeUpdateWithScalar(1.f, world.scale[0], std::abs(world.scale[0]), gimbal_event_t::ScaleXInc, gimbal_event_t::ScaleXDec, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(1.f, world.scale[1], std::abs(world.scale[1]), gimbal_event_t::ScaleYInc, gimbal_event_t::ScaleYDec, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(1.f, world.scale[2], std::abs(world.scale[2]), gimbal_event_t::ScaleZInc, gimbal_event_t::ScaleZDec, m_imguizmoVirtualEventMap);
            }

            postprocess(m_imguizmoVirtualEventMap, output, count);
        }
    }

private:

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
    void requestMagnitudeUpdateWithScalar(float signPivot, float dScalar, float dMagnitude, EncodeType positive, EncodeType negative, Map& map)
    {
        if (dScalar != signPivot)
        {
            auto code = (dScalar > signPivot) ? positive : negative;
            auto request = map.find(code);
            if (request != map.end())
                request->second.event.magnitude += dMagnitude;
        }
    }

    double m_frameDeltaTime = {};
    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // namespace nbl::ui

#endif // _NBL_I_GIMBAL_INPUT_PROCESSOR_HPP_
