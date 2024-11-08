#ifndef _NBL_C_CAMERA_CONTROLLER_HPP_
#define _NBL_C_CAMERA_CONTROLLER_HPP_

#include "ICameraController.hpp"
#include "ICamera.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl 
{

template<typename T>
class CCameraController : public ICameraController
{
public:
    using ICameraController::ICameraController;
    using interface_camera_t = ICamera<T>;

    using keyboard_to_virtual_events_t = std::unordered_map<ui::E_KEY_CODE, CHashInfo>;
    using mouse_to_virtual_events_t = std::unordered_map<ui::E_MOUSE_CODE, CHashInfo>;

    CCameraController(core::smart_refctd_ptr<interface_camera_t> camera) : m_camera(core::smart_refctd_ptr(camera)) {}

    // Binds mouse key codes to virtual events, the mapKeys lambda will be executed with controller mouse_to_virtual_events_t table 
    void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys)
    {
        mapKeys(m_keyboardVirtualEventMap);
    }

    // Binds mouse key codes to virtual events, the mapKeys lambda will be executed with controller mouse_to_virtual_events_t table 
    void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys)
    {
        mapKeys(m_mouseVirtualEventMap);
    }

    struct SUpdateParameters
    {
        std::span<const ui::SKeyboardEvent> keyboardEvents = {};
        std::span<const ui::SMouseEvent> mouseEvents = {};
    };

    // Processes SUpdateParameters events to generate virtual manipulation events
    void process(CVirtualGimbalEvent* output, uint32_t& count, const SUpdateParameters parameters = {})
    {
        count = 0u;
        uint32_t vKeyboardEventsCount, vMouseEventsCount;

        if (output)
        {
            processKeyboard(output, vKeyboardEventsCount, parameters.keyboardEvents);
            processMouse(output + vKeyboardEventsCount, vMouseEventsCount, parameters.mouseEvents);
        }
        else
        {
            processKeyboard(nullptr, vKeyboardEventsCount, {});
            processMouse(nullptr, vMouseEventsCount, {});
        }

        count = vKeyboardEventsCount + vMouseEventsCount;
    }

    inline const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() { return m_keyboardVirtualEventMap; }
    inline const mouse_to_virtual_events_t& getMouseVirtualEventMap() { return m_mouseVirtualEventMap; }
    inline interface_camera_t* getCamera() { return m_camera.get(); }

private:
    // Processes keyboard events to generate virtual manipulation events
    void processKeyboard(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SKeyboardEvent> events)
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
            const auto frameDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
            assert(frameDeltaTime >= 0.f);

            for (auto& [key, hash] : m_keyboardVirtualEventMap)
            {
                hash.event.magnitude = 0.f;

                /*
                    if a key was already being held down from previous frames we compute with this
                    assumption that the key will be held down for this whole frame as well and its
                    delta action is simply frame delta time
                */

                if (hash.active)
                    hash.event.magnitude = frameDeltaTime;
            }

            for (const auto& keyboardEvent : events)
            {
                auto request = m_keyboardVirtualEventMap.find(keyboardEvent.keyCode);
                if (request != std::end(m_keyboardVirtualEventMap))
                {
                    auto& hash = request->second;

                    if (keyboardEvent.action == nbl::ui::SKeyboardEvent::ECA_PRESSED)
                    {
                        if (!hash.active)
                        {
                            // and if a key has been pressed after its last release event then first delta action is the key delta
                            const auto keyDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - keyboardEvent.timeStamp).count();
                            assert(keyDeltaTime >= 0);

                            hash.active = true;
                            hash.event.magnitude = keyDeltaTime;
                        }
                    }
                    else if (keyboardEvent.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
                        hash.active = false;
                }
            }

            for (const auto& [key, hash] : m_keyboardVirtualEventMap)
            {
                if (hash.event.magnitude)
                {
                    auto* virtualEvent = output + count;
                    virtualEvent->type = hash.event.type;
                    virtualEvent->magnitude = hash.event.magnitude;
                    ++count;
                }
            }
        }
    }

    // Processes mouse events to generate virtual manipulation events
    void processMouse(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SMouseEvent> events)
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
            const auto frameDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
            assert(frameDeltaTime >= 0.f);

            for (auto& [key, hash] : m_mouseVirtualEventMap)
            {
                hash.event.magnitude = 0.f;
                if (hash.active)
                    hash.event.magnitude = frameDeltaTime;
            }

            for (const auto& mouseEvent : events)
            {
                ui::E_MOUSE_CODE mouseCode = ui::EMC_NONE;

                switch (mouseEvent.type)
                {
                    case ui::SMouseEvent::EET_CLICK:
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

                            if (mouseEvent.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_PRESSED)
                            {
                                if (!hash.active)
                                {
                                    const auto keyDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - mouseEvent.timeStamp).count();
                                    assert(keyDeltaTime >= 0);

                                    hash.active = true;
                                    hash.event.magnitude += keyDeltaTime;
                                }
                            }
                            else if (mouseEvent.clickEvent.action == ui::SMouseEvent::SClickEvent::EA_RELEASED)
                                hash.active = false;
                        }
                    } break;

                    case ui::SMouseEvent::EET_SCROLL:
                    {
                        if (mouseEvent.scrollEvent.verticalScroll != 0)
                        {
                            mouseCode = (mouseEvent.scrollEvent.verticalScroll > 0) ? ui::EMC_VERTICAL_POSITIVE_SCROLL : ui::EMC_VERTICAL_NEGATIVE_SCROLL;
                            auto request = m_mouseVirtualEventMap.find(mouseCode);
                            if (request != std::end(m_mouseVirtualEventMap))
                                request->second.event.magnitude += std::abs(mouseEvent.scrollEvent.verticalScroll);
                        }

                        if (mouseEvent.scrollEvent.horizontalScroll != 0)
                        {
                            mouseCode = (mouseEvent.scrollEvent.horizontalScroll > 0) ? ui::EMC_HORIZONTAL_POSITIVE_SCROLL : ui::EMC_HORIZONTAL_NEGATIVE_SCROLL;
                            auto request = m_mouseVirtualEventMap.find(mouseCode);
                            if (request != std::end(m_mouseVirtualEventMap))
                                request->second.event.magnitude += std::abs(mouseEvent.scrollEvent.horizontalScroll);
                        }
                    } break;

                    case ui::SMouseEvent::EET_MOVEMENT:
                    {
                        if (mouseEvent.movementEvent.relativeMovementX != 0)
                        {
                            mouseCode = (mouseEvent.movementEvent.relativeMovementX > 0) ? ui::EMC_RELATIVE_POSITIVE_MOVEMENT_X : ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_X;
                            auto request = m_mouseVirtualEventMap.find(mouseCode);
                            if (request != std::end(m_mouseVirtualEventMap))
                                request->second.event.magnitude += std::abs(mouseEvent.movementEvent.relativeMovementX);
                        }

                        if (mouseEvent.movementEvent.relativeMovementY != 0)
                        {
                            mouseCode = (mouseEvent.movementEvent.relativeMovementY > 0) ? ui::EMC_RELATIVE_POSITIVE_MOVEMENT_Y : ui::EMC_RELATIVE_NEGATIVE_MOVEMENT_Y;
                            auto request = m_mouseVirtualEventMap.find(mouseCode);
                            if (request != std::end(m_mouseVirtualEventMap))
                                request->second.event.magnitude += std::abs(mouseEvent.movementEvent.relativeMovementY);
                        }
                    } break;

                    default:
                        break;
                }
            }

            for (const auto& [key, hash] : m_mouseVirtualEventMap)
            {
                if (hash.event.magnitude)
                {
                    auto* virtualEvent = output + count;
                    virtualEvent->type = hash.event.type;
                    virtualEvent->magnitude = hash.event.magnitude;
                    ++count;
                }
            }
        }
    }

    core::smart_refctd_ptr<interface_camera_t> m_camera;
    keyboard_to_virtual_events_t m_keyboardVirtualEventMap;
    mouse_to_virtual_events_t m_mouseVirtualEventMap;
};

} // nbl::hlsl namespace

#endif // _NBL_C_CAMERA_CONTROLLER_HPP_