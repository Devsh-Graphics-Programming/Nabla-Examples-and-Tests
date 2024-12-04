#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

/////////////////////////
// TODO: TEMPORARY!!!
#include "common.hpp"
namespace ImGuizmo
{
    void DecomposeMatrixToComponents(const float*, float*, float*, float*);
}
/////////////////////////

#include "IProjection.hpp"
#include "IGimbal.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl 
{

struct IGimbalManipulateEncoder
{
    IGimbalManipulateEncoder() {}
    virtual ~IGimbalManipulateEncoder() {}

    //! output of any controller process method
    using gimbal_event_t = CVirtualGimbalEvent;

    //! encode keyboard code used to translate to gimbal_event_t event
    using encode_keyboard_code_t = ui::E_KEY_CODE;

    //! encode mouse code used to translate to gimbal_event_t event
    using encode_mouse_code_t = ui::E_MOUSE_CODE;

    //! encode ImGuizmo code used to translate to gimbal_event_t event
    using encode_imguizmo_code_t = gimbal_event_t::VirtualEventType;

    //! Encoder types, a controller takes encoder type events and outputs gimbal_event_t events
    enum EncoderType : uint8_t
    {
        Keyboard,
        Mouse,
        Imguizmo,

        Count
    };

    //! A key in a hash map which is "encode_<EncoderType>_code_t" as union with information about EncoderType the encode value got produced from
    struct CKeyInfo
    {
        union
        {
            encode_keyboard_code_t keyboardCode;
            encode_mouse_code_t mouseCode;
            encode_imguizmo_code_t imguizmoCode;
        };

        CKeyInfo(encode_keyboard_code_t code) : keyboardCode(code), type(Keyboard) {}
        CKeyInfo(encode_mouse_code_t code) : mouseCode(code), type(Mouse) {}
        CKeyInfo(encode_imguizmo_code_t code) : imguizmoCode(code), type(Imguizmo) {}

        EncoderType type;
    };

    //! Hash value in hash map which is gimbal_event_t & state
    struct CHashInfo
    {
        CHashInfo() {}
        CHashInfo(gimbal_event_t::VirtualEventType _type) : event({ .type = _type }) {}
        ~CHashInfo() = default;

        gimbal_event_t event = {};
        bool active = false;
    };

    using keyboard_to_virtual_events_t = std::unordered_map<encode_keyboard_code_t, CHashInfo>;
    using mouse_to_virtual_events_t = std::unordered_map<encode_mouse_code_t, CHashInfo>;
    using imguizmo_to_virtual_events_t = std::unordered_map<encode_imguizmo_code_t, CHashInfo>;

    //! default preset with encode_keyboard_code_t to gimbal_virtual_event_t map
    virtual const keyboard_to_virtual_events_t& getKeyboardMappingPreset() const = 0u;

    //! default preset with encode_keyboard_code_t to gimbal_virtual_event_t map
    virtual const mouse_to_virtual_events_t& getMouseMappingPreset() const = 0u;

    //! default preset with encode_keyboard_code_t to gimbal_virtual_event_t map
    virtual const imguizmo_to_virtual_events_t& getImguizmoMappingPreset() const = 0u;
};

class IGimbalController : public IGimbalManipulateEncoder
{
public:
    using IGimbalManipulateEncoder::IGimbalManipulateEncoder;

    IGimbalController() {}
    virtual ~IGimbalController() {}

    //! input of keyboard gimbal controller process utility - Nabla UI event handler produces ui::SKeyboardEvent events
    using input_keyboard_event_t = ui::SKeyboardEvent;

    //! input of mouse gimbal controller process utility - Nabla UI event handler produces ui::SMouseEvent events 
    using input_mouse_event_t = ui::SMouseEvent;

    //! input of ImGuizmo gimbal controller process utility - ImGuizmo manipulate utility produces "delta (TRS) matrix" events
    using input_imguizmo_event_t = std::pair<float32_t4x4, float32_t3x3>;

    void beginInputProcessing(const std::chrono::microseconds nextPresentationTimeStamp)
    {
        m_nextPresentationTimeStamp = nextPresentationTimeStamp;
        m_frameDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_nextPresentationTimeStamp - m_lastVirtualUpTimeStamp).count();
        assert(m_frameDeltaTime >= 0.f);
    }

    void endInputProcessing()
    {
        m_lastVirtualUpTimeStamp = m_nextPresentationTimeStamp;
    }

    // Binds mouse key codes to virtual events, the mapKeys lambda will be executed with controller keyboard_to_virtual_events_t table 
    void updateKeyboardMapping(const std::function<void(keyboard_to_virtual_events_t&)>& mapKeys) { mapKeys(m_keyboardVirtualEventMap); }

    // Binds mouse key codes to virtual events, the mapKeys lambda will be executed with controller mouse_to_virtual_events_t table 
    void updateMouseMapping(const std::function<void(mouse_to_virtual_events_t&)>& mapKeys) { mapKeys(m_mouseVirtualEventMap); }

    // Binds imguizmo key codes to virtual events, the mapKeys lambda will be executed with controller imguizmo_to_virtual_events_t table 
    void updateImguizmoMapping(const std::function<void(imguizmo_to_virtual_events_t&)>& mapKeys) { mapKeys(m_imguizmoVirtualEventMap); }

    struct SUpdateParameters
    {
        std::span<const input_keyboard_event_t> keyboardEvents = {};
        std::span<const input_mouse_event_t> mouseEvents = {};
        std::span<const input_imguizmo_event_t> imguizmoEvents = {};
    };

    // Processes SUpdateParameters events to generate virtual manipulation events
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

    inline const keyboard_to_virtual_events_t& getKeyboardVirtualEventMap() { return m_keyboardVirtualEventMap; }
    inline const mouse_to_virtual_events_t& getMouseVirtualEventMap() { return m_mouseVirtualEventMap; }
    inline const imguizmo_to_virtual_events_t& getImguizmoVirtualEventMap() { return m_imguizmoVirtualEventMap; }

private:
    // Processes keyboard events to generate virtual manipulation events
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
                        {
                            const auto keyDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_nextPresentationTimeStamp - keyboardEvent.timeStamp).count();
                            assert(keyDeltaTime >= 0);

                            hash.active = true;
                            hash.event.magnitude = keyDeltaTime;
                        }
                    }
                    else if (keyboardEvent.action == input_keyboard_event_t::ECA_RELEASED)
                        hash.active = false;
                }
            }

            postprocess(m_keyboardVirtualEventMap, output, count);
        }
    }

    // Processes mouse events to generate virtual manipulation events
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
                                {
                                    const auto keyDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(m_nextPresentationTimeStamp - mouseEvent.timeStamp).count();
                                    assert(keyDeltaTime >= 0);

                                    hash.active = true;
                                    hash.event.magnitude += keyDeltaTime;
                                }
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
                // TODO: debug assert "orientationBasis" is orthonormal
                const auto& [deltaWorldTRS, orientationBasis] = ev;

                struct
                {
                    float32_t3 dTranslation, dRotation, dScale;
                } world, local; // its important to notice our imguizmo deltas are written in world base, so I will assume you generate events with respect for input local basis

                // TODO: since I assume the delta matrix is from imguizmo I must follow its API (but this one I could actually steal & rewrite to Nabla to not call imguizmo stuff here <- its TEMP)
                ImGuizmo::DecomposeMatrixToComponents(&deltaWorldTRS[0][0], &world.dTranslation[0], &world.dRotation[0], &world.dScale[0]);

                // FP precision threshold, lets filter small artifacts for this event generator to be more accurate
                constexpr float threshold = 1e-8f;

                auto filterDelta = [](const float32_t3& in) -> float32_t3 
                {
                    return 
                    {
                        (std::abs(in[0]) > threshold) ? in[0] : 0.0f,
                        (std::abs(in[1]) > threshold) ? in[1] : 0.0f,
                        (std::abs(in[2]) > threshold) ? in[2] : 0.0f
                    };
                };

                local.dTranslation = filterDelta(mul(orientationBasis, world.dTranslation));
                
                // TODO
                local.dRotation = { 0.f, 0.f, 0.f };
                // TODO
                local.dScale = { 1.f, 1.f, 1.f };

                // Delta translation impulse
                requestMagnitudeUpdateWithScalar(0.f, local.dTranslation[0], std::abs(local.dTranslation[0]), gimbal_event_t::MoveRight, gimbal_event_t::MoveLeft, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, local.dTranslation[1], std::abs(local.dTranslation[1]), gimbal_event_t::MoveUp, gimbal_event_t::MoveDown, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, local.dTranslation[2], std::abs(local.dTranslation[2]), gimbal_event_t::MoveForward, gimbal_event_t::MoveBackward, m_imguizmoVirtualEventMap);

                // Delta rotation impulse
                requestMagnitudeUpdateWithScalar(0.f, local.dRotation[0], std::abs(local.dRotation[0]), gimbal_event_t::PanRight, gimbal_event_t::PanLeft, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, local.dRotation[1], std::abs(local.dRotation[1]), gimbal_event_t::TiltUp, gimbal_event_t::TiltDown, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(0.f, local.dRotation[2], std::abs(local.dRotation[2]), gimbal_event_t::RollRight, gimbal_event_t::RollLeft, m_imguizmoVirtualEventMap);

                // Delta scale impulse
                requestMagnitudeUpdateWithScalar(1.f, local.dScale[0], std::abs(local.dScale[0]), gimbal_event_t::ScaleXInc, gimbal_event_t::ScaleXDec, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(1.f, local.dScale[1], std::abs(local.dScale[1]), gimbal_event_t::ScaleYInc, gimbal_event_t::ScaleYDec, m_imguizmoVirtualEventMap);
                requestMagnitudeUpdateWithScalar(1.f, local.dScale[2], std::abs(local.dScale[2]), gimbal_event_t::ScaleZInc, gimbal_event_t::ScaleZDec, m_imguizmoVirtualEventMap);
            }

            postprocess(m_imguizmoVirtualEventMap, output, count);
        }
    }

    //! helper utility, for any controller this should be called before any update of hash map
    void preprocess(auto& map)
    {
        for (auto& [key, hash] : map)
        {
            hash.event.magnitude = 0.0f;

            if (hash.active)
                hash.event.magnitude = m_frameDeltaTime;
        }
    }

    //! helper utility, for any controller this should be called after updating a hash map
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

    //! helper utility, it *doesn't* assume we keep requested events alive but only increase their magnitude
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

    keyboard_to_virtual_events_t m_keyboardVirtualEventMap;
    mouse_to_virtual_events_t m_mouseVirtualEventMap;
    imguizmo_to_virtual_events_t m_imguizmoVirtualEventMap;

    size_t m_frameDeltaTime = {};
    std::chrono::microseconds m_nextPresentationTimeStamp = {}, m_lastVirtualUpTimeStamp = {};
};

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_