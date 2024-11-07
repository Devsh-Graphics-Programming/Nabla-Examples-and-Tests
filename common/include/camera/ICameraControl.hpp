#ifndef _NBL_I_CAMERA_CONTROLLER_HPP_
#define _NBL_I_CAMERA_CONTROLLER_HPP_

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include "IProjection.hpp"
#include "CGeneralPurposeGimbal.hpp"

// TODO: DIFFERENT NAMESPACE
namespace nbl::hlsl 
{

template<typename T>
class ICameraController : virtual public core::IReferenceCounted
{
public:
    using precision_t = typename T;

    struct CRequestInfo
    {
        CRequestInfo() {}
        CRequestInfo(CVirtualGimbalEvent::VirtualEventType _type) : type(_type) {}
        ~CRequestInfo() = default;

        CVirtualGimbalEvent::VirtualEventType type = CVirtualGimbalEvent::VirtualEventType::None;
        bool active = false;
        float64_t dtAction = {};
    };

    using keys_to_virtual_events_t = std::unordered_map<ui::E_KEY_CODE, CRequestInfo>;

    // Gimbal with view parameters representing a camera in world space
    class CGimbal : public IGimbal<precision_t>
    {
    public:
        using base_t = IGimbal<precision_t>;

        CGimbal(typename base_t::SCreationParameters&& parameters) : base_t(std::move(parameters)) {}
        ~CGimbal() = default;

        struct SView
        {
            matrix<precision_t, 3, 4> matrix = {};
            bool isLeftHandSystem = true;
        };

        inline void updateView()
        {
            if (base_t::getManipulationCounter())
            {
                const auto& gRight = base_t::getXAxis(), gUp = base_t::getYAxis(), gForward = base_t::getZAxis();

                // TODO: I think this should be set as request state, depending on the value we do m_view.matrix[2u] flip accordingly
                // m_view.isLeftHandSystem;

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

                assert(isOrthoBase(gRight, gUp, gForward));

                const auto& position = base_t::getPosition();
                m_view.matrix[0u] = vector<precision_t, 4u>(gRight, -glm::dot(gRight, position));
                m_view.matrix[1u] = vector<precision_t, 4u>(gUp, -glm::dot(gUp, position));
                m_view.matrix[2u] = vector<precision_t, 4u>(gForward, -glm::dot(gForward, position));
            }
        }

        // Getter for gimbal's view
        inline const SView& getView() const { return m_view; }

    private:
        SView m_view;
    };

    ICameraController() {}

    // Binds key codes to virtual events, the mapKeys lambda will be executed with controller keys_to_virtual_events_t table 
    void updateKeysToEvent(const std::function<void(keys_to_virtual_events_t&)>& mapKeys)
    {
        mapKeys(m_keysToVirtualEvents);
    }

    // Manipulates camera with view gimbal & virtual events
    virtual void manipulate(std::span<const CVirtualGimbalEvent> virtualEvents) = 0;

    // TODO: *maybe* would be good to have a class interface for virtual event generators,
    // eg keyboard, mouse but maybe custom stuff too eg events from gimbal drag & drop

    void beginInputProcessing(const std::chrono::microseconds _nextPresentationTimeStamp)
    {
        nextPresentationTimeStamp = _nextPresentationTimeStamp;
    }

    // Processes keyboard events to generate virtual manipulation events, note that it doesn't make the manipulation itself!
    void processKeyboard(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SKeyboardEvent> events)
    {
        count = 0u;
        const auto mappedVirtualEventsCount = m_keysToVirtualEvents.size();

        if (!output)
        {
            count = mappedVirtualEventsCount;
            return;
        }

        if (mappedVirtualEventsCount)
        {
            const auto frameDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - lastVirtualUpTimeStamp).count();
            assert(frameDeltaTime >= 0.f);

            for (auto& [key, info] : m_keysToVirtualEvents)
            {
                info.dtAction = 0.f;

                /*
                    if a key was already being held down from previous frames we compute with this 
                    assumption that the key will be held down for this whole frame as well and its
                    delta action is simply frame delta time
                */

                if (info.active)
                    info.dtAction = frameDeltaTime;
            }

            for (const auto& keyboardEvent : events)
            {
                auto request = m_keysToVirtualEvents.find(keyboardEvent.keyCode);
                if (request != std::end(m_keysToVirtualEvents))
                {
                    auto& info = request->second;

                    if (keyboardEvent.action == nbl::ui::SKeyboardEvent::ECA_PRESSED)
                    {
                        if (!info.active)
                        {
                            // and if a key has been pressed after its last release event then first delta action is the key delta
                            const auto keyDeltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(nextPresentationTimeStamp - keyboardEvent.timeStamp).count();
                            assert(keyDeltaTime >= 0);

                            info.active = true;
                            info.dtAction = keyDeltaTime;
                        }
                    }
                    else if (keyboardEvent.action == nbl::ui::SKeyboardEvent::ECA_RELEASED)
                        info.active = false;
                }
            }

            for (const auto& [key, info] : m_keysToVirtualEvents)
            {
                if (info.active)
                {
                    auto* virtualEvent = output + count;
                    virtualEvent->type = info.type;
                    virtualEvent->magnitude = info.dtAction;
                    ++count;
                }
            }
        }
    }

    // Processes mouse events to generate virtual manipulation events, note that it doesn't make the manipulation itself!
    // Limited to Pan & Tilt rotation events, camera type implements how event magnitudes should be interpreted
    void processMouse(CVirtualGimbalEvent* output, uint32_t& count, std::span<const ui::SMouseEvent> events)
    {
        count = 0u;

        if (!output)
        {
            count = 2u;
            return;
        }

        if (events.empty())
            return;

        double dYaw = {}, dPitch = {};

        for (const auto& ev : events)
            if (ev.type == nbl::ui::SMouseEvent::EET_MOVEMENT)
            {
                dYaw += ev.movementEvent.relativeMovementX;
                dPitch += ev.movementEvent.relativeMovementY;
            }

        if (dPitch)
        {
            auto* pitch = output + count;
            pitch->type = dPitch > 0.f ? CVirtualGimbalEvent::TiltUp : CVirtualGimbalEvent::TiltDown;
            pitch->magnitude = std::abs(dPitch);
            count++;
        }

        if (dYaw)
        {
            auto* yaw = output + count;
            assert(yaw); // TODO: maybe just log error and return 0 count
            yaw->type = dYaw > 0.f ? CVirtualGimbalEvent::PanRight : CVirtualGimbalEvent::PanLeft;
            yaw->magnitude = std::abs(dYaw);
            count++;
        }
    }

    void endInputProcessing()
    {
        lastVirtualUpTimeStamp = nextPresentationTimeStamp;
    }

    inline const keys_to_virtual_events_t& getKeysToVirtualEvents() { return m_keysToVirtualEvents; }

protected:
    virtual void initKeysToEvent() = 0;

private:
    keys_to_virtual_events_t m_keysToVirtualEvents;
    bool m_keysDown[CVirtualGimbalEvent::EventsCount] = {};
    std::chrono::microseconds nextPresentationTimeStamp = {}, lastVirtualUpTimeStamp = {};
};

#if 0 // TOOD: update
template<typename R>
concept GimbalRange = GeneralPurposeRange<R> && requires 
{
    requires ProjectionMatrix<typename std::ranges::range_value_t<R>::projection_t>;
    requires std::same_as<std::ranges::range_value_t<R>, typename ICameraController<typename std::ranges::range_value_t<R>::projection_t>::CGimbal>;
};

template<GimbalRange Range>
class IGimbalRange : public IRange<typename Range>
{
public:
    using base_t = IRange<typename Range>;
    using range_t = typename base_t::range_t;
    using gimbal_t = typename base_t::range_value_t;

    IGimbalRange(range_t&& gimbals) : base_t(std::move(gimbals)) {}
    inline const range_t& getGimbals() const { return base_t::m_range; }

protected:
    inline range_t& getGimbals() const { return base_t::m_range; }
};

// TODO NOTE: eg. "follow camera" should use GimbalRange<std::array<ICameraController<T>::CGimbal, 2u>>, 
// one per camera itself and one for target it follows
#endif

} // nbl::hlsl namespace

#endif // _NBL_I_CAMERA_CONTROLLER_HPP_