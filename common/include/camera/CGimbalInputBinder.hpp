#ifndef _NBL_C_GIMBAL_INPUT_BINDER_HPP_
#define _NBL_C_GIMBAL_INPUT_BINDER_HPP_

#include <vector>

#include "IGimbalController.hpp"

namespace nbl::hlsl
{

class CGimbalInputBinder final : public IGimbalController
{
public:
    using base_t = IGimbalController;
    using base_t::base_t;

    struct SCollectedVirtualEvents
    {
        std::vector<gimbal_event_t> events;
        uint32_t keyboardCount = 0u;
        uint32_t mouseCount = 0u;
        uint32_t imguizmoCount = 0u;

        inline uint32_t totalCount() const
        {
            return keyboardCount + mouseCount + imguizmoCount;
        }
    };

    // Runtime input binder. It translates external keyboard/mouse/ImGuizmo input into virtual events.
    inline void clearActiveBindings()
    {
        updateKeyboardMapping([](auto& map) { map.clear(); });
        updateMouseMapping([](auto& map) { map.clear(); });
        updateImguizmoMapping([](auto& map) { map.clear(); });
    }

    inline void clearBindingLayout()
    {
        clearActiveBindings();
    }

    inline void copyActiveBindingsFromEncoder(const IGimbalManipulateEncoder& encoder)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(encoder.getKeyboardVirtualEventMap()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(encoder.getMouseVirtualEventMap()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(encoder.getImguizmoVirtualEventMap()); });
    }

    inline void copyBindingLayoutFrom(const IGimbalManipulateEncoder& encoder)
    {
        copyActiveBindingsFromEncoder(encoder);
    }

    inline void copyDefaultBindingsFromEncoder(const IGimbalManipulateEncoder& encoder)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(encoder.getKeyboardMappingPreset()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(encoder.getMouseMappingPreset()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(encoder.getImguizmoMappingPreset()); });
    }

    inline void copyPresetLayoutFrom(const IGimbalManipulateEncoder& encoder)
    {
        copyDefaultBindingsFromEncoder(encoder);
    }

    inline void copyActiveBindingsToEncoder(IGimbalManipulateEncoder& encoder) const
    {
        encoder.updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(getKeyboardVirtualEventMap()); });
        encoder.updateMouseMapping([&](auto& map) { map = sanitizeMapping(getMouseVirtualEventMap()); });
        encoder.updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(getImguizmoVirtualEventMap()); });
    }

    inline void copyBindingLayoutTo(IGimbalManipulateEncoder& encoder) const
    {
        copyActiveBindingsToEncoder(encoder);
    }

    inline SCollectedVirtualEvents collectVirtualEvents(
        const std::chrono::microseconds nextPresentationTimeStamp,
        const SUpdateParameters parameters = {})
    {
        beginInputProcessing(nextPresentationTimeStamp);

        SCollectedVirtualEvents output;
        uint32_t keyboardPotentialCount = 0u;
        uint32_t mousePotentialCount = 0u;
        uint32_t imguizmoPotentialCount = 0u;

        processKeyboard(nullptr, keyboardPotentialCount, {});
        processMouse(nullptr, mousePotentialCount, {});
        processImguizmo(nullptr, imguizmoPotentialCount, {});

        output.events.resize(keyboardPotentialCount + mousePotentialCount + imguizmoPotentialCount);
        auto* dst = output.events.data();

        if (keyboardPotentialCount)
        {
            output.keyboardCount = keyboardPotentialCount;
            processKeyboard(dst, output.keyboardCount, parameters.keyboardEvents);
            dst += output.keyboardCount;
        }

        if (mousePotentialCount)
        {
            output.mouseCount = mousePotentialCount;
            processMouse(dst, output.mouseCount, parameters.mouseEvents);
            dst += output.mouseCount;
        }

        if (imguizmoPotentialCount)
        {
            output.imguizmoCount = imguizmoPotentialCount;
            processImguizmo(dst, output.imguizmoCount, parameters.imguizmoEvents);
        }

        endInputProcessing();
        output.events.resize(output.totalCount());
        return output;
    }

private:
    template<typename Map>
    inline static Map sanitizeMapping(const Map& source)
    {
        Map result;
        for (const auto& [code, hash] : source)
            result.emplace(code, typename Map::mapped_type(hash.event.type));
        return result;
    }
};

} // namespace nbl::hlsl

#endif // _NBL_C_GIMBAL_INPUT_BINDER_HPP_
