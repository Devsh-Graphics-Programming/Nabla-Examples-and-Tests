#ifndef _NBL_C_GIMBAL_INPUT_BINDER_HPP_
#define _NBL_C_GIMBAL_INPUT_BINDER_HPP_

#include "IGimbalController.hpp"

namespace nbl::hlsl
{

class CGimbalInputBinder final : public IGimbalController
{
public:
    using base_t = IGimbalController;
    using base_t::base_t;

    inline void clearBindingLayout()
    {
        updateKeyboardMapping([](auto& map) { map.clear(); });
        updateMouseMapping([](auto& map) { map.clear(); });
        updateImguizmoMapping([](auto& map) { map.clear(); });
    }

    inline void copyBindingLayoutFrom(const IGimbalManipulateEncoder& encoder)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(encoder.getKeyboardVirtualEventMap()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(encoder.getMouseVirtualEventMap()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(encoder.getImguizmoVirtualEventMap()); });
    }

    inline void copyPresetLayoutFrom(const IGimbalManipulateEncoder& encoder)
    {
        updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(encoder.getKeyboardMappingPreset()); });
        updateMouseMapping([&](auto& map) { map = sanitizeMapping(encoder.getMouseMappingPreset()); });
        updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(encoder.getImguizmoMappingPreset()); });
    }

    inline void copyBindingLayoutTo(IGimbalManipulateEncoder& encoder) const
    {
        encoder.updateKeyboardMapping([&](auto& map) { map = sanitizeMapping(getKeyboardVirtualEventMap()); });
        encoder.updateMouseMapping([&](auto& map) { map = sanitizeMapping(getMouseVirtualEventMap()); });
        encoder.updateImguizmoMapping([&](auto& map) { map = sanitizeMapping(getImguizmoVirtualEventMap()); });
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
