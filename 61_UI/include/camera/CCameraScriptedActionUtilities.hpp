#ifndef _NBL_THIS_EXAMPLE_CAMERA_SCRIPTED_ACTION_UTILITIES_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_CAMERA_SCRIPTED_ACTION_UTILITIES_HPP_INCLUDED_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "nbl/ext/Cameras/CCameraScriptedRuntime.hpp"

namespace nbl::this_example
{

enum class ECameraScriptedActionCode : int32_t
{
    SetActiveRenderWindow = 1,
    SetActivePlanar = 2,
    SetProjectionType = 3,
    SetProjectionIndex = 4,
    SetUseWindow = 5,
    SetLeftHanded = 6,
    ResetActiveCamera = 7
};

struct CCameraScriptedActionEvent final
{
    uint64_t frame = 0ull;
    int32_t code = 0;
    int32_t value = 0;
};

struct CCameraScriptedActionUtilities final
{
    static inline constexpr int32_t toCode(const ECameraScriptedActionCode code)
    {
        return static_cast<int32_t>(code);
    }

    static inline bool hasCode(const CCameraScriptedActionEvent& action, const ECameraScriptedActionCode code)
    {
        return action.code == toCode(code);
    }

    static inline void appendActionEvent(
        std::vector<CCameraScriptedActionEvent>& actions,
        const uint64_t frame,
        const ECameraScriptedActionCode code,
        const int32_t value)
    {
        actions.emplace_back(CCameraScriptedActionEvent{
            .frame = frame,
            .code = toCode(code),
            .value = value
        });
    }

    static inline void finalizeActionEvents(std::vector<CCameraScriptedActionEvent>& actions)
    {
        std::stable_sort(actions.begin(), actions.end(),
            [](const CCameraScriptedActionEvent& lhs, const CCameraScriptedActionEvent& rhs)
            {
                return lhs.frame < rhs.frame;
            });
    }

    static inline void dequeueFrameActions(
        const std::vector<CCameraScriptedActionEvent>& actions,
        size_t& nextActionIndex,
        const uint64_t frame,
        std::vector<CCameraScriptedActionEvent>& out)
    {
        out.clear();
        while (nextActionIndex < actions.size() && actions[nextActionIndex].frame == frame)
        {
            out.emplace_back(actions[nextActionIndex]);
            ++nextActionIndex;
        }
    }
};

} // namespace nbl::this_example

#endif
