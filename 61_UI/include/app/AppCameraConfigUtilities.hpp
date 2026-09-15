#ifndef _APP_CAMERA_CONFIG_UTILITIES_HPP_
#define _APP_CAMERA_CONFIG_UTILITIES_HPP_

#include <span>
#include <string>
#include <optional>
#include <vector>

#include "app/AppResourceUtilities.hpp"
#include "app/AppTypes.hpp"

namespace nbl::system
{

struct SCameraInputBindingCollections final
{
    std::vector<ui::IGimbalBindingLayout::keyboard_to_virtual_events_t> keyboard;
    std::vector<ui::IGimbalBindingLayout::mouse_to_virtual_events_t> mouse;
};

struct SCameraViewportBindingSelection final
{
    std::optional<uint32_t> keyboard = std::nullopt;
    std::optional<uint32_t> mouse = std::nullopt;
};

struct SCameraViewportConfig final
{
    uint32_t projectionIx = 0u;
    SCameraViewportBindingSelection bindings = {};
};

struct SCameraPlanarConfig final
{
    uint32_t cameraIx = 0u;
    std::vector<uint32_t> viewportIxs = {};
};

struct SCameraPlanarConfigCollections final
{
    std::vector<SCameraViewportConfig> viewports = {};
    std::vector<SCameraPlanarConfig> planars = {};

    inline bool valid() const
    {
        return !planars.empty();
    }
};

struct SCameraConfigCollections final
{
    std::string embeddedScriptedInputText = {};
    std::vector<core::smart_refctd_ptr<core::ICamera>> cameras = {};
    std::vector<core::IPlanarProjection::CProjection> projections = {};
    SCameraInputBindingCollections bindings = {};
    SCameraPlanarConfigCollections planarConfig = {};

    inline bool hasEmbeddedScriptedInputText() const
    {
        return !embeddedScriptedInputText.empty();
    }
};

struct SCameraPlanarRuntimeBootstrap final
{
    SCameraConfigLoadResult loadResult = {};
    SCameraConfigCollections collections = {};
    std::vector<core::smart_refctd_ptr<planar_projection_t>> planars = {};
};

bool tryLoadCameraConfigCollections(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraConfigLoadResult& outLoadResult,
    SCameraConfigCollections& outCollections,
    std::string* error = nullptr);

bool tryBuildCameraConfigCollections(
    const std::string_view text,
    SCameraConfigCollections& outCollections,
    std::string& error);

bool tryBuildCameraPlanarRuntime(
    const SCameraConfigCollections& collections,
    std::vector<core::smart_refctd_ptr<planar_projection_t>>& outPlanars,
    std::string& error);

bool tryBuildCameraPlanarRuntimeBootstrap(
    const SCameraAppResourceContext& context,
    const SCameraConfigLoadRequest& request,
    SCameraPlanarRuntimeBootstrap& outBootstrap,
    std::string* error = nullptr);

bool tryGetEmbeddedCameraScriptedInputText(
    const SCameraConfigCollections& collections,
    std::string& outText);

bool tryCaptureInitialPlanarPresets(
    const core::CCameraGoalSolver& goalSolver,
    std::span<const core::smart_refctd_ptr<planar_projection_t>> planars,
    std::vector<core::CCameraPreset>& outPresets,
    std::string& outError);

} // namespace nbl::system

#endif // _APP_CAMERA_CONFIG_UTILITIES_HPP_
