#include "app/AppCameraConfigUtilities.hpp"

#include <span>
#include <string>

namespace
{

template<typename BindingMap, typename ApplyBinding>
bool tryApplyProjectionBindingSelection(
    const std::optional<uint32_t>& bindingIx,
    std::span<const BindingMap> bindings,
    const char* label,
    ApplyBinding&& applyBinding,
    std::string& error)
{
    if (!bindingIx.has_value())
        return true;
    if (bindingIx.value() >= bindings.size())
    {
        error = std::string(label) + " binding index out of range.";
        return false;
    }

    applyBinding(bindings[bindingIx.value()]);
    return true;
}

} // namespace

namespace nbl::system
{

bool tryCaptureInitialPlanarPresets(
    const core::CCameraGoalSolver& goalSolver,
    std::span<const core::smart_refctd_ptr<planar_projection_t>> planars,
    std::vector<core::CCameraPreset>& outPresets,
    std::string& outError)
{
    outPresets.clear();
    outPresets.reserve(planars.size());
    for (uint32_t planarIx = 0u; planarIx < planars.size(); ++planarIx)
    {
        auto* camera = planars[planarIx] ? planars[planarIx]->getCamera() : nullptr;
        const std::string presetName = "Planar " + std::to_string(planarIx);
        const auto captureAnalysis = core::CCameraGoalAnalysisUtilities::analyzeCameraCapture(goalSolver, camera);
        if (!captureAnalysis.canCapture)
        {
            const auto kindLabel = camera ? std::string(core::CCameraKindUtilities::getCameraKindLabel(camera->getKind())) : std::string("Unknown");
            const auto reason =
                !captureAnalysis.hasCamera ? "missing camera" :
                (!captureAnalysis.capturedGoal ? "capture failed" :
                (!captureAnalysis.finiteGoal ? "non-finite goal" : "unknown"));
            outError =
                "Failed to capture initial planar preset " + std::to_string(planarIx) +
                " for camera kind \"" + kindLabel + "\": " + reason;
            return false;
        }

        core::CCameraPreset preset = {};
        if (!core::tryCapturePreset(captureAnalysis, camera, presetName, preset))
        {
            outError =
                "Failed to build initial planar preset " + std::to_string(planarIx) +
                " for camera kind \"" + (camera ? std::string(core::CCameraKindUtilities::getCameraKindLabel(camera->getKind())) : std::string("Unknown")) + "\".";
            return false;
        }

        outPresets.emplace_back(std::move(preset));
    }

    return true;
}

bool tryBuildPlanarProjectionCollectionFromConfig(
    const SCameraPlanarConfigCollections& planarConfig,
    const std::span<const core::smart_refctd_ptr<core::ICamera>> cameras,
    const std::span<const core::IPlanarProjection::CProjection> projections,
    const SCameraInputBindingCollections& bindings,
    std::vector<core::smart_refctd_ptr<planar_projection_t>>& outPlanars,
    std::string& error)
{
    outPlanars.clear();
    if (!planarConfig.valid())
    {
        error = "Camera planar config is missing.";
        return false;
    }

    outPlanars.reserve(planarConfig.planars.size());
    for (const auto& planarConfigEntry : planarConfig.planars)
    {
        const auto cameraIx = planarConfigEntry.cameraIx;
        if (cameraIx >= cameras.size())
        {
            error = "Planar camera index out of range.";
            return false;
        }

        auto& planar = outPlanars.emplace_back() = planar_projection_t::create(core::smart_refctd_ptr(cameras[cameraIx]));
        for (const auto viewportIx : planarConfigEntry.viewportIxs)
        {
            if (viewportIx >= planarConfig.viewports.size())
            {
                error = "Viewport index out of range in planar definition.";
                return false;
            }

            const auto& viewport = planarConfig.viewports[viewportIx];
            const auto projectionIx = viewport.projectionIx;
            if (projectionIx >= projections.size())
            {
                error = "Planar projection index out of range.";
                return false;
            }

            auto& projection = planar->getPlanarProjections().emplace_back(projections[projectionIx]);
            auto& projectionBinding = projection.getInputBinding();
            if (!tryApplyProjectionBindingSelection(
                    viewport.bindings.keyboard,
                    std::span<const decltype(bindings.keyboard)::value_type>(bindings.keyboard.data(), bindings.keyboard.size()),
                    "Keyboard",
                    [&](const auto& map) { projectionBinding.updateKeyboardMapping([&](auto& dst) { dst = map; }); },
                    error))
            {
                return false;
            }

            if (!tryApplyProjectionBindingSelection(
                    viewport.bindings.mouse,
                    std::span<const decltype(bindings.mouse)::value_type>(bindings.mouse.data(), bindings.mouse.size()),
                    "Mouse",
                    [&](const auto& map) { projectionBinding.updateMouseMapping([&](auto& dst) { dst = map; }); },
                    error))
            {
                return false;
            }
        }
    }

    return !outPlanars.empty();
}

bool tryBuildCameraPlanarRuntime(
    const SCameraConfigCollections& collections,
    std::vector<core::smart_refctd_ptr<planar_projection_t>>& outPlanars,
    std::string& error)
{
    if (!collections.planarConfig.valid())
    {
        error = "Camera planar configuration is missing.";
        return false;
    }

    return tryBuildPlanarProjectionCollectionFromConfig(
        collections.planarConfig,
        std::span<const core::smart_refctd_ptr<core::ICamera>>(collections.cameras.data(), collections.cameras.size()),
        std::span<const core::IPlanarProjection::CProjection>(collections.projections.data(), collections.projections.size()),
        collections.bindings,
        outPlanars,
        error);
}

bool tryGetEmbeddedCameraScriptedInputText(
    const SCameraConfigCollections& collections,
    std::string& outText)
{
    if (!collections.hasEmbeddedScriptedInputText())
        return false;

    outText = collections.embeddedScriptedInputText;
    return true;
}

} // namespace nbl::system
