#include "app/AppCameraConfigUtilities.hpp"

#include <span>
#include <string>

namespace nbl::system
{

bool tryCaptureInitialPlanarPresets(
    const CCameraGoalSolver& goalSolver,
    std::span<const core::smart_refctd_ptr<planar_projection_t>> planars,
    std::vector<CCameraPreset>& outPresets,
    std::string& outError)
{
    outPresets.clear();
    outPresets.reserve(planars.size());
    for (uint32_t planarIx = 0u; planarIx < planars.size(); ++planarIx)
    {
        auto* camera = planars[planarIx] ? planars[planarIx]->getCamera() : nullptr;
        const std::string presetName = "Planar " + std::to_string(planarIx);
        const auto captureAnalysis = CCameraGoalAnalysisUtilities::analyzeCameraCapture(goalSolver, camera);
        if (!captureAnalysis.canCapture)
        {
            const auto kindLabel = camera ? std::string(ext::cameras::CCameraKindUtilities::getCameraKindLabel(camera->getKind())) : std::string("Unknown");
            const auto reason =
                !captureAnalysis.hasCamera ? "missing camera" :
                (!captureAnalysis.capturedGoal ? "capture failed" :
                (!captureAnalysis.finiteGoal ? "non-finite goal" : "unknown"));
            std::string goalDetails;
            if (!captureAnalysis.finiteGoal)
            {
                const auto& goal = captureAnalysis.goal;
                goalDetails =
                    " position=(" + std::to_string(goal.position.x) + "," + std::to_string(goal.position.y) + "," + std::to_string(goal.position.z) + ")" +
                    " orientation=(" + std::to_string(goal.orientation.data.x) + "," + std::to_string(goal.orientation.data.y) + "," + std::to_string(goal.orientation.data.z) + "," + std::to_string(goal.orientation.data.w) + ")" +
                    " hasTarget=" + std::to_string(goal.hasTargetPosition) +
                    " target=(" + std::to_string(goal.targetPosition.x) + "," + std::to_string(goal.targetPosition.y) + "," + std::to_string(goal.targetPosition.z) + ")" +
                    " hasDistance=" + std::to_string(goal.hasDistance) +
                    " distance=" + std::to_string(goal.distance) +
                    " hasOrbit=" + std::to_string(goal.hasOrbitState) +
                    " orbit=(" + std::to_string(goal.orbitUv.x) + "," + std::to_string(goal.orbitUv.y) + "," + std::to_string(goal.orbitDistance) + ")";
            }
            outError =
                "Failed to capture initial planar preset " + std::to_string(planarIx) +
                " for camera kind \"" + kindLabel + "\": " + reason + goalDetails;
            return false;
        }

        CCameraPreset preset = {};
        if (!CCameraPresetFlowUtilities::tryCapturePreset(captureAnalysis, camera, presetName, preset))
        {
            outError =
                "Failed to build initial planar preset " + std::to_string(planarIx) +
                " for camera kind \"" + (camera ? std::string(ext::cameras::CCameraKindUtilities::getCameraKindLabel(camera->getKind())) : std::string("Unknown")) + "\".";
            return false;
        }

        outPresets.emplace_back(std::move(preset));
    }

    return true;
}

bool tryBuildPlanarProjectionCollectionFromConfig(
    const SCameraPlanarConfigCollections& planarConfig,
    const std::span<const core::smart_refctd_ptr<ext::cameras::ICamera>> cameras,
    const std::span<const ext::cameras::CPlanarProjection> projections,
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

            planar->getProjections().emplace_back(projections[projectionIx]);
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
        std::span<const core::smart_refctd_ptr<ext::cameras::ICamera>>(collections.cameras.data(), collections.cameras.size()),
        std::span<const ext::cameras::CPlanarProjection>(collections.projections.data(), collections.projections.size()),
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
