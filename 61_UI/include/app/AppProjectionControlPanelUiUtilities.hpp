#ifndef _NBL_THIS_EXAMPLE_APP_PROJECTION_CONTROL_PANEL_UI_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_PROJECTION_CONTROL_PANEL_UI_UTILITIES_HPP_

#include <string>

#include "app/AppViewportBindingUtilities.hpp"

namespace nbl::ui
{

using camera_panel_slider_spec_t = SCameraControlPanelSliderSpec;

template<typename RefreshRuntime>
inline bool drawRenderWindowSelector(
    const size_t windowCount,
    uint32_t& activeWindowIx,
    RefreshRuntime&& refreshRuntime)
{
    if (windowCount == 0u)
        return false;

    if (activeWindowIx >= windowCount)
        activeWindowIx = 0u;

    int currentWindowIx = static_cast<int>(activeWindowIx);
    const auto currentWindowLabel = "Window " + std::to_string(currentWindowIx);
    if (!ImGui::BeginCombo("Render Window", currentWindowLabel.c_str()))
        return true;

    for (size_t windowIx = 0u; windowIx < windowCount; ++windowIx)
    {
        const bool isSelected = currentWindowIx == static_cast<int>(windowIx);
        const auto windowLabel = "Window " + std::to_string(windowIx);
        if (ImGui::Selectable(windowLabel.c_str(), isSelected))
        {
            currentWindowIx = static_cast<int>(windowIx);
            activeWindowIx = static_cast<uint32_t>(currentWindowIx);
            if (!refreshRuntime())
            {
                ImGui::EndCombo();
                return false;
            }
        }

        if (isSelected)
            ImGui::SetItemDefaultFocus();
    }

    ImGui::EndCombo();
    return true;
}

template<typename RefreshRuntime>
inline bool drawProjectionPlanarSelector(
    std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
    SActiveProjectionTabContext& runtime,
    RefreshRuntime&& refreshRuntime)
{
    auto& binding = runtime.requireBinding();
    int currentPlanarIx = static_cast<int>(binding.activePlanarIx);
    const auto currentPlanarLabel = "Planar " + std::to_string(currentPlanarIx);
    if (!ImGui::BeginCombo("Active Planar", currentPlanarLabel.c_str()))
        return true;

    for (size_t planarIx = 0u; planarIx < planarProjections.size(); ++planarIx)
    {
        const bool isSelected = currentPlanarIx == static_cast<int>(planarIx);
        const auto planarLabel = "Planar " + std::to_string(planarIx);
        if (ImGui::Selectable(planarLabel.c_str(), isSelected))
        {
            currentPlanarIx = static_cast<int>(planarIx);
            trySelectBindingPlanar(
                planarProjections,
                binding,
                static_cast<uint32_t>(currentPlanarIx));
            if (!refreshRuntime())
            {
                ImGui::EndCombo();
                return false;
            }
        }

        if (isSelected)
            ImGui::SetItemDefaultFocus();
    }

    ImGui::EndCombo();
    return true;
}

inline std::string getProjectionPresetName(
    const IPlanarProjection::CProjection::ProjectionType projectionType,
    const uint32_t projectionIx)
{
    switch (projectionType)
    {
        case IPlanarProjection::CProjection::Perspective:
            return "Perspective Projection Preset " + std::to_string(projectionIx);
        case IPlanarProjection::CProjection::Orthographic:
            return "Orthographic Projection Preset " + std::to_string(projectionIx);
        default:
            return "Unknown Projection Preset " + std::to_string(projectionIx);
    }
}

inline bool drawProjectionPresetSelector(
    std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
    SActiveProjectionTabContext& runtime,
    const IPlanarProjection::CProjection::ProjectionType projectionType)
{
    bool updateBoundVirtualMaps = false;
    auto& binding = runtime.requireBinding();
    auto& projections = runtime.requirePlanar().getPlanarProjections();
    if (!ImGui::BeginCombo("Projection Preset", getProjectionPresetName(projectionType, binding.boundProjectionIx.value()).c_str()))
        return false;

    for (uint32_t projectionIx = 0u; projectionIx < projections.size(); ++projectionIx)
    {
        const auto& projection = projections[projectionIx];
        if (projection.getParameters().m_type != projectionType)
            continue;

        const bool isSelected = projectionIx == binding.boundProjectionIx.value();
        if (ImGui::Selectable(getProjectionPresetName(projectionType, projectionIx).c_str(), isSelected))
        {
            updateBoundVirtualMaps = trySelectBindingProjectionIndex(
                planarProjections,
                binding,
                projectionIx);
        }

        if (isSelected)
            ImGui::SetItemDefaultFocus();
    }

    ImGui::EndCombo();
    return updateBoundVirtualMaps;
}

template<typename RefreshRuntime>
inline bool drawProjectionTypeSelector(
    std::span<const smart_refctd_ptr<planar_projection_t>> planarProjections,
    SActiveProjectionTabContext& runtime,
    RefreshRuntime&& refreshRuntime)
{
    auto& binding = runtime.requireBinding();
    auto selectedProjectionType = runtime.requirePlanar().getPlanarProjections()[binding.boundProjectionIx.value()].getParameters().m_type;
    constexpr const char* ProjectionTypeLabels[] = { "Perspective", "Orthographic" };
    int type = static_cast<int>(selectedProjectionType);
    if (ImGui::Combo("Projection Type", &type, ProjectionTypeLabels, IM_ARRAYSIZE(ProjectionTypeLabels)))
    {
        selectedProjectionType = static_cast<IPlanarProjection::CProjection::ProjectionType>(type);
        trySelectBindingProjectionType(
            planarProjections,
            binding,
            selectedProjectionType);
        if (!refreshRuntime())
            return false;
    }

    CCameraControlPanelUiUtilities::drawHoverHint("Switch projection type for this planar");
    return true;
}

inline void drawProjectionHandednessControls(SWindowControlBinding& binding)
{
    if (ImGui::RadioButton("LH", binding.leftHandedProjection))
        binding.leftHandedProjection = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("RH", !binding.leftHandedProjection))
        binding.leftHandedProjection = false;
    CCameraControlPanelUiUtilities::drawHoverHint("Toggle left or right handed projection");
}

inline void drawProjectionParameterControls(
    SWindowControlBinding& binding,
    IPlanarProjection::CProjection& boundProjection,
    const bool useWindow)
{
    auto updateParameters = boundProjection.getParameters();
    if (useWindow)
        CCameraControlPanelUiUtilities::drawCheckboxWithHint({ .label = "Allow axes to flip##allowAxesToFlip", .value = &binding.allowGizmoAxesToFlip, .hint = "Allow ImGuizmo axes to flip based on view" });
    if (useWindow)
        CCameraControlPanelUiUtilities::drawCheckboxWithHint({ .label = "Draw debug grid##drawDebugGrid", .value = &binding.enableDebugGridDraw, .hint = "Toggle debug grid in the render window" });

    drawProjectionHandednessControls(binding);

    updateParameters.m_zNear = std::clamp(
        updateParameters.m_zNear,
        SCameraAppProjectionUiDefaults::NearPlaneMin,
        SCameraAppProjectionUiDefaults::NearPlaneMax);
    updateParameters.m_zFar = std::clamp(
        updateParameters.m_zFar,
        SCameraAppProjectionUiDefaults::FarPlaneMin,
        SCameraAppProjectionUiDefaults::FarPlaneMax);
    for (const auto& spec : {
        camera_panel_slider_spec_t{ .label = "zNear", .value = &updateParameters.m_zNear, .minValue = SCameraAppProjectionUiDefaults::NearPlaneMin, .maxValue = SCameraAppProjectionUiDefaults::NearPlaneMax, .format = "%.2f", .flags = ImGuiSliderFlags_Logarithmic, .hint = "Near clip plane" },
        camera_panel_slider_spec_t{ .label = "zFar", .value = &updateParameters.m_zFar, .minValue = SCameraAppProjectionUiDefaults::FarPlaneMin, .maxValue = SCameraAppProjectionUiDefaults::FarPlaneMax, .format = "%.1f", .flags = ImGuiSliderFlags_Logarithmic, .hint = "Far clip plane" }
    })
    {
        CCameraControlPanelUiUtilities::drawSliderFloatWithHint(spec);
    }

    switch (boundProjection.getParameters().m_type)
    {
        case IPlanarProjection::CProjection::Perspective:
            CCameraControlPanelUiUtilities::drawSliderFloatWithHint({
                .label = "Fov",
                .value = &updateParameters.m_planar.perspective.fov,
                .minValue = SCameraAppProjectionUiDefaults::PerspectiveFovMinDeg,
                .maxValue = SCameraAppProjectionUiDefaults::PerspectiveFovMaxDeg,
                .format = "%.1f",
                .flags = ImGuiSliderFlags_Logarithmic,
                .hint = "Perspective field of view"
            });
            boundProjection.setPerspective(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.perspective.fov);
            break;
        case IPlanarProjection::CProjection::Orthographic:
            CCameraControlPanelUiUtilities::drawSliderFloatWithHint({
                .label = "Ortho width",
                .value = &updateParameters.m_planar.orthographic.orthoWidth,
                .minValue = SCameraAppProjectionUiDefaults::OrthoWidthMin,
                .maxValue = SCameraAppProjectionUiDefaults::OrthoWidthMax,
                .format = "%.1f",
                .flags = ImGuiSliderFlags_Logarithmic,
                .hint = "Orthographic width"
            });
            boundProjection.setOrthographic(updateParameters.m_zNear, updateParameters.m_zFar, updateParameters.m_planar.orthographic.orthoWidth);
            break;
        default:
            break;
    }
}

inline void drawCursorBehaviourControls(bool& captureCursorInMoveMode, bool& resetCursorToCenter)
{
    if (!ImGui::TreeNodeEx("Cursor Behaviour"))
        return;

    CCameraControlPanelUiUtilities::drawCheckboxWithHint({ .label = "Capture OS cursor in move mode", .value = &captureCursorInMoveMode, .hint = "When disabled the app never warps or clamps system cursor" });
    if (captureCursorInMoveMode)
    {
        if (ImGui::RadioButton("Clamp to the window", !resetCursorToCenter))
            resetCursorToCenter = false;
        if (ImGui::RadioButton("Reset to the window center", resetCursorToCenter))
            resetCursorToCenter = true;
    }
    else
    {
        ImGui::TextDisabled("Cursor lock disabled");
    }

    ImGui::TreePop();
}

inline void drawBoundCameraMotionControls(ICamera& camera)
{
    float moveSpeed = camera.getMoveSpeedScale();
    float rotationSpeed = camera.getRotationSpeedScale();
    ImGui::SliderFloat(
        "Move speed factor",
        &moveSpeed,
        SCameraAppControlPanelRangeDefaults::MotionScaleMin,
        SCameraAppControlPanelRangeDefaults::MotionScaleMax,
        "%.4f",
        ImGuiSliderFlags_Logarithmic);
    CCameraControlPanelUiUtilities::drawHoverHint("Scale translation speed for this camera");
    if (camera.getAllowedVirtualEvents() & CVirtualGimbalEvent::Rotate)
    {
        ImGui::SliderFloat(
            "Rotate speed factor",
            &rotationSpeed,
            SCameraAppControlPanelRangeDefaults::MotionScaleMin,
            SCameraAppControlPanelRangeDefaults::MotionScaleMax,
            "%.4f",
            ImGuiSliderFlags_Logarithmic);
    }
    CCameraControlPanelUiUtilities::drawHoverHint("Scale rotation speed for this camera");
    camera.setMotionScales(moveSpeed, rotationSpeed);
}

template<typename AddMatrixTable, typename SyncBinding, typename SyncBindingToProjection>
inline void drawBoundCameraSection(
    SActiveProjectionTabContext& runtime,
    const uint32_t planarIx,
    AddMatrixTable&& addMatrixTableFn,
    SyncBinding&& syncBinding,
    SyncBindingToProjection&& syncBindingToProjection)
{
    auto& binding = runtime.requireBinding();
    auto& camera = runtime.requireCamera();
    const auto flags = ImGuiTreeNodeFlags_DefaultOpen;
    if (!ImGui::TreeNodeEx("Bound Camera", flags))
        return;

    ImGui::Text("Type: %s", camera.getIdentifier().data());
    ImGui::Text("Object Ix: %u", planarIx + SCameraAppSceneDefaults::CameraObjectIxOffset);
    ImGui::Separator();

    drawBoundCameraMotionControls(camera);

    ICamera::SphericalTargetState sphericalState;
    if (camera.tryGetSphericalTargetState(sphericalState))
    {
        float distance = sphericalState.distance;
        ImGui::SliderFloat(
            "Distance",
            &distance,
            sphericalState.minDistance,
            sphericalState.maxDistance,
            "%.4f",
            ImGuiSliderFlags_Logarithmic);
        CCameraControlPanelUiUtilities::drawHoverHint("Current orbit distance");
        camera.trySetSphericalDistance(distance);
    }

    if (ImGui::TreeNodeEx("World Data", flags))
    {
        auto& gimbal = camera.getGimbal();
        const auto position = getCastedVector<float32_t>(gimbal.getPosition());
        const auto orientation = getCastedVector<float32_t>(gimbal.getOrientation().data);
        const auto viewMatrix = getCastedMatrix<float32_t>(gimbal.getViewMatrix());

        addMatrixTableFn("Position", ("PositionTable_" + runtime.activePlanarIxString).c_str(), 1, 3, &position[0], false);
        addMatrixTableFn("Orientation (Quaternion)", ("OrientationTable_" + runtime.activePlanarIxString).c_str(), 1, 4, &orientation[0], false);
        addMatrixTableFn("View Matrix", ("ViewMatrixTable_" + runtime.activePlanarIxString).c_str(), 3, 4, &viewMatrix[0][0], false);
        ImGui::TreePop();
    }

    if (ImGui::TreeNodeEx("Virtual Event Mappings", flags))
    {
        syncBinding(binding);
        if (displayKeyMappingsAndVirtualStatesInline(&binding.inputBinding))
            syncBindingToProjection(binding);
        ImGui::TreePop();
    }

    ImGui::TreePop();
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_PROJECTION_CONTROL_PANEL_UI_UTILITIES_HPP_
