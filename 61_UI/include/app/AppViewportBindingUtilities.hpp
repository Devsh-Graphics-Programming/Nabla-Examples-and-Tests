#ifndef _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_

#include <limits>
#include <span>

#include "app/AppTypes.hpp"

namespace nbl::ui
{

struct SBoundViewportCameraState final
{
    ICamera* camera = nullptr;
    IPlanarProjection::CProjection* projection = nullptr;
    float32_t4x4 viewMatrix = float32_t4x4(1.0f);
    float32_t4x4 projectionMatrix = float32_t4x4(1.0f);
    float32_t4x4 viewProjMatrix = float32_t4x4(1.0f);
    ImGuizmoPlanarM16InOut imguizmoPlanar = {};
};

inline bool tryBuildWindowBindingMatrices(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    SBoundViewportCameraState& outState)
{
    if (!binding.boundProjectionIx.has_value())
        return false;
    if (binding.activePlanarIx >= planarProjections.size())
        return false;

    auto& planar = planarProjections[binding.activePlanarIx];
    if (!planar)
        return false;

    auto* const camera = planar->getCamera();
    if (!camera)
        return false;

    auto& projections = planar->getPlanarProjections();
    const uint32_t projectionIx = binding.boundProjectionIx.value();
    if (projectionIx >= projections.size())
        return false;

    auto& projection = projections[projectionIx];
    nbl::core::syncDynamicPerspectiveProjection(camera, projection);
    projection.update(binding.leftHandedProjection, binding.aspectRatio);

    outState.camera = camera;
    outState.projection = &projection;
    outState.viewMatrix = getMatrix3x4As4x4(getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix()));
    outState.projectionMatrix = getCastedMatrix<float32_t>(projection.getProjectionMatrix());
    outState.viewProjMatrix = mul(outState.projectionMatrix, outState.viewMatrix);

    binding.isOrthographicProjection = projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic;
    binding.viewMatrix = getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix());
    binding.projectionMatrix = outState.projectionMatrix;
    binding.viewProjMatrix = outState.viewProjMatrix;
    return true;
}

inline bool tryBuildViewportBoundCameraState(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    const ImVec2& viewportSize,
    const bool flipGizmoY,
    SBoundViewportCameraState& outState)
{
    constexpr float MinViewportExtent = std::numeric_limits<float>::epsilon();
    if (viewportSize.x <= MinViewportExtent || viewportSize.y <= MinViewportExtent)
        return false;

    binding.aspectRatio = viewportSize.x / viewportSize.y;
    if (!tryBuildWindowBindingMatrices(planarProjections, binding, outState))
        return false;

    outState.imguizmoPlanar.view = getCastedMatrix<float32_t>(hlsl::transpose(outState.viewMatrix));
    outState.imguizmoPlanar.projection = getCastedMatrix<float32_t>(hlsl::transpose(outState.projectionMatrix));
    if (flipGizmoY)
        outState.imguizmoPlanar.projection[1][1] *= -1.0f;
    return true;
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_
