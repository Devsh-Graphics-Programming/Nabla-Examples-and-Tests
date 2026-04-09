#ifndef _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_

#include <limits>
#include <span>

#include "app/AppTypes.hpp"
#include "camera/CCameraFollowRegressionUtilities.hpp"

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

inline bool tryBuildCameraQueryBinding(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    ICamera* camera,
    SWindowControlBinding& outBinding)
{
    if (!camera)
        return false;

    for (uint32_t planarIx = 0u; planarIx < planarProjections.size(); ++planarIx)
    {
        const auto& planar = planarProjections[planarIx];
        if (!planar || planar->getCamera() != camera)
            continue;

        const auto& projections = planar->getPlanarProjections();
        if (projections.empty())
            return false;

        outBinding = {};
        outBinding.activePlanarIx = planarIx;
        outBinding.aspectRatio = 1.0f;
        outBinding.leftHandedProjection = true;
        outBinding.boundProjectionIx = 0u;

        for (uint32_t ix = 0u; ix < projections.size(); ++ix)
        {
            if (projections[ix].getParameters().m_type != IPlanarProjection::CProjection::Perspective)
                continue;

            outBinding.boundProjectionIx = ix;
            break;
        }
        return true;
    }

    return false;
}

inline bool tryGetBindingPlanarProjections(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    const SWindowControlBinding& binding,
    const planar_projections_range_t*& outProjections)
{
    outProjections = nullptr;
    if (binding.activePlanarIx >= planarProjections.size())
        return false;

    const auto& planar = planarProjections[binding.activePlanarIx];
    if (!planar)
        return false;

    outProjections = &planar->getPlanarProjections();
    return true;
}

inline bool trySelectBindingPlanar(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    const uint32_t planarIx)
{
    if (planarIx >= planarProjections.size())
        return false;

    const auto& planar = planarProjections[planarIx];
    if (!planar)
        return false;

    binding.activePlanarIx = planarIx;
    binding.pickDefaultProjections(planar->getPlanarProjections());
    return true;
}

inline bool ensureBindingDefaultProjections(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding)
{
    const planar_projections_range_t* projections = nullptr;
    if (!tryGetBindingPlanarProjections(planarProjections, binding, projections))
        return false;
    if (binding.lastBoundPerspectivePresetProjectionIx.has_value() && binding.lastBoundOrthoPresetProjectionIx.has_value())
        return true;

    binding.pickDefaultProjections(*projections);
    return true;
}

inline bool trySelectBindingProjectionType(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    const IPlanarProjection::CProjection::ProjectionType projectionType)
{
    if (!ensureBindingDefaultProjections(planarProjections, binding))
        return false;

    switch (projectionType)
    {
        case IPlanarProjection::CProjection::Perspective:
            if (!binding.lastBoundPerspectivePresetProjectionIx.has_value())
                return false;
            binding.boundProjectionIx = binding.lastBoundPerspectivePresetProjectionIx.value();
            return true;
        case IPlanarProjection::CProjection::Orthographic:
            if (!binding.lastBoundOrthoPresetProjectionIx.has_value())
                return false;
            binding.boundProjectionIx = binding.lastBoundOrthoPresetProjectionIx.value();
            return true;
        default:
            return false;
    }
}

inline bool trySelectBindingProjectionIndex(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    const uint32_t projectionIx)
{
    const planar_projections_range_t* projections = nullptr;
    if (!tryGetBindingPlanarProjections(planarProjections, binding, projections))
        return false;
    if (projectionIx >= projections->size())
        return false;

    binding.boundProjectionIx = projectionIx;
    const auto projectionType = (*projections)[projectionIx].getParameters().m_type;
    if (projectionType == IPlanarProjection::CProjection::Perspective)
        binding.lastBoundPerspectivePresetProjectionIx = projectionIx;
    else if (projectionType == IPlanarProjection::CProjection::Orthographic)
        binding.lastBoundOrthoPresetProjectionIx = projectionIx;
    return true;
}

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
    outState.viewMatrix = getCastedMatrix<float32_t>(getMatrix3x4As4x4(camera->getGimbal().getViewMatrix()));
    outState.projectionMatrix = getCastedMatrix<float32_t>(projection.getProjectionMatrix());
    outState.viewProjMatrix = mul(outState.projectionMatrix, outState.viewMatrix);

    binding.isOrthographicProjection = projection.getParameters().m_type == IPlanarProjection::CProjection::Orthographic;
    binding.viewMatrix = getCastedMatrix<float32_t>(camera->getGimbal().getViewMatrix());
    binding.projectionMatrix = outState.projectionMatrix;
    binding.viewProjMatrix = outState.viewProjMatrix;
    return true;
}

inline void buildProjectionContextFromViewportState(
    const SBoundViewportCameraState& viewportState,
    nbl::system::SCameraProjectionContext& outProjectionContext)
{
    outProjectionContext.viewMatrix = viewportState.viewMatrix;
    outProjectionContext.projectionMatrix = viewportState.projectionMatrix;
}

inline bool tryBuildActiveViewportRuntimeState(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    std::span<SWindowControlBinding> windowBindings,
    const uint32_t activeWindowIx,
    SActiveViewportRuntimeState& outState)
{
    outState = {};
    if (activeWindowIx >= windowBindings.size())
        return false;

    auto& binding = windowBindings[activeWindowIx];
    if (binding.activePlanarIx >= planarProjections.size())
        return false;

    auto& planar = planarProjections[binding.activePlanarIx];
    auto* camera = planar ? planar->getCamera() : nullptr;
    if (!planar || !camera)
        return false;

    outState = {
        .binding = &binding,
        .planar = planar.get(),
        .camera = camera
    };
    return true;
}

inline bool tryBuildBindingProjectionContext(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    SWindowControlBinding& binding,
    nbl::system::SCameraProjectionContext& outProjectionContext)
{
    SBoundViewportCameraState viewportState = {};
    if (!tryBuildWindowBindingMatrices(planarProjections, binding, viewportState))
        return false;

    buildProjectionContextFromViewportState(viewportState, outProjectionContext);
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

inline bool tryBuildCameraProjectionContext(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    ICamera* camera,
    nbl::system::SCameraProjectionContext& outProjectionContext)
{
    SWindowControlBinding binding = {};
    if (!tryBuildCameraQueryBinding(planarProjections, camera, binding))
        return false;

    return tryBuildBindingProjectionContext(planarProjections, binding, outProjectionContext);
}

inline bool initializeWindowBindingDefaults(
    std::span<const nbl::core::smart_refctd_ptr<planar_projection_t>> planarProjections,
    std::span<SWindowControlBinding> windowBindings)
{
    if (planarProjections.empty())
        return false;

    for (uint32_t windowIx = 0u; windowIx < windowBindings.size(); ++windowIx)
    {
        auto& binding = windowBindings[windowIx];
        binding.activePlanarIx = 0u;

        const auto& planar = planarProjections[binding.activePlanarIx];
        if (!planar)
            return false;

        binding.pickDefaultProjections(planar->getPlanarProjections());
        binding.boundProjectionIx = windowIx == 0u ?
            binding.lastBoundPerspectivePresetProjectionIx :
            binding.lastBoundOrthoPresetProjectionIx;
    }

    return true;
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_VIEWPORT_BINDING_UTILITIES_HPP_
