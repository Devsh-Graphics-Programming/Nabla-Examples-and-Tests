#ifndef _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_

#include "app/AppTypes.hpp"
#include "app/AppViewportBindingUtilities.hpp"

namespace nbl::ui
{

inline ImGuizmoModelM16InOut makeImGuizmoModel(const float32_t4x4& transform)
{
    return {
        .inTRS = transform,
        .outTRS = transform,
        .outDeltaTRS = SCameraAppTransformEditorUiDefaults::IdentityTransform
    };
}

inline hlsl::SRigidTransformComponents<hlsl::float32_t> extractRigidTransformComponentsOrDefault(const float32_t4x4& transform)
{
    hlsl::SRigidTransformComponents<hlsl::float32_t> components = {};
    if (hlsl::CCameraMathUtilities::tryExtractRigidTransformComponents(transform, components))
        return components;

    components.translation = float32_t3(transform[3].x, transform[3].y, transform[3].z);
    components.orientation = hlsl::CCameraMathUtilities::makeIdentityQuaternion<hlsl::float32_t>();
    components.scale = SCameraAppTransformEditorUiDefaults::IdentityScale;
    return components;
}

inline float32_t4x4 composeRigidTransform(
    const hlsl::float32_t3& translation,
    const hlsl::float32_t3& eulerDegrees,
    const hlsl::float32_t3& scale)
{
    return hlsl::CCameraMathUtilities::composeTransformMatrix(
        translation,
        hlsl::CCameraMathUtilities::makeQuaternionFromEulerDegrees(eulerDegrees),
        scale);
}

inline float computeViewportGizmoClipSize(
    const SBoundViewportCameraState& viewportState,
    const float32_t3& worldPosition,
    const float worldRadius)
{
    const auto viewPosition = mul(viewportState.viewMatrix, float32_t4(worldPosition, 1.0f));
    const float depth = std::max(SCameraAppViewportDefaults::MinPerspectiveGizmoDepth, hlsl::abs(viewPosition.z));
    if (viewportState.projection->getParameters().m_type == IPlanarProjection::CProjection::Perspective)
        return (worldRadius * viewportState.projectionMatrix[1][1]) / depth;

    return worldRadius * viewportState.projectionMatrix[1][1];
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_
