#ifndef _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_

#include "app/AppTypes.hpp"
#include "app/AppViewportBindingUtilities.hpp"

namespace nbl::ui
{

// ImGuizmo reads and writes its matrices as column major floats, through `&m[0][0]`. Reinterpreted as
// `hlsl::matrix` rows that is the transpose of the engine layout (basis in the columns, translation in the last
// column), so every matrix crossing that boundary is transposed here and nowhere else.
inline float64_t4x4 imguizmoTransformToEngine(const float32_t4x4& imguizmoTRS)
{
    return hlsl::transpose(getCastedMatrix<float64_t>(imguizmoTRS));
}

inline ImGuizmoModelM16InOut makeImGuizmoModel(const float32_t4x4& transform)
{
    return {
        .inTRS = transform,
        .outTRS = transform,
        .outDeltaTRS = SCameraAppTransformEditorUiDefaults::IdentityTransform
    };
}

inline SRigidTransformComponents<hlsl::float32_t> extractRigidTransformComponentsOrDefault(const float32_t4x4& transform)
{
    SRigidTransformComponents<hlsl::float32_t> components = {};
    // `transform` comes straight out of ImGuizmo, see `imguizmoTransformToEngine`
    if (CCameraMathUtilities::tryExtractRigidTransformComponents(hlsl::transpose(transform), components))
        return components;

    components.translation = float32_t3(transform[3].x, transform[3].y, transform[3].z);
    components.orientation = hlsl::math::quaternion<hlsl::float32_t>::identity();
    components.scale = SCameraAppTransformEditorUiDefaults::IdentityScale;
    return components;
}

inline float32_t4x4 composeRigidTransform(
    const hlsl::float32_t3& translation,
    const hlsl::float32_t3& eulerDegrees,
    const hlsl::float32_t3& scale)
{
    // the result is handed back to ImGuizmo, so it goes back into its layout
    return hlsl::transpose(CCameraMathUtilities::composeTransformMatrix(
        translation,
        hlsl::math::quaternion<hlsl::float32_t>::createFromEulerAnglesXYZ(hlsl::radians(eulerDegrees.x), hlsl::radians(eulerDegrees.y), hlsl::radians(eulerDegrees.z)),
        scale));
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
