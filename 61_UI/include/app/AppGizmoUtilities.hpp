#ifndef _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_

#include "app/AppTypes.hpp"
#include "app/AppViewportBindingUtilities.hpp"

namespace nbl::ui
{

// ImGuizmo reads and writes its matrices as column major floats, through `&m[0][0]`. Reinterpreted as
// `hlsl::matrix` rows that is the transpose of the engine layout (basis in the columns, translation in the last
// column), so every matrix crossing that boundary is transposed: model matrices by the two helpers below, view
// and projection by `tryBuildViewportBoundCameraState`.
inline float64_t4x4 imguizmoTransformToEngine(const float32_t4x4& imguizmoTRS)
{
    return hlsl::transpose(getCastedMatrix<float64_t>(imguizmoTRS));
}

inline ImGuizmoModelM16InOut makeImGuizmoModel(const float32_t4x4& engineTransform)
{
    const auto imguizmoTRS = hlsl::transpose(engineTransform);
    return {
        .inTRS = imguizmoTRS,
        .outTRS = imguizmoTRS,
        .outDeltaTRS = SCameraAppTransformEditorUiDefaults::IdentityTransform
    };
}

/// @brief Translation, rotation and scale of the transform shown in the transform editor.
struct STransformEditorComponents
{
    float32_t3 translation = float32_t3(0.0f);
    hlsl::math::quaternion<hlsl::float32_t> orientation = hlsl::math::quaternion<hlsl::float32_t>::identity();
    float32_t3 scale = SCameraAppTransformEditorUiDefaults::IdentityScale;
};

inline STransformEditorComponents extractRigidTransformComponentsOrDefault(const float32_t4x4& transform)
{
    STransformEditorComponents components = {};
    // `transform` comes straight out of ImGuizmo, see `imguizmoTransformToEngine`
    const auto engineTransform = hlsl::transpose(transform);
    if (CCameraMathUtilities::tryExtractPositionAndQuaternionFromTransform(engineTransform, components.translation, components.orientation))
    {
        // the extraction divides the scale out of the basis columns, so the scale is their lengths
        components.scale = float32_t3(
            hlsl::length(float32_t3(engineTransform[0].x, engineTransform[1].x, engineTransform[2].x)),
            hlsl::length(float32_t3(engineTransform[0].y, engineTransform[1].y, engineTransform[2].y)),
            hlsl::length(float32_t3(engineTransform[0].z, engineTransform[1].z, engineTransform[2].z)));
        return components;
    }

    components.translation = float32_t3(transform[3].x, transform[3].y, transform[3].z);
    components.orientation = hlsl::math::quaternion<hlsl::float32_t>::identity();
    components.scale = SCameraAppTransformEditorUiDefaults::IdentityScale;
    return components;
}

/// @brief `eulerDegrees` is (pitch, yaw, roll), the layout `CCameraMathUtilities::getPitchYawRollDegrees` returns.
inline float32_t4x4 composeRigidTransform(
    const hlsl::float32_t3& translation,
    const hlsl::float32_t3& eulerDegrees,
    const hlsl::float32_t3& scale)
{
    // the result is handed back to ImGuizmo, so it goes back into its layout
    return hlsl::transpose(CCameraMathUtilities::composeTransformMatrix(
        translation,
        hlsl::math::quaternion<hlsl::float32_t>::createFromYawPitchRoll(hlsl::radians(eulerDegrees.y), hlsl::radians(eulerDegrees.x), hlsl::radians(eulerDegrees.z)),
        scale));
}

inline float computeViewportGizmoClipSize(
    const SBoundViewportCameraState& viewportState,
    const float32_t3& worldPosition,
    const float worldRadius)
{
    const auto viewPosition = mul(viewportState.viewMatrix, float32_t4(worldPosition, 1.0f));
    const float depth = std::max(SCameraAppViewportDefaults::MinPerspectiveGizmoDepth, hlsl::abs(viewPosition.z));
    if (viewportState.projection->getParameters().kind == CPlanarProjection::EKind::Perspective)
        return (worldRadius * viewportState.projectionMatrix[1][1]) / depth;

    return worldRadius * viewportState.projectionMatrix[1][1];
}

} // namespace nbl::ui

#endif // _NBL_THIS_EXAMPLE_APP_GIZMO_UTILITIES_HPP_
