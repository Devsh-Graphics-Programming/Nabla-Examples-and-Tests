#include "app/App.hpp"
#include "app/AppGizmoUtilities.hpp"

void App::drawViewportManipulationGizmos(
	uint32_t windowIx,
	SWindowControlBinding& binding,
	const nbl::ui::SBoundViewportCameraState& viewportState,
	size_t& gizmoIx)
{
	for (uint32_t objectIx = 0u; objectIx < getManipulableObjectCount(); ++objectIx)
	{
		ImGuizmo::PushID(gizmoIx);
		++gizmoIx;

		SManipulableObjectContext objectContext = {};
		if (!tryBuildManipulableObjectContext(objectIx, objectContext))
		{
			ImGuizmo::PopID();
			continue;
		}

		if (objectContext.camera == viewportState.camera)
		{
			ImGuizmo::PopID();
			continue;
		}

		auto imguizmoModel = nbl::ui::makeImGuizmoModel(objectContext.transform);

		const float gizmoWorldRadius = objectContext.isFollowTarget() ? SCameraAppViewportDefaults::FollowTargetGizmoWorldRadius : SCameraAppViewportDefaults::DefaultGizmoWorldRadius;
		const float gizmoSizeClip = nbl::ui::computeViewportGizmoClipSize(
			viewportState,
			objectContext.worldPosition,
			gizmoWorldRadius);
		ImGuizmo::SetGizmoSizeClipSpace(gizmoSizeClip);

		const bool success = ImGuizmo::Manipulate(
			&viewportState.imguizmoPlanar.view[0][0],
			&viewportState.imguizmoPlanar.projection[0][0],
			ImGuizmo::OPERATION::UNIVERSAL,
			m_gizmoState.mode,
			&imguizmoModel.outTRS[0][0],
			&imguizmoModel.outDeltaTRS[0][0],
			m_gizmoState.useSnap ? &m_gizmoState.snap[0] : nullptr);

		if (success)
		{
			bindManipulableObject(objectContext);
			applyManipulableObjectTransform(objectContext, getCastedMatrix<float64_t>(imguizmoModel.outTRS));
		}

		if (ImGuizmo::IsOver() && !ImGuizmo::IsUsingAny() && !m_viewports.enableActiveCameraMovement)
		{
			if (objectContext.isCamera() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
			{
				const uint32_t newPlanarIx = objectContext.planarIx.value();
				if (nbl::ui::trySelectBindingPlanar(
					getPlanarProjectionSpan(),
						binding,
						newPlanarIx))
				{
					updateActiveRenderWindowFromViewport(windowIx, false, true);
				}
			}
			else if (objectContext.isFollowTarget() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
			{
				bindManipulableObject(objectContext);
			}
			else if (!objectContext.isCamera() && !objectContext.isFollowTarget() && ImGui::IsMouseClicked(ImGuiMouseButton_Right))
			{
				bindManipulableObject(objectContext);
			}

			drawManipulableObjectHoverOverlay(objectContext);
		}

		ImGuizmo::PopID();
	}
}

void App::drawManipulableObjectHoverOverlay(const SManipulableObjectContext& objectContext) const
{
	const ImVec2 mousePos = ImGui::GetIO().MousePos;
	nbl::ui::CCameraViewportOverlayUtilities::beginHoverInfoOverlay("InfoOverlay", mousePos);

	ImGui::Text("Identifier: %s", objectContext.label.c_str());
	ImGui::Text("Object Ix: %u", objectContext.objectIx);
	if (objectContext.isCamera())
	{
		ImGui::Separator();
		ImGui::TextDisabled("RMB: switch view to this camera");
		ImGui::TextDisabled("LMB drag: manipulate gizmo");
		ImGui::TextDisabled("SPACE: toggle move mode");
	}
	else if (objectContext.isFollowTarget())
	{
		ImGui::Separator();
		ImGui::TextDisabled("RMB: select follow target");
		ImGui::TextDisabled("LMB drag: move or rotate tracked target");
		ImGui::TextDisabled("Enabled follow cameras update on the next frame");
	}

	nbl::ui::CCameraViewportOverlayUtilities::endHoverInfoOverlay();
}
