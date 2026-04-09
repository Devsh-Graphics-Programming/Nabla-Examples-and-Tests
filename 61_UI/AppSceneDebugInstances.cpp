#include "app/App.hpp"

std::optional<uint32_t> App::findFrustumSourceBindingIx(const uint32_t planarIx) const
{
	if (m_viewports.activeRenderWindowIx < m_viewports.windowBindings.size())
	{
		const auto& activeBinding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
		if (activeBinding.activePlanarIx == planarIx && activeBinding.boundProjectionIx.has_value())
			return m_viewports.activeRenderWindowIx;
	}

	for (uint32_t bindingIx = 0u; bindingIx < m_viewports.windowBindings.size(); ++bindingIx)
	{
		const auto& binding = m_viewports.windowBindings[bindingIx];
		if (binding.activePlanarIx != planarIx)
			continue;
		if (!binding.boundProjectionIx.has_value())
			continue;
		return bindingIx;
	}

	return std::nullopt;
}

std::optional<uint32_t> App::tryBuildFrustumOverlaySourceBindingIx() const
{
	if (m_sceneInteraction.boundPlanarCameraIxToManipulate.has_value())
	{
		if (const auto sourceBindingIx = findFrustumSourceBindingIx(m_sceneInteraction.boundPlanarCameraIxToManipulate.value()); sourceBindingIx.has_value())
			return sourceBindingIx;
	}

	if (m_viewports.activeRenderWindowIx < m_viewports.windowBindings.size())
	{
		const auto& activeBinding = m_viewports.windowBindings[m_viewports.activeRenderWindowIx];
		if (activeBinding.boundProjectionIx.has_value() && activeBinding.activePlanarIx < m_planarProjections.size())
			return m_viewports.activeRenderWindowIx;
	}

	return std::nullopt;
}

void App::updateAuxSceneInstances(const size_t geometryCount)
{
	const uint32_t gridInstanceIx = SCameraAppSceneDefaults::CameraObjectIxOffset - 1u;
	if (m_debugScene.gridGeometryIx.has_value() && m_debugScene.renderer->m_instances.size() > gridInstanceIx)
	{
		const auto gridGeometryIx = m_debugScene.gridGeometryIx.value();
		if (gridGeometryIx < geometryCount)
		{
			auto& gridInstance = m_debugScene.renderer->m_instances[gridInstanceIx];
			gridInstance.packedGeo = m_debugScene.renderer->getGeometries().data() + gridGeometryIx;

			float32_t3x4 gridWorld = float32_t3x4(1.0f);
			gridWorld[0][0] = SCameraAppSceneDebugDefaults::GridExtent;
			gridWorld[2][2] = SCameraAppSceneDebugDefaults::GridExtent;
			hlsl::setTranslation(
				gridWorld,
				float32_t3(
					-0.5f * SCameraAppSceneDebugDefaults::GridExtent,
					SCameraAppSceneDebugDefaults::GridVerticalOffset,
					-0.5f * SCameraAppSceneDebugDefaults::GridExtent));
			gridInstance.world = gridWorld;
		}
	}

	const uint32_t followInstanceIx = m_debugScene.gridGeometryIx.has_value() ?
		SCameraAppSceneDefaults::CameraObjectIxOffset :
		SCameraAppSceneDefaults::FollowTargetObjectIx;
	if (m_debugScene.renderer->m_instances.size() <= followInstanceIx)
		return;

	auto& followInstance = m_debugScene.renderer->m_instances[followInstanceIx];
	if (m_sceneInteraction.followTargetVisible && m_debugScene.followTargetGeometryIx.has_value() && m_debugScene.followTargetGeometryIx.value() < geometryCount)
	{
		followInstance.packedGeo = m_debugScene.renderer->getGeometries().data() + m_debugScene.followTargetGeometryIx.value();
		followInstance.world = computeFollowTargetMarkerWorld();
	}
	else
	{
		followInstance.packedGeo = nullptr;
		followInstance.world = float32_t3x4(1.0f);
	}
}

void App::updateSceneDebugInstances()
{
	if (!m_debugScene.renderer || m_debugScene.renderer->m_instances.empty())
		return;

	auto& modelInstance = m_debugScene.renderer->m_instances[SCameraAppSceneDefaults::ModelObjectIx];
	modelInstance.world = m_sceneInteraction.model;

	const auto geometryCount = m_debugScene.renderer->getGeometries().size();
	if (geometryCount)
	{
		if (m_debugScene.geometrySelectionIx >= geometryCount)
			m_debugScene.geometrySelectionIx = 0u;
		modelInstance.packedGeo = m_debugScene.renderer->getGeometries().data() + m_debugScene.geometrySelectionIx;
	}
	updateAuxSceneInstances(geometryCount);
}
