#include "app/App.hpp"

bool App::initializeGeometrySceneResources()
{
	const uint32_t additionalBufferOwnershipFamilies[] = { getGraphicsQueue()->getFamilyIndex() };
	m_debugScene.scene = CGeometryCreatorScene::create(
		{
			.transferQueue = getTransferUpQueue(),
			.utilities = m_utils.get(),
			.logger = m_logger.get(),
			.addtionalBufferOwnershipFamilies = additionalBufferOwnershipFamilies
		},
		CSimpleDebugRenderer::DefaultPolygonGeometryPatch);

	return m_debugScene.scene || logFail("Could not create geometry creator scene!");
}

bool App::initializeDebugSceneRendererResources()
{
	const auto& geometries = m_debugScene.scene->getInitParams().geometries;
	if (geometries.empty())
		return logFail("No geometries found for scene!");

	m_debugScene.renderer = CSimpleDebugRenderer::create(m_assetMgr.get(), m_debugScene.renderpass.get(), 0, { &geometries.front().get(), geometries.size() });
	if (!m_debugScene.renderer)
		return logFail("Failed to create debug renderer!");

	const auto& pipelines = m_debugScene.renderer->getInitParams().pipelines;
	m_debugScene.gridGeometryIx = std::nullopt;
	m_debugScene.followTargetGeometryIx = std::nullopt;
	auto ix = 0u;
	for (const auto& name : m_debugScene.scene->getInitParams().geometryNames)
	{
		if (name == "Cube")
		{
			if (!m_debugScene.followTargetGeometryIx.has_value())
				m_debugScene.followTargetGeometryIx = ix;
		}
		else if (name == "Cone")
		{
			m_debugScene.renderer->getGeometry(ix).pipeline = pipelines[CSimpleDebugRenderer::SInitParams::PipelineType::Cone];
		}
		else if (name == "Grid")
		{
			m_debugScene.gridGeometryIx = ix;
		}
		++ix;
	}

	m_debugScene.renderer->m_instances.resize(1u + (m_debugScene.gridGeometryIx.has_value() ? 1u : 0u) + (m_debugScene.followTargetGeometryIx.has_value() ? 1u : 0u));
	return true;
}
