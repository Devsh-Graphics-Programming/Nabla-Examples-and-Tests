#include "app/App.hpp"

namespace
{

inline float32_t4x4 buildModelManipulationTransform(const float32_t3x4& model)
{
	return hlsl::transpose(getMatrix3x4As4x4(model));
}

inline float32_t3 extractWorldPosition(const float32_t4x4& transform)
{
	return float32_t3(transform[3].x, transform[3].y, transform[3].z);
}

inline float32_t4x4 buildCameraManipulationTransform(ICamera& camera)
{
	return getCastedMatrix<float32_t>(camera.getGimbal().template operator()<float64_t4x4>());
}

inline float32_t3 buildCameraWorldPosition(ICamera& camera)
{
	return getCastedVector<float32_t>(camera.getGimbal().getPosition());
}

inline float32_t4x4 buildFollowTargetTransform(const CTrackedTarget& trackedTarget)
{
	return getCastedMatrix<float32_t>(trackedTarget.getGimbal().template operator()<float64_t4x4>());
}

inline float32_t3 buildFollowTargetWorldPosition(const CTrackedTarget& trackedTarget)
{
	return getCastedVector<float32_t>(trackedTarget.getGimbal().getPosition());
}

} // namespace

uint32_t App::getManipulableObjectCount() const
{
	return SCameraAppSceneDefaults::CameraObjectIxOffset + static_cast<uint32_t>(m_planarProjections.size());
}

bool App::isManipulableObjectFollowTarget(const uint32_t objectIx) const
{
	return objectIx == SCameraAppSceneDefaults::FollowTargetObjectIx;
}

std::optional<uint32_t> App::getManipulableObjectPlanarIx(const uint32_t objectIx) const
{
	if (objectIx < SCameraAppSceneDefaults::CameraObjectIxOffset)
		return std::nullopt;

	const auto planarIx = objectIx - SCameraAppSceneDefaults::CameraObjectIxOffset;
	if (planarIx >= m_planarProjections.size())
		return std::nullopt;
	return planarIx;
}

bool App::tryBuildManipulableObjectContext(const uint32_t objectIx, SManipulableObjectContext& outContext) const
{
	outContext = {};
	outContext.objectIx = objectIx;

	if (objectIx == SCameraAppSceneDefaults::ModelObjectIx)
	{
		const auto modelTransform = buildModelManipulationTransform(m_sceneInteraction.model);
		outContext.kind = SceneManipulatedObjectKind::Model;
		outContext.label = "Model";
		outContext.transform = modelTransform;
		outContext.worldPosition = extractWorldPosition(modelTransform);
		return true;
	}

	if (isManipulableObjectFollowTarget(objectIx))
	{
		outContext.kind = SceneManipulatedObjectKind::FollowTarget;
		outContext.label = m_sceneInteraction.followTarget.getIdentifier();
		outContext.transform = buildFollowTargetTransform(m_sceneInteraction.followTarget);
		outContext.worldPosition = buildFollowTargetWorldPosition(m_sceneInteraction.followTarget);
		return true;
	}

	const auto planarIx = getManipulableObjectPlanarIx(objectIx);
	if (!planarIx.has_value())
		return false;

	auto* camera = m_planarProjections[planarIx.value()] ? m_planarProjections[planarIx.value()]->getCamera() : nullptr;
	if (!camera)
		return false;

	outContext.kind = SceneManipulatedObjectKind::Camera;
	outContext.planarIx = planarIx;
	outContext.camera = camera;
	outContext.label = std::string(getCameraTypeLabel(camera)) + " Camera";
	outContext.transform = buildCameraManipulationTransform(*camera);
	outContext.worldPosition = buildCameraWorldPosition(*camera);
	return true;
}

bool App::tryBuildActiveManipulatedObjectContext(SManipulableObjectContext& outContext) const
{
	return tryBuildManipulableObjectContext(getManipulatedObjectIx(), outContext);
}

uint32_t App::getManipulatedObjectIx() const
{
	switch (m_sceneInteraction.manipulatedObjectKind)
	{
		case SceneManipulatedObjectKind::Model:
			return SCameraAppSceneDefaults::ModelObjectIx;
		case SceneManipulatedObjectKind::FollowTarget:
			return SCameraAppSceneDefaults::FollowTargetObjectIx;
		case SceneManipulatedObjectKind::Camera:
		default:
			return m_sceneInteraction.boundPlanarCameraIxToManipulate.has_value() ?
				(m_sceneInteraction.boundPlanarCameraIxToManipulate.value() + SCameraAppSceneDefaults::CameraObjectIxOffset) :
				SCameraAppSceneDefaults::ModelObjectIx;
	}
}

void App::bindManipulatedModel()
{
	m_sceneInteraction.manipulatedObjectKind = SceneManipulatedObjectKind::Model;
	m_sceneInteraction.boundCameraToManipulate = nullptr;
	m_sceneInteraction.boundPlanarCameraIxToManipulate = std::nullopt;
}

void App::bindManipulatedFollowTarget()
{
	m_sceneInteraction.manipulatedObjectKind = SceneManipulatedObjectKind::FollowTarget;
	m_sceneInteraction.boundCameraToManipulate = nullptr;
	m_sceneInteraction.boundPlanarCameraIxToManipulate = std::nullopt;
}

void App::bindManipulatedCamera(const uint32_t planarIx)
{
	if (planarIx >= m_planarProjections.size())
	{
		bindManipulatedModel();
		return;
	}

	auto* camera = m_planarProjections[planarIx] ? m_planarProjections[planarIx]->getCamera() : nullptr;
	if (!camera)
	{
		bindManipulatedModel();
		return;
	}

	m_sceneInteraction.manipulatedObjectKind = SceneManipulatedObjectKind::Camera;
	m_sceneInteraction.boundPlanarCameraIxToManipulate = planarIx;
	m_sceneInteraction.boundCameraToManipulate = smart_refctd_ptr<ICamera>(camera);
}

void App::bindManipulableObject(const SManipulableObjectContext& context)
{
	switch (context.kind)
	{
		case SceneManipulatedObjectKind::Model:
			bindManipulatedModel();
			break;
		case SceneManipulatedObjectKind::FollowTarget:
			bindManipulatedFollowTarget();
			break;
		case SceneManipulatedObjectKind::Camera:
			if (context.planarIx.has_value())
				bindManipulatedCamera(context.planarIx.value());
			else
				bindManipulatedModel();
			break;
	}
}

void App::bindManipulatedObjectByIx(const uint32_t objectIx)
{
	SManipulableObjectContext context = {};
	if (!tryBuildManipulableObjectContext(objectIx, context))
	{
		bindManipulatedModel();
		return;
	}

	bindManipulableObject(context);
}

std::string App::getManipulableObjectLabel(const uint32_t objectIx) const
{
	SManipulableObjectContext context = {};
	if (!tryBuildManipulableObjectContext(objectIx, context))
		return "Unknown";
	return context.label;
}

float32_t4x4 App::getManipulableObjectTransform(const uint32_t objectIx) const
{
	SManipulableObjectContext context = {};
	if (!tryBuildManipulableObjectContext(objectIx, context))
		return float32_t4x4(1.0f);
	return context.transform;
}

float32_t3 App::getManipulableObjectWorldPosition(const uint32_t objectIx) const
{
	SManipulableObjectContext context = {};
	if (!tryBuildManipulableObjectContext(objectIx, context))
		return float32_t3(0.0f);
	return context.worldPosition;
}

void App::applyManipulableObjectTransform(const SManipulableObjectContext& context, const float64_t4x4& transform)
{
	switch (context.kind)
	{
		case SceneManipulatedObjectKind::Camera:
			if (context.camera)
			{
				nbl::core::applyReferenceFrameToCamera(context.camera, transform);
				if (context.planarIx.has_value())
					refreshFollowOffsetConfigForPlanar(context.planarIx.value());
			}
			break;
		case SceneManipulatedObjectKind::FollowTarget:
			setFollowTargetTransform(transform);
			applyFollowToConfiguredCameras();
			break;
		case SceneManipulatedObjectKind::Model:
			m_sceneInteraction.model = float32_t3x4(hlsl::transpose(getCastedMatrix<float32_t>(transform)));
			break;
	}
}
