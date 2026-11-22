#include "app/App.hpp"

bool App::initializeSceneResources()
{
	if (!initializeGeometrySceneResources())
		return false;
	if (!initializeSceneRenderpass())
		return false;
	if (!initializeSpaceEnvironmentResources())
		return false;
	if (!initializeDebugSceneRendererResources())
		return false;
	if (!initializeWindowSceneFramebufferResources())
		return false;

	return true;
}
