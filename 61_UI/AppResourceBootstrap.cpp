#include "app/App.hpp"
#include "app/AppResourceUtilities.hpp"

bool App::initializeMountedCameraResources(smart_refctd_ptr<ISystem>&& system)
{
	if (!asset_base_t::onAppInitialized(std::move(system)))
		return false;

	nbl::system::mountOptionalSharedEnvmapResources(getCameraAppResourceContext(), m_logger.get());
	return true;
}
