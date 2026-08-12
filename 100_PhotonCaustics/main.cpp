// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "common.hpp"

#include "nbl/this_example/builtin/build/spirv/keys.hpp"
#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

class PhotonCausticsApp final : public SimpleWindowedApplication, public BuiltinResourcesApplication
{
	using device_base_t = SimpleWindowedApplication;
	using asset_base_t = BuiltinResourcesApplication;

	constexpr static inline uint32_t WIN_W = 1280;
	constexpr static inline uint32_t WIN_H = 720;
	constexpr static inline uint32_t MaxFramesInFlight = 3;
	constexpr static inline uint8_t MaxUITextureCount = 1;

public:
	inline PhotonCausticsApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD)
		:IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD)
	{
	}


	inline SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		auto retVal = device_base_t::getRequiredDeviceFeatures();
		retVal.rayTracingPipeline = true;
		retVal.accelerationStructure = true;
		return retVal;
	}

	inline core::vector<queue_req_t> getQueueRequirements() const override
	{
		auto reqs = device_base_t::getQueueRequirements();
		reqs.front().requiredFlags |= IQueue::FAMILY_FLAGS::COMPUTE_BIT;
		return reqs;
	}

	inline core::vector<video::SPhysicalDeviceFilter::SurfaceCompatibility> getSurfaces() const override
	{
		if (!m_surface)
		{
			{
				auto windowCallback = core::make_smart_refctd_ptr<CEventCallback>(smart_refctd_ptr(m_inputSystem), smart_refctd_ptr(m_logger));
				IWindow::SCreationParams params = {};
				params.callback = core::make_smart_refctd_ptr<ISimpleManagedSurface::ICallback>();
				params.width = WIN_W;
				params.height = WIN_H;
				params.x = 32;
				params.y = 32;
				params.flags = ui::IWindow::ECF_HIDDEN | IWindow::ECF_BORDERLESS | IWindow::ECF_RESIZABLE;
				params.windowCaption = "PhotonCaustics";
				params.callback = windowCallback;
				const_cast<std::remove_const_t<decltype(m_window)>&>(m_window) = m_winMgr->createWindow(std::move(params));
			}

			auto surface = CSurfaceVulkanWin32::create(smart_refctd_ptr(m_api), smart_refctd_ptr_static_cast<IWindowWin32>(m_window));
			const_cast<std::remove_const_t<decltype(m_surface)>&>(m_surface) = CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>::create(std::move(surface));
		}

		m_logger->log("Creating window and surface for the application!", system::ILogger::ELL_INFO);

		if (m_surface)
			return { {m_surface->getSurface()/*,EQF_NONE*/} };

		return {};
	}

	inline bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// create the input system with a logger, use the rvalue ref? weird way to use 2 constructors and why did we use smart_refctd_ptr on m_logger again? to get rvalue ref? but why tho?
		// cant i just std:: move instead of using smart_refctd_ptr ctor again? make sense that loppfer_opt_xxx needs a r value ref and everyone else too
		m_inputSystem = make_smart_refctd_ptr<InputSystem>(logger_opt_smart_ptr(smart_refctd_ptr(m_logger)));

		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(m_system)))
			return false;

		auto loadPreCompiledShader = [&]<core::StringLiteral ShaderKey>() -> smart_refctd_ptr<IShader>
		{
			IAssetLoader::SAssetLoadParams loadParams = {};
			loadParams.logger = m_logger.get();
			loadParams.workingDirectory = "app_resources"; // kinda like VFS?

			// what and why do we need this_example for?
			auto key = nbl::this_example::builtin::build::get_spirv_key<ShaderKey>(m_device.get());
			auto assetBundle = m_assetMgr->getAsset(key.data(), loadParams);
			const auto assets = assetBundle.getContents();

			if (assets.empty())
				return nullptr;

			auto shader = IAsset::castDown<IShader>(assets[0]);
			if (!shader)
			{
				m_logger->log("Failed to load a precompiled shader.", ILogger::ELL_ERROR);
				return nullptr;
			}
			return shader;
		};

		const auto fragmentShader = loadPreCompiledShader.operator() < "present_frag" > ();
		if (!fragmentShader)
			return logFail("Could not load present fragment shader");

		//----------------------------------------------------------------------------


	}

private:
	smart_refctd_ptr<IWindow> m_window;
	smart_refctd_ptr<CSimpleResizeSurface<ISimpleManagedSurface::ISwapchainResources>> m_surface;
	core::smart_refctd_ptr<InputSystem> m_inputSystem;

	smart_refctd_ptr<ISemaphore> m_semaphore;
	smart_refctd_ptr<IGPUCommandPool> m_cmdPool;
	uint64_t m_realFrameIx = 0;
	std::array<smart_refctd_ptr<IGPUCommandBuffer>, MaxFramesInFlight> m_cmdBufs;
	ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
};

// define an entry point as always!
NBL_MAIN_FUNC(PhotonCausticsApp)