// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_WINDOW_PRESENTER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_BASIC_RWMC_RESOLVER_H_INCLUDED_


#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include "renderer/CRenderer.h"
#include "renderer/present/IPresenter.h"

#include "renderer/shaders/present/push_constants.hlsl"


namespace nbl::this_example
{

class CWindowPresenter : public IPresenter
{
    public:
		using swapchain_resources_t = video::CDefaultSwapchainFramebuffers;
		static const video::IGPURenderpass::SCreationParams::SSubpassDependency dependencies[3];

		struct SCachedCreationParams
		{
			core::smart_refctd_ptr<ui::IWindowManager> winMgr = nullptr;
			system::logger_opt_smart_ptr logger = nullptr;
			// for the UI, 1080p with 50% scaling
			hlsl::uint16_t2 minResolution = {1264,698};
		};
		struct SCreationParams : SCachedCreationParams
		{
			inline operator bool() const {return winMgr && api && callback;}
			
			core::smart_refctd_ptr<video::CVulkanConnection> api = {};
			core::smart_refctd_ptr<ui::IWindow::IEventCallback> callback = {};
			std::string_view initialWindowCaption = "";
		};
		static core::smart_refctd_ptr<CWindowPresenter> create(SCreationParams&& _params);

		//
		inline const video::ISurface* getSurface() const {return m_construction.surface->getSurface();}

		//
		bool init(CRenderer* renderer);

		//
		inline const SCachedCreationParams& getCreationParams() const {return m_creation;}

		//
		inline ui::ICursorControl* getCursorControl() const {return m_construction.cursorControl;}

		//
		inline const video::IGPURenderpass* getRenderpass() const {return getSwapchainResources()->getRenderpass();}

		//
		bool irrecoverable() const {return m_construction.surface->irrecoverable();}

		// returns expected presentation time for frame pacing
		clock_t::time_point acquire(const video::ISwapchain::SAcquireInfo& info, CSession* session) override;
		//
		bool beginRenderpass(video::IGPUCommandBuffer* cb) override;
		//
		bool endRenderpassAndPresent(video::IGPUCommandBuffer* cb, video::ISemaphore* presentBeginSignal) override;

    protected:		
		using surface_t = video::CSimpleResizeSurface<swapchain_resources_t>;
		struct SCachedConstructionParams
		{
			core::smart_refctd_ptr<surface_t> surface;
			ui::IWindow* window;
			ui::ICursorControl* cursorControl;
			hlsl::float64_t2 aspectRatioRange;
			hlsl::uint16_t2 maxResolution;
		};
		struct SConstructorParams : SCachedCreationParams, SCachedConstructionParams
		{
		};
		inline CWindowPresenter(SConstructorParams&& _params) : m_creation(std::move(_params)), m_construction(std::move(_params)), m_pushConstants({}) {}
		
		inline video::ISurface* getSurface() {return m_construction.surface->getSurface();}

		inline swapchain_resources_t* getSwapchainResources() {return static_cast<swapchain_resources_t*>(m_construction.surface->getSwapchainResources());}
		inline const swapchain_resources_t* getSwapchainResources() const {return static_cast<const swapchain_resources_t*>(m_construction.surface->getSwapchainResources());}

		SCachedCreationParams m_creation;
		SCachedConstructionParams m_construction;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_present;
		video::ISimpleManagedSurface::SAcquireResult m_currentImageAcquire = {};
		DefaultResolvePushConstants m_pushConstants;
};

}
#endif
