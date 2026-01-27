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
		static const video::IGPURenderpass::SCreationParams::SSubpassDependency Dependencies[3];

		struct SCachedCreationParams
		{
			core::smart_refctd_ptr<ui::IWindowManager> winMgr = nullptr;
			// for the UI, 1080p with 50% scaling
			hlsl::uint16_t2 minResolution = {1248,688};
		};
		struct SCreationParams : IPresenter::SCachedCreationParams, SCachedCreationParams
		{
			inline operator bool() const {return assMan && winMgr && api && callback;}
			
			core::smart_refctd_ptr<video::CVulkanConnection> api = {};
			core::smart_refctd_ptr<ui::IWindow::IEventCallback> callback = {};
			std::string_view initialWindowCaption = "";
		};
		static core::smart_refctd_ptr<CWindowPresenter> create(SCreationParams&& _params);

		//
		inline const video::ISurface* getSurface() const {return m_construction.surface->getSurface();}

		//
		inline const SCachedCreationParams& getCreationParams() const {return m_creation;}

		//
		inline ui::ICursorControl* getCursorControl() const {return m_construction.cursorControl;}

		//
		inline const video::IGPURenderpass* getRenderpass() const {return getSwapchainResources()->getRenderpass();}

		//
		bool irrecoverable() const {return m_construction.surface->irrecoverable() || !m_construction.surface->isWindowOpen();}

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
		struct SConstructorParams : IPresenter::SCachedCreationParams, SCachedCreationParams, SCachedConstructionParams
		{
		};
		inline CWindowPresenter(SConstructorParams&& _params) : IPresenter(std::move(_params)), m_creation(std::move(_params)), m_construction(std::move(_params)), m_pushConstants({}) {}
		//
		bool init_impl(CRenderer* renderer) override;

		//
		clock_t::time_point acquire_impl(const CSession* session, video::ISemaphore::SWaitInfo* p_currentImageAcquire) override;
		bool beginRenderpass_impl() override;
		inline bool present(const video::IQueue::SSubmitInfo::SSemaphoreInfo& readyToPresent) override
		{
			return m_construction.surface->present(m_currentImageIndex,{&readyToPresent,1});
		}
		
		inline video::ISurface* getSurface() {return m_construction.surface->getSurface();}

		inline swapchain_resources_t* getSwapchainResources() {return static_cast<swapchain_resources_t*>(m_construction.surface->getSwapchainResources());}
		inline const swapchain_resources_t* getSwapchainResources() const {return static_cast<const swapchain_resources_t*>(m_construction.surface->getSwapchainResources());}

		SCachedCreationParams m_creation;
		SCachedConstructionParams m_construction;
		core::smart_refctd_ptr<video::IGPUGraphicsPipeline> m_present;
		SDefaultResolvePushConstants m_pushConstants;
		uint8_t m_currentImageIndex = ~0u;
};

}
#endif
