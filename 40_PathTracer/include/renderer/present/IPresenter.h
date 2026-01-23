// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_I_PRESENTER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_I_PRESENTER_H_INCLUDED_


#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "renderer/shaders/pathtrace/push_constants.hlsl"


namespace nbl::this_example
{

class IPresenter : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		//
		virtual bool irrecoverable() const {return false;}
		// returns expected presentation time for frame pacing
		using clock_t = std::chrono::steady_clock;
		virtual clock_t::time_point acquire(const video::ISwapchain::SAcquireInfo& info, const CSession* background) = 0;
		//
		virtual bool beginRenderpass(video::IGPUCommandBuffer* cb) = 0;
		//
		virtual bool endRenderpassAndPresent(video::IGPUCommandBuffer* cb, video::ISemaphore* presentBeginSignal) = 0;
};

}
#endif
