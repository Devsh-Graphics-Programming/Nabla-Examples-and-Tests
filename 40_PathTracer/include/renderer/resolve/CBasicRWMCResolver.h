// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_BASIC_RWMC_RESOLVER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_BASIC_RWMC_RESOLVER_H_INCLUDED_


#include "renderer/CRenderer.h"
#include "renderer/resolve/IResolver.h"
#include "renderer/shaders/resolve/rwmc.hlsl"


namespace nbl::this_example
{

class CBasicRWMCResolver : public IResolver
{
    public:
		enum class AutoExposure : uint8_t
		{
			GeometricAverage,
			Median,
			Count
		};
		enum class Tonemapping : uint8_t
		{
			Reinhard,
			ACES,
			Count
		};

		//
		struct SCachedCreationParams
		{
		};
		struct SCreationParams : SCachedCreationParams
		{
			inline operator bool() const {return renderer;}

			CRenderer* renderer;
		};
		static core::smart_refctd_ptr<CBasicRWMCResolver> create(SCreationParams&& _params);

		//
		inline const SCachedCreationParams& getCreationParams() const { return m_creation; }
		
		struct SCachedConstructionParams
		{
			core::smart_refctd_ptr<video::IGPUPipelineLayout> layout;
			// TODO: autoexposure
			core::smart_refctd_ptr<video::IGPUComputePipeline> lumaMeasure;
			// TODO: motion vector stuff
			// rwmc resolve, apply exposure, interleave into OptiX input formats
			core::smart_refctd_ptr<video::IGPUComputePipeline> rwmcResolve;
			// TODO: OIDN denoise
			// deinterlave from OptiX output format, perform first axis of FFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> postDenoise; // TODO
			// second axis FFT, spectrum multiply and iFFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisBloom; // TODO
			// first axis iFFT, tonemap, encode into final EXR format
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisFFTTonemap; // TODO
			//
			core::smart_refctd_ptr<video::IGPUBuffer> persistentExposureArgs;
			//
			core::smart_refctd_ptr<video::IGPUImageView> bloomKernelSpectrum;
		};
		//
		inline const SCachedConstructionParams& getConstructionParams() const {return m_construction;}

		//
		inline uint64_t computeScratchSize(const CSession* session) const override
		{
			if (!session)
				return 0ull;
			switch (session->getConstructionParams().mode)
			{
				case CSession::RenderMode::Previs: [[fallthrough]];
				case CSession::RenderMode::Debug:
					return 0ull;
				case CSession::RenderMode::Beauty:
					return 0ull; // for now, as long as we blit
				default:
					break;
			}
			assert(false); // unimplemented
			return ~0ull;
		}
		//
		bool resolve(video::IGPUCommandBuffer* cb, video::IGPUBuffer* scratch) override;

    protected:
		struct SConstructorParams : SCachedCreationParams, SCachedConstructionParams
		{
		};
		inline CBasicRWMCResolver(SConstructorParams&& _params) : m_creation(std::move(_params)), m_construction(std::move(_params)) {}

		bool changeSession_impl() override;

		SCachedCreationParams m_creation;
		SCachedConstructionParams m_construction;
};

}
#endif
