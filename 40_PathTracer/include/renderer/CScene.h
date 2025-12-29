// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_


#include "io/CSceneLoader.h"


namespace nbl::this_example
{

class CScene : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		struct SCachedCreationParams
		{
		};
		struct SCreationParams : SCachedCreationParams
		{
			CSceneLoader::SLoadResult load = {};
			video::CAssetConverter* converter = nullptr;

			inline operator bool() const
			{
				if (!load)
					return false;
				// converter can be null, we can make a new one
				return true;
			}
		};

		// TODO: figure out whats constant, and whats state that can be passed around
		inline std::span<const CSceneLoader::SLoadResult::SSensor> getSensors() const {return m_params.sensors;}

		// TODO: function to initialize per-sensor stuff

    protected:
		friend class CRenderer;
		struct SConstructorParams : SCachedCreationParams
		{
			// descriptor set for a scene shall contain sampled textures and compiled materials
			core::smart_refctd_ptr<video::IGPUDescriptorSet> sceneDS;

			core::vector<CSceneLoader::SLoadResult::SSensor> sensors;
#if 0
			nbl::core::aabbox3df m_sceneBound;
			float m_maxAreaLightLuma;
			StaticViewData_t m_staticViewData;
			RaytraceShaderCommonData_t m_raytraceCommonData;
			// Resources used for envmap sampling
			nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_finalEnvmap;
#endif
		};
		inline CScene(SConstructorParams&& _params) : m_params(std::move(_params)) {}
		virtual inline ~CScene() {}

		SConstructorParams m_params;
#if 0
		// TODO: sensor stuff
		uint16_t hideEnvironment : 1;
		uint32_t maxSensorSamples;

		uint32_t m_framesDispatched;
		vec2 m_rcpPixelSize;
		uint64_t m_totalRaysCast;
		StaticViewData_t m_staticViewData;
		RaytraceShaderCommonData_t m_raytraceCommonData;

		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_accumulation,m_tonemapOutput;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_albedoAcc,m_albedoRslv;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_normalAcc,m_normalRslv;
		nbl::core::smart_refctd_ptr<nbl::video::IGPUImageView> m_maskAcc;
		
#endif
};

}
#endif
