// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_


#include "io/CSceneLoader.h"
#include "renderer/CSession.h"
#include "renderer/shaders/scene.hlsl"


namespace nbl::this_example
{
class CRenderer;

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

		//
		inline CRenderer* getRenderer() const {return m_construction.renderer.get();}

		//
		inline video::IGPURayTracingPipeline* getPipeline(const CSession::RenderMode mode) const
		{
			return m_construction.pipelines[static_cast<uint8_t>(mode)].get();
		}

		//
		inline const auto& getSBT(const CSession::RenderMode mode) const {return m_construction.sbts[static_cast<uint8_t>(mode)];}

		//
		inline const video::IGPUDescriptorSet* getDescriptorSet() const {return m_construction.sceneDS->getDescriptorSet();}

		using sensor_t = CSceneLoader::SLoadResult::SSensor;
		//
		inline std::span<const sensor_t> getSensors() const {return m_construction.sensors;}

		//
		core::smart_refctd_ptr<CSession> createSession(const CSession::SCreationParams& sensor);

    protected:
		friend class CRenderer;
		struct SCachedConstructorParams
		{
			//
			hlsl::shapes::AABB<> sceneBound;
			//
			core::vector<sensor_t> sensors;
			// backward link for reference counting
			core::smart_refctd_ptr<CRenderer> renderer;
			// specialized per-scene pipelines
			core::smart_refctd_ptr<video::IGPURayTracingPipeline> pipelines[uint8_t(CSession::RenderMode::Count)];
			//
			video::IGPURayTracingPipeline::SShaderBindingTable sbts[uint8_t(CSession::RenderMode::Count)];
			// descriptor set for a scene shall contain sampled textures and compiled materials
			core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> sceneDS;
			// main TLAS
			core::smart_refctd_ptr<video::IGPUTopLevelAccelerationStructure> TLAS;
		};
		struct SConstructorParams : SCachedCreationParams, SCachedConstructorParams
		{
			// sensor list can be empty, we can just make one up as we go along
			inline operator bool() const
			{
				for (uint8_t i=0; i<static_cast<uint8_t>(CSession::RenderMode::Count); i++)
				if (const auto* pipeline=pipelines[i].get(); !pipeline || !sbts[i].valid(pipeline->getCreationFlags()))
					return false;
				return renderer && sceneDS;
			}
		};
		inline CScene(SConstructorParams&& _params) : m_creation(std::move(_params)), m_construction(std::move(_params)) {}
		virtual inline ~CScene() {}

		SCachedCreationParams m_creation;
		SCachedConstructorParams m_construction;
};

}
#endif
