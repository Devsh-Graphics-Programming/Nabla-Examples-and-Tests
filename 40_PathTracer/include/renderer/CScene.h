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

		using sensor_t = CSceneLoader::SLoadResult::SSensor;
		//
		inline std::span<const sensor_t> getSensors() const {return m_construction.sensors;}

		//
		core::smart_refctd_ptr<CSession> createSession(const sensor_t& sensor);

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
