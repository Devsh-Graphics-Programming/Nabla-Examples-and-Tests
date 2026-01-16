// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_


#include "io/CSceneLoader.h"
#include "renderer/CSession.h"

// TODO: move to HLSL file
namespace nbl::this_example
{
struct SSceneUniforms
{
	struct SIndirectInit
	{
		//
//		bda_t<QuantizedSequence> pQuantizedSequence;
		// because the PDF is rescaled to log2(luma)/log2(Max)*255
		// and you get it out as `exp2(texValue)*factor`
		hlsl::float32_t envmapPDFNormalizationFactor;
		hlsl::float16_t envmapScale;
		uint16_t unused;
	} indirect;
};

struct SceneDSBindings
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t UBO = 0;
	// RGB9E5 post multiplied by a max value
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Envmap = 1;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t TLASes = 2;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Samplers = 3;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SampledImages = 4;
	// UINT8 log2(luma) meant for stochastic descent or querying the PDF of the Warp Map
	NBL_CONSTEXPR_STATIC_INLINE uint32_t EnvmapPDF = 5;
	// R16G16_UNORM or R32G32_SFLOAT (depending on envmap resolution) meant for skipping stochastic descent
	NBL_CONSTEXPR_STATIC_INLINE uint32_t EnvmapWarpMap = 6;
};

struct SceneDSBindingCounts
{
	// Mostly held back by Intel ARC, important to not have more than this many light geometries, can increase to 
	// https://vulkan.gpuinfo.org/displayextensionproperty.php?extensionname=VK_KHR_acceleration_structure&extensionproperty=maxDescriptorSetUpdateAfterBindAccelerationStructures&platform=all
	// https://vulkan.gpuinfo.org/displayextensionproperty.php?extensionname=VK_KHR_acceleration_structure&extensionproperty=maxPerStageDescriptorUpdateAfterBindAccelerationStructures&platform=all
	NBL_CONSTEXPR_STATIC_INLINE uint32_t TLASes = 65535;
	// Reasonable combo (esp if we implement a cache over the DS)
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Samplers = 128;
	// Spec mandated minimum
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SampledImages = 500000;
};
}

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
