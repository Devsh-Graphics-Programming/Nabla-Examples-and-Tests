// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SCENE_H_INCLUDED_


#include "io/CSceneLoader.h"
#include "renderer/CLightTree.h"
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
		inline video::IGPURayTracingPipeline* getPipeline(const CSession::RenderMode mode, const CSession::MisMode misMode = CSession::MisMode::Both, const bool useAlias = true) const
		{
			if (mode == CSession::RenderMode::Beauty)
			{
				const auto variant = CSession::beautyVariantFor(misMode, useAlias);
				if (variant != CSession::BeautyVariant::Count)
					return m_construction.beautyVariantPipelines[static_cast<uint8_t>(variant)].get();
			}
			return m_construction.pipelines[static_cast<uint8_t>(mode)].get();
		}

		//
		inline const auto& getSBT(const CSession::RenderMode mode, const CSession::MisMode misMode = CSession::MisMode::Both, const bool useAlias = true) const
		{
			if (mode == CSession::RenderMode::Beauty)
			{
				const auto variant = CSession::beautyVariantFor(misMode, useAlias);
				if (variant != CSession::BeautyVariant::Count)
					return m_construction.beautyVariantSbts[static_cast<uint8_t>(variant)];
			}
			return m_construction.sbts[static_cast<uint8_t>(mode)];
		}

		//
		inline const video::IGPUDescriptorSet* getDescriptorSet() const {return m_construction.sceneDS->getDescriptorSet();}

		using sensor_t = CSceneLoader::SLoadResult::SSensor;
		//
		inline std::span<const sensor_t> getSensors() const {return m_construction.sensors;}

		//
		inline const SLightTree& getLightTree() const {return m_construction.lightTree;}

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
			// Beauty (NBL_MIS_MODE, NBL_NEE_USE_ALIAS) variants, distinct SPIR-V each, indexed by
			// CSession::BeautyVariant. The default (Both + alias) is the regular pipelines/sbts[Beauty]
			// slot, so these cover only the non-default combos (BeautyVariant::Count entries).
			core::smart_refctd_ptr<video::IGPURayTracingPipeline> beautyVariantPipelines[uint8_t(CSession::BeautyVariant::Count)];
			//
			video::IGPURayTracingPipeline::SShaderBindingTable beautyVariantSbts[uint8_t(CSession::BeautyVariant::Count)];
			// descriptor set for a scene shall contain sampled textures and compiled materials
			core::smart_refctd_ptr<video::SubAllocatedDescriptorSet> sceneDS;
			// main TLAS
			core::smart_refctd_ptr<video::IGPUTopLevelAccelerationStructure> TLAS;
			// CPU-built light tree (BVH2 over emitter instance AABBs)
			SLightTree lightTree;
			// device-local buffer of LightcutTreePackedWideNode (32 B each); BDA in the scene UBO
			core::smart_refctd_ptr<video::IGPUBuffer> lightTreeNodes;
			// device-local buffer of LightcutTreePackedLeaf (32 B each); BDA in the scene UBO
			core::smart_refctd_ptr<video::IGPUBuffer> lightTreeLeaves;
			// device-local buffer of SEmitterGPU (radiance etc.), one entry per emitter; BDA in the UBO
			core::smart_refctd_ptr<video::IGPUBuffer> emitters;
			// device-local buffer of uint32_t mapping emitterID -> heap leaf index; BDA in the UBO
			core::smart_refctd_ptr<video::IGPUBuffer> emitterToLeafIdx;
			// device-local uint32_t-per-geometry buffer: instancedGeometryID (= instanceCustomIndex +
			// GeometryIndex()) -> emitterID (NonEmitterCustomIndex when non-emissive); BDA in the UBO.
			core::smart_refctd_ptr<video::IGPUBuffer> instancedGeometryToEmitter;
			// Power-only global alias table for NEE emitter sampling. 2 buffers:
			// packed words (uint32 each) + per-bin pdf (float each), both sized aliasTableSize.
			core::smart_refctd_ptr<video::IGPUBuffer> aliasEntries;
			core::smart_refctd_ptr<video::IGPUBuffer> aliasPdf;
			// Per-internal-node alias tables for the descent's early-stop. Single buffer holding
			// 4 sections back-to-back (each entry 4 B): offsets[N+1], leafBases[N], entries[E],
			// pdfs[E], where N = numInternalNodes (= lightTreeFirstLeafIndex) and E =
			// subtreeAliasTotalEntries. Section bases are derived in the shader from those two scalars.
			core::smart_refctd_ptr<video::IGPUBuffer> subtreeAlias;
			// host-coherent 32 B buffer holding SDebugProbe; CPU writes update the next frame's shader read.
			core::smart_refctd_ptr<video::IGPUBuffer> debugProbe;
			// host-coherent float[numEmittersActual] of per-emitter NEE backward pdfs
			// against the current debug probe. Recomputed CPU-side on every setProbe()
			// call (probe moves at gizmo speed, recomputation is cheap). The debug
			// shader / overlay just RawBufferLoad<float>(addr + emitterID * 4) instead
			// of running the descent per pixel, eliminates the at-density spill.
			core::smart_refctd_ptr<video::IGPUBuffer> probeDebugPdfs;
			// host-coherent float[lightTree.nodes.size()] of per-heap-node cumulative
			// descent probability against the current probe; debug viz tints clusters by it.
			core::smart_refctd_ptr<video::IGPUBuffer> nodePdfs;
			// device-local float[numEmittersActual] of per-emitter quantization quality:
			// max-axis ratio of decoded-quantized child extent to precise leaf extent at the
			// emitter's leaf-parent wide-node. 1.0 = exact, >1.0 = inflated (the value the
			// descent's weight evaluator sees vs. truth). Used by debug.hlsl to tint badly-
			// quantized lights so they pop out visually.
			core::smart_refctd_ptr<video::IGPUBuffer> quantQuality;
		};
		struct SConstructorParams : SCachedCreationParams, SCachedConstructorParams
		{
			// sensor list can be empty, we can just make one up as we go along
			inline operator bool() const
			{
				for (uint8_t i=0; i<static_cast<uint8_t>(CSession::RenderMode::Count); i++)
				if (const auto* pipeline=pipelines[i].get(); !pipeline || !sbts[i].valid(pipeline->getCreationFlags()))
					return false;
				for (uint8_t i=0; i<static_cast<uint8_t>(CSession::BeautyVariant::Count); i++)
				if (const auto* pipeline=beautyVariantPipelines[i].get(); !pipeline || !beautyVariantSbts[i].valid(pipeline->getCreationFlags()))
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
