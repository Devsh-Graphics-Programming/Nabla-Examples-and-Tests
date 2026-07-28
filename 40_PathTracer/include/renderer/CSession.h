// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SESSION_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SESSION_H_INCLUDED_


#include "io/CSceneLoader.h"

#include "renderer/shaders/session.hlsl"
#include "renderer/shaders/pathtrace/push_constants.hlsl"


namespace nbl::this_example
{
class CScene;

class CSession final : public core::IReferenceCounted
{
	public:
		using sensor_t = CSceneLoader::SLoadResult::SSensor;
		using sensor_type_e = sensor_t::SMutable::Raygen::Type;
		enum class RenderMode : uint8_t
		{
			Previs,
			Beauty,
			Debug,
			Count
		};
		enum class MisMode : uint8_t
		{
			NEEOnly,
			BxDFOnly,
			Both,
			Count
		};

		// OBB is the paper's method, the three Tri* are the prior-art baseline. All Tri* share one
		// per-triangle light tree and differ only in the shader, so only switching to or from OBB rebuilds.
		enum class LightSampler : uint8_t
		{
			OBB,
			TriUniform,
			TriArvo,
			TriProjected,
			Count
		};

		enum class BeautyVariant : uint8_t
		{
			NEEOnly_Alias,      // NBL_MIS_MODE=0
			BxDFOnly,           // NBL_MIS_MODE=1
			Both_Tree,          // NBL_NEE_USE_ALIAS=0
			NEEOnly_Tree,       // NBL_MIS_MODE=0, NBL_NEE_USE_ALIAS=0
			TriUniform_Both,    // NBL_NEE_LEAF_MODE=1
			TriUniform_NEEOnly, // NBL_NEE_LEAF_MODE=1, NBL_MIS_MODE=0
			TriArvo_Both,       // NBL_NEE_LEAF_MODE=2
			TriArvo_NEEOnly,    // NBL_NEE_LEAF_MODE=2, NBL_MIS_MODE=0
			TriProjected_Both,  // NBL_NEE_LEAF_MODE=3
			TriProjected_NEEOnly, // NBL_NEE_LEAF_MODE=3, NBL_MIS_MODE=0
			// Triangle + alias-table selection (NBL_NEE_USE_ALIAS=1); the tree variants above are USE_ALIAS=0.
			TriUniform_Alias_Both,      // NBL_NEE_LEAF_MODE=1
			TriUniform_Alias_NEEOnly,   // NBL_NEE_LEAF_MODE=1, NBL_MIS_MODE=0
			TriArvo_Alias_Both,         // NBL_NEE_LEAF_MODE=2
			TriArvo_Alias_NEEOnly,      // NBL_NEE_LEAF_MODE=2, NBL_MIS_MODE=0
			TriProjected_Alias_Both,    // NBL_NEE_LEAF_MODE=3
			TriProjected_Alias_NEEOnly, // NBL_NEE_LEAF_MODE=3, NBL_MIS_MODE=0
			// Raygen only writes per-bounce request slots and differs solely by emitter-ID resolution, so
			// one variant per (leaf granularity, MIS mode). The sampler choice lives in the compute pipeline.
			NEEOnly_Deferred,    // NBL_NEE_DEFERRED=1, NBL_MIS_MODE=0
			TriNEEOnly_Deferred, // NBL_NEE_DEFERRED=1, NBL_MIS_MODE=0, NBL_NEE_LEAF_MODE=1
			Both_Deferred,       // NBL_NEE_DEFERRED=1
			TriBoth_Deferred,    // NBL_NEE_DEFERRED=1, NBL_NEE_LEAF_MODE=1
			Count
		};

		// Maps a triangle sampler + MIS mode onto its precompiled variant. BxDFOnly disables NEE so the
		// leaf sampler is irrelevant there (handled by the caller, which returns BxDFOnly).
		static BeautyVariant triBeautyVariant(const LightSampler sampler, const bool neeOnly, const bool useAlias)
		{
			if (useAlias)
				switch (sampler)
				{
					case LightSampler::TriUniform:   return neeOnly ? BeautyVariant::TriUniform_Alias_NEEOnly   : BeautyVariant::TriUniform_Alias_Both;
					case LightSampler::TriArvo:      return neeOnly ? BeautyVariant::TriArvo_Alias_NEEOnly      : BeautyVariant::TriArvo_Alias_Both;
					default:                         return neeOnly ? BeautyVariant::TriProjected_Alias_NEEOnly : BeautyVariant::TriProjected_Alias_Both;
				}
			switch (sampler)
			{
				case LightSampler::TriUniform:   return neeOnly ? BeautyVariant::TriUniform_NEEOnly   : BeautyVariant::TriUniform_Both;
				case LightSampler::TriArvo:      return neeOnly ? BeautyVariant::TriArvo_NEEOnly      : BeautyVariant::TriArvo_Both;
				default:                         return neeOnly ? BeautyVariant::TriProjected_NEEOnly : BeautyVariant::TriProjected_Both;
			}
		}

		static BeautyVariant beautyVariantFor(const MisMode misMode, const bool useAlias, const LightSampler sampler = LightSampler::OBB, const bool deferredNEE = false)
		{
			// Wavefront NEE: BxDF-only has no NEE so it falls through to the inline variant.
			if (deferredNEE && misMode == MisMode::NEEOnly)
				return (sampler == LightSampler::OBB) ? BeautyVariant::NEEOnly_Deferred : BeautyVariant::TriNEEOnly_Deferred;
			if (deferredNEE && misMode == MisMode::Both)
				return (sampler == LightSampler::OBB) ? BeautyVariant::Both_Deferred : BeautyVariant::TriBoth_Deferred;
			if (sampler != LightSampler::OBB)
			{
				// Triangle baseline selects by alias table or tree descent (useAlias); BxDFOnly has no NEE.
				if (misMode == MisMode::BxDFOnly)
					return BeautyVariant::BxDFOnly;
				return triBeautyVariant(sampler, misMode == MisMode::NEEOnly, useAlias);
			}
			switch (misMode)
			{
				case MisMode::BxDFOnly: return BeautyVariant::BxDFOnly; // no NEE -> alias/tree irrelevant
				case MisMode::NEEOnly:  return useAlias ? BeautyVariant::NEEOnly_Alias : BeautyVariant::NEEOnly_Tree;
				default:                return useAlias ? BeautyVariant::Count : BeautyVariant::Both_Tree; // Both
			}
		}
		struct SCachedCreationParams
		{
			RenderMode mode = RenderMode::Beauty;
		};
		struct SCreationParams : SCachedCreationParams
		{
			inline operator bool() const {return sensor;}

			const sensor_t* sensor;
		};

		//
		bool init(video::SIntendedSubmitInfo& info);

		//
		inline bool isInitialized() const {return bool(m_active.immutables);}

		// heavy VRAM data and data only needed during an active session
		struct SImageWithViews
		{
			inline operator bool() const
			{
				return image && !views.empty() && views.begin()->second;
			}

			inline video::IGPUImageView* getView(const asset::E_FORMAT format) const
			{
				if (const auto found=views.find(format); found!=views.end())
					return found->second.get();
				return nullptr;
			}

			core::smart_refctd_ptr<video::IGPUImage> image = {};
			core::unordered_map<asset::E_FORMAT,core::smart_refctd_ptr<video::IGPUImageView>> views = {};
		};
		struct SActiveResources
		{
			struct SImmutables
			{
				inline operator bool() const
				{
					return bool(scrambleKey) && sampleCount && rwmcCascades && albedo && normal && motion && mask && ds;
				}

				// QUESTION: No idea how to marry RWMC with Temporal Denoise, do we denoise separately per cascade?
				// ANSWER: RWMC relies on many spp, can use denoised/reprojected to confidence measures from other cascades.
				// Shouldn't touch the previous frame, denoiser needs to know what was on screen last frame, only touch current.
				// QUESTION: with temporal denoise do we turn the `sampleCount` into a `sequenceOffset` texutre?
				SImageWithViews scrambleKey = {}, sampleCount = {}, beauty = {}, rwmcCascades = {}, albedo = {}, normal = {}, motion = {}, mask = {};
				// stores all the sensor data required
				core::smart_refctd_ptr<video::IGPUDescriptorSet> ds = {};
			};
			SImmutables immutables = {};
			SSensorDynamics currentSensorState = {}, prevSensorState = {};
		};

		//
		inline const SActiveResources& getActiveResources() const {return m_active;}

		//
		bool reset(const SSensorDynamics& newVal, video::SIntendedSubmitInfo& info);

		//
		bool update(const SSensorDynamics& newVal);

		inline float getProgress() const
		{
			const uint32_t maxSPP = m_active.currentSensorState.maxSPP;
			if (maxSPP == 0)
				return 0.f;
			return std::min(1.f, float(m_accumulatedSpp) / float(maxSPP));
		}

		inline void onFrameRendered(uint16_t sppThisFrame)
		{
			if (m_active.currentSensorState.keepAccumulating)
				m_accumulatedSpp += sppThisFrame;
			else
				m_accumulatedSpp = sppThisFrame;
		}

		//
		inline void deinit()
		{
			m_active = {};
			m_accumulatedSpp = 0;
		}

		//
		struct SConstructionParams : SCachedCreationParams
		{
			core::string name = "TODO from `sensor`";
			core::smart_refctd_ptr<const CScene> scene;
			SResolveConstants initResolveConstants;
			SSensorUniforms uniforms;
			SSensorDynamics initDynamics;
			hlsl::uint16_t2 cropOffsets;
			hlsl::uint16_t2 cropResolution;
			sensor_type_e type;
		};
		inline const SConstructionParams& getConstructionParams() const {return m_params;}

	private:
		friend class CScene;
		inline CSession(SConstructionParams&& _params) : m_params(std::move(_params)) {}

		const SConstructionParams m_params;
		SActiveResources m_active = {};
		uint32_t m_accumulatedSpp = 0;
};

}

//
namespace nbl::system::impl
{
template<>
struct to_string_helper<nbl::this_example::CSession::RenderMode>
{
	private:
		using enum_t = nbl::this_example::CSession::RenderMode;

	public:
		static inline std::string __call(const enum_t value)
		{
			switch (value)
			{
				case enum_t::Beauty:
					return "Beauty";
				case enum_t::Previs:
					return "Previs";
				case enum_t::Debug:
					return "Debug";
				default:
					break;
			}
			return "";
		}
};
}
#endif
