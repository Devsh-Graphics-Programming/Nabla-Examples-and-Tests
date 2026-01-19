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

class CSession final : public core::IReferenceCounted, public core::InterfaceUnmovable
{
	public:
		enum class RenderMode : uint8_t
		{
			Previs,
			Beauty,
			//Albedo,
			//Normal,
			//Motion,
			DebugIDs,
			Count
		};
		using sensor_t = CSceneLoader::SLoadResult::SSensor;
		using sensor_type_e = sensor_t::SMutable::Raygen::Type;

		//
		bool init(video::IGPUCommandBuffer* cb);

		//
		bool reset(const SSensorDynamics& newVal, video::IGPUCommandBuffer* cb);

		//
		inline void deinit() {m_active = {};}

	private:
		friend class CScene;

		struct SConstructionParams
		{
			core::string name = "TODO from `sensor`";
			core::smart_refctd_ptr<const CScene> scene;
			SSensorUniforms uniforms;
			SSensorDynamics initDynamics;
			SResolveConstants initResolveConstants;
			sensor_type_e type;
		};
		inline CSession(SConstructionParams&& _params) : m_params(std::move(_params)) {}

		const SConstructionParams m_params;
		// heavy VRAM data and data only needed during an active session
		struct SActiveResources
		{
			struct SImageWithViews
			{
				inline operator bool() const
				{
					return image && !views.empty() && views.begin()->second;
				}

				core::smart_refctd_ptr<video::IGPUImage> image = {};
				core::unordered_map<asset::E_FORMAT, core::smart_refctd_ptr<video::IGPUImageView>> views = {};
			};
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
				//
			};
			SImmutables immutables = {};
			SSensorDynamics prevSensorState = {};
		} m_active = {};
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
				case enum_t::DebugIDs:
					return "DebugIDs";
				default:
					break;
			}
			return "";
		}
};
}
#endif
