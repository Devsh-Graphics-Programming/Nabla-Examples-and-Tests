// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_SESSION_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_SESSION_H_INCLUDED_


#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "nbl/builtin/hlsl/rwmc/ResolveParameters.hlsl"

#include "io/CSceneLoader.h"


// TODO: move to HLSL file
namespace nbl::this_example
{
#define MAX_SPP_LOG2 15
NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxSPPLog2 = MAX_SPP_LOG2;
// need to be able to count (represent) both 0 and Max
NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSPP = (0x1u << MaxSPPLog2) - 1;

// We do it so weirdly because https://github.com/microsoft/DirectXShaderCompiler/issues/7131
#define MAX_CASCADE_COUNT_LOG2 3
struct SSensorUniforms
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t ScrambleKeyTextureSize = 512;

#define MAX_PATH_DEPTH_LOG2 7
	NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxCascadeCountLog2 = MAX_CASCADE_COUNT_LOG2;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxPathDepthLog2 = MAX_PATH_DEPTH_LOG2;

	hlsl::float32_t2 rcpPixelSize;
	hlsl::rwmc::SplattingParameters splatting;
	hlsl::uint16_t2 renderSize;
	// bitfield
	uint16_t lastCascadeIndex : MAX_CASCADE_COUNT_LOG2;
	uint16_t unused0 : 13;
	// bitfield
	uint16_t unused1 : 1;
	uint16_t hideEnvironment : 1;
	uint16_t lastPathDepth : MAX_PATH_DEPTH_LOG2;
	uint16_t lastNoRussianRouletteDepth : MAX_PATH_DEPTH_LOG2;
};
#undef MAX_PATH_DEPTH_LOG2

// no uint16_t to be used because its going to be a push constant
struct SSensorDynamics
{
	// assuming input will be ndc = [-1,1]^2 x {-1}
	hlsl::float32_t3x4 ndcToRay;
	hlsl::float32_t tMax;
	// we can adaptively sample per-pixel, but 
	uint32_t minSPP : MAX_SPP_LOG2;
	uint32_t maxSPP : MAX_SPP_LOG2;
	uint32_t unused : 2;
};

// no uint16_t to be used because its going to be a push constant
struct SResolveConstants
{
	struct SProtoRWMC
	{
		hlsl::float32_t initialEmin;
		hlsl::float32_t reciprocalBase;
		hlsl::float32_t reciprocalKappa;
		hlsl::float32_t colorReliabilityFactor;
	} rwmc;
	uint32_t cascadeCount : BOOST_PP_ADD(MAX_CASCADE_COUNT_LOG2,1);
	uint32_t unused : 28;
};
#undef MAX_CASCADE_COUNT_LOG2



struct SensorDSBindings
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t UBO = 0;
	// R32G32_UINT storage texture (can get animated/rearranged)
	NBL_CONSTEXPR_STATIC_INLINE uint32_t ScrambleKey = 1;
	// R16_UINT Per Pixel Sample Count (so don't need to read all RWMC cascades)
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SampleCount = 2;
	// R64_UINT with packing RGB14E6 or RGB14E7 and using rest for spp in the cascade
	NBL_CONSTEXPR_STATIC_INLINE uint32_t RWMCCascades = 3;
	// R10G10B10_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Albedo = 4;
	// R10G10B10_SNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Normal = 5;
	// R10G10B10_SNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Motion = 6;
	// R16_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Mask = 7;
};
}


namespace nbl::this_example
{
class CScene;

class CSession final : public core::IReferenceCounted, public core::InterfaceUnmovable
{
	public:
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

				SImageWithViews scrambleKey = {}, sampleCount = {}, rwmcCascades = {}, albedo = {}, normal = {}, motion = {}, mask = {};
				// stores all the sensor data required
				core::smart_refctd_ptr<video::IGPUDescriptorSet> ds = {};
			};
			SImmutables immutables = {};
			SSensorDynamics prevSensorState = {};
		} m_active = {};
};

}
#endif
