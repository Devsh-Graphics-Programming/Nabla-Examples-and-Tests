#ifndef _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_


#include "renderer/shaders/common.hlsl"
#include "renderer/shaders/resolve/rwmc.hlsl"


namespace nbl
{
namespace this_example
{
// [0].xyz for subpixel jitter and stochastic opacity anyhit, [1].xy for DoF and motion Blur
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t PrimaryRayRandTripletsUsed = 2;
// [0].xyz for BRDF Lobe sampling, then reuse [0].z for Russian Roulette, [1].xyz for BTDF Lobe sampling and [1].z for RIS lobe resampling, [2].xyz for NEE
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint16_t RandDimTriplesPerDepth = 3;

struct SSensorUniforms
{
	NBL_CONSTEXPR_STATIC_INLINE uint16_t ScrambleKeyTextureSize = 512;
	NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxCascadeCountLog2 = MAX_CASCADE_COUNT_LOG2;

	hlsl::float32_t2 rcpPixelSize;
	hlsl::rwmc::SPackedSplattingParameters splatting;
	hlsl::uint16_t2 renderSize;
	// bitfield
	uint16_t lastPathDepth : MAX_PATH_DEPTH_LOG2;
	uint16_t lastNoRussianRouletteDepth : MAX_PATH_DEPTH_LOG2;
	uint16_t lastCascadeIndex : MAX_CASCADE_COUNT_LOG2;
	uint16_t unused0 : 12; //BOOST_PP_SUB(15, BOOST_PP_ADD(BOOST_PP_MUL(MAX_PATH_DEPTH_LOG2, 2), MAX_CASCADE_COUNT_LOG2));
	uint16_t hideEnvironment : 1;
};

struct SensorDSBindings
{
	NBL_CONSTEXPR_STATIC_INLINE uint32_t UBO = 0;
	// R32G32_UINT storage texture (can get animated/rearranged)
	NBL_CONSTEXPR_STATIC_INLINE uint32_t ScrambleKey = 1;
	// R16_UINT Per Pixel Sample Count (so don't need to read all RWMC cascades)
	NBL_CONSTEXPR_STATIC_INLINE uint32_t SampleCount = 2;
	// R64_UINT with packing RGB14E6 or RGB14E7 and using rest for spp in the cascade
	NBL_CONSTEXPR_STATIC_INLINE uint32_t RWMCCascades = 3;
	// RGB5E9
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Beauty = 4;
	// RGBA16F
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Albedo = 5;
	// RGBA16F
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Normal = 6;
	// modified R10G10B10_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Motion = 7;
	// R16_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Mask = 8;
	//
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Samplers = 9;
	//
	NBL_CONSTEXPR_STATIC_INLINE uint32_t AsSampledImages = 10;

	enum class SampledImageIndex : uint16_t
	{
		ScrambleKey = ScrambleKey-ScrambleKey,
		SampleCount = SampleCount-ScrambleKey,
		RWMCCascades = RWMCCascades-ScrambleKey,
		Beauty = Beauty-ScrambleKey,
		Albedo = Albedo-ScrambleKey,
		Normal = Normal-ScrambleKey,
		Motion = Motion-ScrambleKey,
		Mask = Mask-ScrambleKey,
		Count
	};
};

struct SensorDSBindingCounts
{
	//
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Samplers = 1;
	NBL_CONSTEXPR_STATIC_INLINE uint32_t AsSampledImages = SensorDSBindings::Samplers-SensorDSBindings::ScrambleKey;
};


#ifdef __HLSL_VERSION
[[vk::binding(SensorDSBindings::UBO,SessionDSIndex)]] ConstantBuffer<SSensorUniforms> gSensor;
[[vk::binding(SensorDSBindings::ScrambleKey,SessionDSIndex)]] RWTexture2DArray<uint32_t2> gScrambleKey;
// could be uint16_t were it not for "Expected Sampled Type to be a 32-bit int, 64-bit int or 32-bit float scalar type for Vulkan environment"
[[vk::binding(SensorDSBindings::SampleCount,SessionDSIndex)]] RWTexture2DArray<uint32_t> gSampleCount;
[[vk::binding(SensorDSBindings::RWMCCascades,SessionDSIndex)]] RWTexture2DArray<uint32_t2> gRWMCCascades;
[[vk::binding(SensorDSBindings::Beauty,SessionDSIndex)]] RWTexture2DArray<uint32_t> gBeauty;
[[vk::binding(SensorDSBindings::Albedo,SessionDSIndex)]] RWTexture2DArray<float32_t4> gAlbedo;
// thse two are snorm but stored as unorm, care needs to be taken to map:
// [-1,1] <-> [0,1] but with 0 being exactly representable, so really [-1,1] <-> [1/1023,1]
// Requires x*1022.f/2046.f+1024.f/2046.f shift/adjust for accumulation and storage
// Then to decode back into [-1,1] need max(y*2046.f/1022.f-1024.f/1022.f,-1) = x
[[vk::binding(SensorDSBindings::Normal,SessionDSIndex)]] RWTexture2DArray<float32_t4> gNormal;
// TODO: motion confidence mask
[[vk::binding(SensorDSBindings::Motion,SessionDSIndex)]] RWTexture2DArray<float32_t2> gMotion;
[[vk::binding(SensorDSBindings::Mask,SessionDSIndex)]] RWTexture2DArray<float32_t1> gMask;
//
[[vk::binding(SensorDSBindings::Samplers,SessionDSIndex)]] SamplerState gSensorSamplers[SensorDSBindingCounts::Samplers];
//
[[vk::binding(SensorDSBindings::AsSampledImages,SessionDSIndex)]] Texture2DArray<float32_t4> gSensorTextures[SensorDSBindingCounts::AsSampledImages];

// For our generic accumulators
// RWMC cascades and Beauty need special treatment
DEFINE_TEXTURE_ACCESSOR(gAlbedo);
DEFINE_TEXTURE_ACCESSOR(gNormal);
DEFINE_TEXTURE_ACCESSOR(gMotion);
DEFINE_TEXTURE_ACCESSOR(gMask);
#endif
}
}
#endif  // _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
