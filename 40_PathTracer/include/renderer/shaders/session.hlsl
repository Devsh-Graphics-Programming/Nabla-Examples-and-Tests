#ifndef _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_


#include "renderer/shaders/resolve/rwmc.hlsl"


namespace nbl
{
namespace this_example
{
#define MAX_SPP_LOG2 15
NBL_CONSTEXPR_STATIC_INLINE uint16_t MaxSPPLog2 = MAX_SPP_LOG2;
// need to be able to count (represent) both 0 and Max
NBL_CONSTEXPR_STATIC_INLINE uint32_t MaxSPP = (0x1u << MaxSPPLog2) - 1;

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
	uint16_t unused0 : BOOST_PP_SUB(16,MAX_CASCADE_COUNT_LOG2);
	// bitfield
	uint16_t unused1 : 1;
	uint16_t hideEnvironment : 1;
	uint16_t lastPathDepth : MAX_PATH_DEPTH_LOG2;
	uint16_t lastNoRussianRouletteDepth : MAX_PATH_DEPTH_LOG2;
};
#undef MAX_PATH_DEPTH_LOG2

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
	// R10G10B10_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Albedo = 5;
	// modified R10G10B10_UNORM
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
	NBL_CONSTEXPR_STATIC_INLINE uint32_t AsSampledImages = hlsl::_static_cast<uint32_t>(SensorDSBindings::SampledImageIndex::Count);
};


#ifdef __HLSL_VERSION
[[vk::binding(SensorDSBindings::UBO,SessionDSIndex)]] ConstantBuffer<SSensorUniforms> gSensor;
// could be uint32_t2
[[vk::binding(SensorDSBindings::ScrambleKey,SessionDSIndex)]] RWTexture2DArray<uint32_t4> gScrambleKey;
// could be uint32_t or even uint16_t
[[vk::binding(SensorDSBindings::SampleCount,SessionDSIndex)]] RWTexture2DArray<uint32_t4> gSampleCount;
// could be uint32_t2
[[vk::binding(SensorDSBindings::RWMCCascades,SessionDSIndex)]] RWTexture2DArray<uint32_t4> gRWMCCascades;
// could be uint32_t
[[vk::binding(SensorDSBindings::Beauty,SessionDSIndex)]] RWTexture2DArray<uint32_t4> gBeauty;
[[vk::binding(SensorDSBindings::Albedo,SessionDSIndex)]] RWTexture2DArray<float32_t4> gAlbedo;
// thse two are snorm but stored as unorm, care needs to be taken to map:
// [-1,1] <-> [0,1] but with 0 being exactly representable, so really [-1,1] <-> [1/1023,1]
// Requires x*1022.f/2046.f+1024.f/2046.f shift/adjust for accumulation and storage
// Then to decode back into [-1,1] need max(y*2046.f/1022.f-1024.f/1022.f,-1) = x
[[vk::binding(SensorDSBindings::Normal,SessionDSIndex)]] RWTexture2DArray<float32_t4> gNormal;
[[vk::binding(SensorDSBindings::Motion,SessionDSIndex)]] RWTexture2DArray<float32_t4> gMotion;
// could be float32_t
[[vk::binding(SensorDSBindings::Mask,SessionDSIndex)]] RWTexture2DArray<float32_t4> gMask;
//
[[vk::binding(SensorDSBindings::Samplers,SessionDSIndex)]] SamplerState gSensorSamplers[SensorDSBindingCounts::Samplers];
//
[[vk::binding(SensorDSBindings::AsSampledImages,SessionDSIndex)]] Texture2DArray<float32_t4> gSensorTextures[SensorDSBindingCounts::AsSampledImages];
#endif
}
}
#endif  // _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
