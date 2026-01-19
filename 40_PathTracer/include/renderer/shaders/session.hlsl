#ifndef _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_


#include "renderer/shaders/rwmc.hlsl"


namespace nbl::this_example
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

// no uint16_t to be used because its going to be a push constant
struct SSensorDynamics
{
	// assuming input will be ndc = [-1,1]^2 x {-1}
	hlsl::float32_t3x4 ndcToRay;
	hlsl::float32_t tMax;
	// we can adaptively sample per-pixel, but 
	uint32_t minSPP : MAX_SPP_LOG2;
	uint32_t maxSPP : MAX_SPP_LOG2;
	uint32_t unused : BOOST_PP_SUB(32,BOOST_PP_MUL(MAX_SPP_LOG2,2));
};
#undef MAX_SPP_LOG2


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
	// R10G10B10_SNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Normal = 6;
	// R10G10B10_SNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Motion = 7;
	// R16_UNORM
	NBL_CONSTEXPR_STATIC_INLINE uint32_t Mask = 8;
};
}

#endif  // _NBL_THIS_EXAMPLE_SESSION_HLSL_INCLUDED_
