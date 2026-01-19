#ifndef _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_


#include "renderer/shaders/common.hlsl"


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

#endif  // _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_
