#ifndef _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_


#include "renderer/shaders/common.hlsl"

namespace nbl
{
namespace this_example
{
struct SSceneUniforms
{
	struct SInit
	{
		//
//		bda_t<QuantizedSequence> pQuantizedSequence;
		// because the PDF is rescaled to log2(luma)/log2(Max)*255
		// and you get it out as `exp2(texValue)*factor`
		hlsl::float32_t envmapPDFNormalizationFactor;
		hlsl::float16_t envmapScale;
		uint16_t unused;
	} init;
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

#ifdef __HLSL_VERSION
[[vk::binding(SceneDSBindings::UBO,SceneDSIndex)]] ConstantBuffer<SSceneUniforms> gScene;
// could be float32_t3
[[vk::binding(SceneDSBindings::Envmap,SceneDSIndex)]] [[vk::combinedImageSampler]] Texture2D<float32_t4> gEnvmap;
[[vk::binding(SceneDSBindings::Envmap,SceneDSIndex)]] [[vk::combinedImageSampler]] SamplerState gEnvmapSampler;
[[vk::binding(SceneDSBindings::TLASes,SceneDSIndex)]] RaytracingAccelerationStructure gTLASes[SceneDSBindingCounts::TLASes];
[[vk::binding(SceneDSBindings::Samplers,SceneDSIndex)]] SamplerState gSamplers[SceneDSBindingCounts::Samplers];
[[vk::binding(SceneDSBindings::SampledImages,SceneDSIndex)]] Texture2DArray<float32_t4> gSampledImages[SceneDSBindingCounts::SampledImages];
// could be float32_t
[[vk::binding(SceneDSBindings::EnvmapPDF,SceneDSIndex)]] [[vk::combinedImageSampler]] Texture2D<float32_t4> gEnvmapPDF;
[[vk::binding(SceneDSBindings::EnvmapPDF,SceneDSIndex)]] [[vk::combinedImageSampler]] SamplerState gEnvmapPDFSampler;
// could be float32_t2
[[vk::binding(SceneDSBindings::EnvmapWarpMap,SceneDSIndex)]] [[vk::combinedImageSampler]] Texture2D<float32_t4> gEnvmapWarpMap;
[[vk::binding(SceneDSBindings::EnvmapWarpMap,SceneDSIndex)]] [[vk::combinedImageSampler]] SamplerState gEnvmapWarpMapSampler;
#endif
}
}
#endif  // _NBL_THIS_EXAMPLE_SCENE_HLSL_INCLUDED_
