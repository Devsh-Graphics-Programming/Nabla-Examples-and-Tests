// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_


#include "renderer/CScene.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include <thread>
#include <future>
#include <filesystem>


namespace nbl::this_example
{

class CRenderer : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
		enum class RenderMode : uint8_t
		{
			Previs,
			Beauty//,
			//Albedo,
			//Normal,
			//Motion
		};
		// TODO: move this somewhere else
		struct DenoiserArgs
		{
			std::filesystem::path bloomFilePath;
			float bloomScale = 0.0f;
			float bloomIntensity = 0.0f;
			std::string tonemapperArgs = "";
		};

		//
		constexpr static video::SPhysicalDeviceFeatures RequiredDeviceFeatures()
		{
			video::SPhysicalDeviceFeatures retval = {};
			retval.rayTracingPipeline = true;
			retval.accelerationStructure = true;
			return retval;
		}
		//
		constexpr static video::SPhysicalDeviceFeatures PreferredDeviceFeatures()
		{
			auto retval = RequiredDeviceFeatures();
			retval.accelerationStructureHostCommands = true;
			return retval;
		}

		struct SCachedCreationParams
		{
			//! Brief guideline to good path depth limits
			// Want to see stuff with indirect lighting on the other side of a pane of glass
			// 5 = glass frontface->glass backface->diffuse surface->diffuse surface->light
			// Want to see through a glass box, vase, or office 
			// 7 = glass frontface->glass backface->glass frontface->glass backface->diffuse surface->diffuse surface->light
			// pick higher numbers for better GI and less bias
			static inline constexpr uint32_t DefaultPathDepth = 8u;
			// TODO: Upload only a subsection of the sample sequence to the GPU, so we can use more samples without trashing VRAM
			static inline constexpr uint32_t MaxFreeviewSamples = 0x10000u;

			inline operator bool() const
			{
				if (!graphicsQueue || !computeQueue || !uploadQueue)
					return false;
				if (!utilities)
					return false;
				if (graphicsQueue->getOriginDevice()!=utilities->getLogicalDevice())
					return false;
				if (computeQueue->getOriginDevice()!=utilities->getLogicalDevice())
					return false;
				if (uploadQueue->getOriginDevice()!=utilities->getLogicalDevice())
					return false;
				return true;
			}

			video::IQueue* graphicsQueue = nullptr;
			video::IQueue* computeQueue = nullptr;
			video::IQueue* uploadQueue = nullptr;
			//
			core::smart_refctd_ptr<video::IUtilities> utilities = nullptr;
		};
		static core::smart_refctd_ptr<CRenderer> create(SCachedCreationParams&& params);

		//
		inline video::ILogicalDevice* getDevice() { return m_params.utilities->getLogicalDevice(); }

    protected:
		struct SConstructorParams : SCachedCreationParams
		{
			core::smart_refctd_ptr<video::CAssetConverter> converter;

			// per pipeline UBO, with fast updates
			core::smart_refctd_ptr<video::IGPUDescriptorSet> uboDS;
			// descriptor set for a scene shall contain sampled textures and compiled materials
			core::smart_refctd_ptr<video::IGPUDescriptorSet> sceneDS;

			// rendering pipelines
			core::smart_refctd_ptr<video::IGPURayTracingPipeline> preVis;
			core::smart_refctd_ptr<video::IGPURayTracingPipeline> pathTracing;

			// rwmc resolve, autoexposure first pass
			core::smart_refctd_ptr<video::IGPUComputePipeline> rwmcResolveAndLumaMeasure; // TODO: autoexposure, and first axis of FFT
			// TODO: motion vector stuff
			// compute and apply exposure, interleave into OptiX input formats, etc.
			core::smart_refctd_ptr<video::IGPUComputePipeline> preOptiXDenoise; // TODO
			// TODO: OIDN denoise
			// deinterlave from OptiX output format, perform first axis of FFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> postOptiXDenoise; // TODO
			// second axis FFT, spectrum multiply and iFFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisBloom; // TODO
			// first axis iFFT, tonemap, encode into final EXR format
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisFFTTonemap; // TODO

			// Present
			core::smart_refctd_ptr<video::IGPURenderpass> presentRenderpass;
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> regularPresent;
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> cubemapPresent; // TODO
		};
		inline CRenderer(SConstructorParams&& _params) : m_params(std::move(_params)) {}
		virtual inline ~CRenderer() {}

		SConstructorParams m_params;
#if 0	
		// semi persistent data
		nbl::io::path sampleSequenceCachePath;
		struct SampleSequence
		{
			public:
				static inline constexpr auto QuantizedDimensionsBytesize = sizeof(uint64_t);
				SampleSequence() : bufferView() {}

				// one less because first path vertex uses a different sequence 
				static inline uint32_t computeQuantizedDimensions(uint32_t maxPathDepth) {return (maxPathDepth-1)*SAMPLING_STRATEGY_COUNT;}
				nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> createCPUBuffer(uint32_t quantizedDimensions, uint32_t sampleCount);

				// from cache
				void createBufferView(nbl::video::IVideoDriver* driver, nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer>&& buff);
				// regenerate
				nbl::core::smart_refctd_ptr<nbl::asset::ICPUBuffer> createBufferView(nbl::video::IVideoDriver* driver, uint32_t quantizedDimensions, uint32_t sampleCount);

				auto getBufferView() const {return bufferView;}

			private:
				nbl::core::smart_refctd_ptr<nbl::video::IGPUBufferView> bufferView;
		} sampleSequence;
		uint16_t maxPathDepth;
		uint16_t noRussianRouletteDepth : 15;
		uint16_t hideEnvironment : 1;
		uint32_t maxSensorSamples;

		nbl::core::matrix3x4SIMD m_prevView;
		nbl::core::matrix4x3 m_prevCamTform;
		nbl::core::aabbox3df m_sceneBound;
		uint32_t m_framesDispatched;
		float m_maxAreaLightLuma;
		vec2 m_rcpPixelSize;
		uint64_t m_totalRaysCast;
		StaticViewData_t m_staticViewData;
		RaytraceShaderCommonData_t m_raytraceCommonData; 
		
		// Resources used for envmap sampling
		nbl::ext::EnvmapImportanceSampling::EnvmapImportanceSampling m_envMapImportanceSampling;
#endif
};

}

//
namespace nbl::system::impl
{
template<>
struct to_string_helper<nbl::this_example::CRenderer::RenderMode>
{
	private:
		using enum_t = nbl::this_example::CRenderer::RenderMode;

	public:
		static inline std::string __call(const enum_t value)
		{
			switch (value)
			{
				case enum_t::Beauty:
					return "Beauty";
				case enum_t::Previs:
					return "Previs";
				default:
					break;
			}
			return "";
		}
};
}
#endif
