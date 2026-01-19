// Copyright (C) 2025-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_C_RENDERER_H_INCLUDED_


#include "renderer/CScene.h"
#include "renderer/CSession.h"

#include "nbl/ext/FullScreenTriangle/FullScreenTriangle.h"

#include <thread>
#include <future>
#include <filesystem>

#include "renderer/shaders/pathtrace/push_constants.hlsl"


namespace nbl::this_example
{

class CRenderer : public core::IReferenceCounted, public core::InterfaceUnmovable
{
    public:
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
#if 0 // see TODO in main.cpp
		constexpr static video::SPhysicalDeviceLimits RequiredDeviceLimits()
		{
			video::SPhysicalDeviceLimits retval = {};
			retval.shaderStorageImageReadWithoutFormat = true;
			return retval;
		}
#endif

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
			// can be null
			system::logger_opt_smart_ptr logger = nullptr;
		};
		struct SCreationParams : SCachedCreationParams
		{
			system::path sampleSequenceCache;
		};
		static core::smart_refctd_ptr<CRenderer> create(SCreationParams&& _params);

		//
		inline const SCachedCreationParams& getCreationParams() const { return m_creation; }
		
		//
		inline video::ILogicalDevice* getDevice() const {return m_creation.utilities->getLogicalDevice();}

		//
		core::smart_refctd_ptr<CScene> createScene(CScene::SCreationParams&& _params);
		
		struct SCachedConstructionParams
		{
			constexpr static inline uint8_t FramesInFlight = 3;

			// per pipeline UBO for other pipelines
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> uboDSLayout;
			// descriptor set for a scene shall contain sampled textures and compiled materials
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> sceneDSLayout;
			// descriptor set for sensors
			core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> sensorDSLayout;

			// TODO
			std::array<core::smart_refctd_ptr<video::IGPURayTracingPipeline>,uint8_t(CSession::RenderMode::Count)> renderingPipelines;

			//
			core::smart_refctd_ptr<video::IGPUCommandBuffer> commandBuffers[FramesInFlight];
		};
		//
		inline const SCachedConstructionParams& getConstructionParams() const {return m_construction;}

    protected:
		struct SConstructorParams : SCachedCreationParams, SCachedConstructionParams
		{

			// Each Atom of the sample sequence provides 3N dimensions (3 for BxDF, 3 for NEE, etc.)
			// Then Atoms are ordered by sampleID, then dimension (cache will be fully trashed by tracing TLASes until next bounce) 
#if 0	
			// semi persistent data
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
		
			// Resources used for envmap sampling
			nbl::ext::EnvmapImportanceSampling::EnvmapImportanceSampling m_envMapImportanceSampling;
#endif

// Denoiser
			// TODO: autoexposure
			core::smart_refctd_ptr<video::IGPUComputePipeline> lumaMeasure;
			// TODO: motion vector stuff
			// rwmc resolve, apply exposure, interleave into OptiX input formats
			core::smart_refctd_ptr<video::IGPUComputePipeline> rwmcResolve;
			// TODO: OIDN denoise
			// deinterlave from OptiX output format, perform first axis of FFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> postDenoise; // TODO
			// second axis FFT, spectrum multiply and iFFT
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisBloom; // TODO
			// first axis iFFT, tonemap, encode into final EXR format
			core::smart_refctd_ptr<video::IGPUComputePipeline> secondAxisFFTTonemap; // TODO

// Presenter (invokes denoiser)
			core::smart_refctd_ptr<video::IGPURenderpass> presentRenderpass; // TODO
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> regularPresent; // TODO
			core::smart_refctd_ptr<video::IGPUGraphicsPipeline> cubemapPresent; // TODO
		};
		inline CRenderer(SConstructorParams&& _params) : m_creation(std::move(_params)), m_construction(std::move(_params)) {}
		virtual inline ~CRenderer() {}

		SCachedCreationParams m_creation;
		SCachedConstructionParams m_construction;
};

}
#endif
