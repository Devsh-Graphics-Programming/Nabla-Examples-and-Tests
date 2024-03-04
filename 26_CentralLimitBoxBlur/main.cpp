// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <nbl/builtin/hlsl/blur/common.hlsl>

#include "CArchive.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#define _NBL_PLATFORM_WINDOWS_

class BoxBlurDemo final : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::MonoDeviceApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

public:
	BoxBlurDemo( 
		const path& _localInputCWD, 
		const path& _localOutputCWD, 
		const path& _sharedInputCWD, 
		const path& _sharedOutputCWD 
	) : system::IApplicationFramework( _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD )
	{}
	
	bool onAppInitialized( smart_refctd_ptr<ISystem>&& system ) override
	{
		// Remember to call the base class initialization!
		if( !device_base_t::onAppInitialized( std::move( system ) ) )
		{
			return false;
		}
		if( !asset_base_t::onAppInitialized( std::move( system ) ) )
		{
			return false;
		}

		constexpr uint32_t WorkgroupSize = 256;
		constexpr uint32_t AxisDimension = 3;
		constexpr uint32_t PassesPerAxis = 4;

		constexpr uint32_t WorkgroupCount = 2048;

		IAssetLoader::SAssetLoadParams lparams = {};
		lparams.logger = m_logger.get();
		lparams.workingDirectory = "";
		auto checkedLoad = [ & ]<class T>( const char* filePath ) -> smart_refctd_ptr<T>
		{
			// The `IAssetManager::getAsset` function is very complex, in essencee it:
			// 1. takes a cache key or an IFile, if you gave it an `IFile` skip to step 3
			// 2. it consults the loader override about how to get an `IFile` from your cache key
			// 3. handles any failure in opening an `IFile` (which is why it takes a supposed filename), it allows the override to give a different file
			// 4. tries to derive a working directory if you haven't provided one
			// 5. looks for the assets in the cache if you haven't disabled that in the loader parameters
			// 5a. lets the override choose relevant assets from the ones found under the cache key
			// 5b. if nothing was found it lets the override intervene one last time
			// 6. if there's no file to load from, return no assets
			// 7. try all loaders associated with a file extension
			// 8. then try all loaders by opening the file and checking if it will load
			// 9. insert loaded assets into cache if required
			// 10. restore assets from dummy state if needed (more on that in other examples)
			// Take the docs with a grain of salt, the `getAsset` will be rewritten to deal with restores better in the near future.
			nbl::asset::SAssetBundle bundle = m_assetMgr->getAsset( filePath, lparams );
			if( bundle.getContents().empty() )
			{
				m_logger->log( "Asset %s failed to load! Are you sure it exists?", ILogger::ELL_ERROR, filePath );
				return nullptr;
			}
			// All assets derive from `nbl::asset::IAsset`, and can be casted down if the type matches
			static_assert( std::is_base_of_v<nbl::asset::IAsset, T> );
			// The type of the root assets in the bundle is not known until runtime, so this is kinda like a `dynamic_cast` which will return nullptr on type mismatch
			auto typedAsset = IAsset::castDown<T>( bundle.getContents()[ 0 ] ); // just grab the first asset in the bundle
			if( !typedAsset )
			{
				m_logger->log( "Asset type mismatch want %d got %d !", ILogger::ELL_ERROR, T::AssetType, bundle.getAssetType() );

			}
			return typedAsset;
		};

		auto textureToBlur = checkedLoad.operator()< nbl::asset::ICPUImage >( "app_resources/tex.jpg" );
		const auto& inCpuTexInfo = textureToBlur->getCreationParameters();
		
		auto createGPUImages = [ & ](
			core::bitflag<IGPUImage::E_USAGE_FLAGS> usageFlags,
			std::string_view name,
			smart_refctd_ptr<nbl::video::IGPUImage>&& imgOut,
			smart_refctd_ptr<nbl::video::IGPUImageView>&& imgViewOut
		) {
			video::IGPUImage::SCreationParams gpuImageCreateInfo;
			gpuImageCreateInfo.flags = inCpuTexInfo.flags;
			gpuImageCreateInfo.type = inCpuTexInfo.type;
			gpuImageCreateInfo.extent = inCpuTexInfo.extent;
			gpuImageCreateInfo.mipLevels = inCpuTexInfo.mipLevels;
			gpuImageCreateInfo.arrayLayers = inCpuTexInfo.arrayLayers;
			gpuImageCreateInfo.samples = inCpuTexInfo.samples;
			gpuImageCreateInfo.tiling = video::IGPUImage::TILING::OPTIMAL;
			gpuImageCreateInfo.usage = usageFlags | asset::IImage::EUF_TRANSFER_DST_BIT;
			gpuImageCreateInfo.queueFamilyIndexCount = 0u;
			gpuImageCreateInfo.queueFamilyIndices = nullptr;

			gpuImageCreateInfo.format = m_physicalDevice->promoteImageFormat(
				{ inCpuTexInfo.format, gpuImageCreateInfo.usage }, gpuImageCreateInfo.tiling
			);
			auto gpuImage = m_device->createImage( std::move( gpuImageCreateInfo ) );

			auto gpuImageMemReqs = gpuImage->getMemoryReqs();
			gpuImageMemReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			m_device->allocate( gpuImageMemReqs, gpuImage.get(), video::IDeviceMemoryAllocation::EMAF_NONE );

			auto imgView = m_device->createImageView( {
				.flags = IGPUImageView::ECF_NONE,
				.subUsages = usageFlags,
				.image = gpuImage,
				.viewType = IGPUImageView::ET_2D,
				.format = gpuImageCreateInfo.format
			} );
			gpuImage->setObjectDebugName( name.data() );
			imgView->setObjectDebugName( ( std::string{ name } + "view" ).c_str() );
			imgOut = gpuImage;
			imgViewOut = imgView;
		};


		smart_refctd_ptr<nbl::video::IGPUImage> inputGpuImg;
		smart_refctd_ptr<nbl::video::IGPUImage> outputGpuImg;
		smart_refctd_ptr<nbl::video::IGPUImageView> inputGpuImgView;
		smart_refctd_ptr<nbl::video::IGPUImageView> outputGpuImgView;
		createGPUImages( IGPUImage::EUF_SAMPLED_BIT, "InputImg", std::move(inputGpuImg), std::move(inputGpuImgView));
		createGPUImages( IGPUImage::EUF_STORAGE_BIT, "OutputImg", std::move(outputGpuImg), std::move(outputGpuImgView));


		auto computeMain = checkedLoad.operator()< nbl::asset::ICPUShader >( "app_resources/main.comp.hlsl" );
		smart_refctd_ptr<ICPUShader> overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
			computeMain.get(), 
			"#define WORKGROUP_SIZE %s\n#define PASSES_PER_AXIS %d\n#define AXIS_DIM %d\n",
			std::to_string( WorkgroupSize ).c_str(), AxisDimension, PassesPerAxis
		);
		smart_refctd_ptr<IGPUShader> shader = m_device->createShader( overridenUnspecialized.get() );
		if( !shader )
		{
			return logFail( "Creation of a GPU Shader to from CPU Shader source failed!" );
		}


		// TODO: move to shaderd cpp/hlsl descriptors file
		NBL_CONSTEXPR_STATIC nbl::video::IGPUDescriptorSetLayout::SBinding bindings[] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_COMBINED_IMAGE_SAMPLER, 
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr
			},
			{
				.binding = 1,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr 
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout( bindings );
		if( !dsLayout )
		{
			return logFail( "Failed to create a Descriptor Layout!\n" );
		}
		const asset::SPushConstantRange pushConst[] = { {.stageFlags = IShader::ESS_COMPUTE, .offset = 0, .size = sizeof( BoxBlurParams )} };
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout( pushConst, smart_refctd_ptr(dsLayout));
		if( !pplnLayout )
		{
			return logFail( "Failed to create a Pipeline Layout!\n" );
		}

		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			// we'll cover the specialization constant API in another example
			if( !m_device->createComputePipelines( nullptr, { &params, 1 }, &pipeline ) )
			{
				return logFail( "Failed to create pipelines (compile & link shaders)!\n" );
			}
		}
		smart_refctd_ptr<video::IGPUSampler> sampler = m_device->createSampler( { .TextureWrapU = ISampler::ETC_CLAMP_TO_EDGE } );
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;
		smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts( 
			IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 } );
		ds = pool->createDescriptorSet( std::move( dsLayout ) );
		{
			IGPUDescriptorSet::SDescriptorInfo info[ 2 ];
			info[ 0 ].desc = inputGpuImgView;
			info[ 0 ].info.image = { .sampler = sampler, .imageLayout = IImage::LAYOUT::READ_ONLY_OPTIMAL };
			info[ 1 ].desc = outputGpuImgView;
			info[ 1 ].info.image = { .sampler = nullptr, .imageLayout = IImage::LAYOUT::GENERAL };

			IGPUDescriptorSet::SWriteDescriptorSet writes[] = {
				{ .dstSet = ds.get(), .binding = 0, .arrayElement = 0, .count = 1, .info = &info[ 0 ] },
				{ .dstSet = ds.get(), .binding = 1, .arrayElement = 0, .count = 1, .info = &info[ 1 ] },
			};
			m_device->updateDescriptorSets( writes, {} );
		}

		uint32_t computeQueueIndex = getComputeQueue()->getFamilyIndex();
		IQueue* queue = m_device->getQueue( computeQueueIndex, 0 );

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(
			computeQueueIndex, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT );
		if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
		{
			return logFail( "Failed to create Command Buffers!\n" );
		}

		constexpr size_t StartedValue = 0;
		constexpr size_t FinishedValue = 45;
		static_assert( FinishedValue > StartedValue );
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore( StartedValue );
		
		IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };

		nbl::video::SIntendedSubmitInfo::SFrontHalf frontHalf = { .queue = queue, .commandBuffers = cmdbufs };
		smart_refctd_ptr<nbl::video::IUtilities> assetStagingMngr = 
			make_smart_refctd_ptr<IUtilities>( smart_refctd_ptr( m_device ), smart_refctd_ptr( m_logger ) );

		cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );

		queue->startCapture();
		bool uploaded = assetStagingMngr->updateImageViaStagingBufferAutoSubmit( 
			frontHalf, textureToBlur->getBuffer(), inCpuTexInfo.format,
			inputGpuImg.get(), IImage::LAYOUT::UNDEFINED, textureToBlur->getRegions()
		);
		queue->endCapture();
		if( !uploaded )
		{
			return logFail( "Failed to upload cpu tex!\n" );
		}

		cmdbuf->reset( IGPUCommandBuffer::RESET_FLAGS::NONE );

		BoxBlurParams pushConstData = {};
		

		cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );
		cmdbuf->beginDebugMarker( "My Compute Dispatch", core::vectorSIMDf( 0, 1, 0, 1 ) );
		nbl::video::IGPUCommandBuffer::SImageResolve regions[] = {
			{
				.srcSubresource = { .layerCount = 1 },
				.srcOffset = {},
				.dstSubresource = { .layerCount = 1 },
				.dstOffset = {},
				.extent = inputGpuImg->getCreationParameters().extent
			}
		};
		cmdbuf->resolveImage( 
			inputGpuImg.get(), IImage::LAYOUT::UNDEFINED,
			inputGpuImg.get(), IImage::LAYOUT::GENERAL,
			std::size( regions ), regions );
		nbl::video::IGPUCommandBuffer::SImageResolve regionsOut[] = {
			{
				.srcSubresource = {.layerCount = 1 },
				.srcOffset = {},
				.dstSubresource = {.layerCount = 1 },
				.dstOffset = {},
				.extent = outputGpuImg->getCreationParameters().extent
			}
		};
		cmdbuf->resolveImage(
			outputGpuImg.get(), IImage::LAYOUT::UNDEFINED,
			outputGpuImg.get(), IImage::LAYOUT::GENERAL,
			std::size( regionsOut ), regionsOut );
		cmdbuf->bindComputePipeline( pipeline.get() );
		cmdbuf->bindDescriptorSets( nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &ds.get() );
		cmdbuf->pushConstants( pplnLayout.get(), IShader::ESS_COMPUTE, 0, sizeof( BoxBlurParams ), &pushConstData );
		cmdbuf->dispatch( WorkgroupCount, 1, 1 );

		const nbl::asset::SMemoryBarrier barriers[] = {
			{
				.srcStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask= nbl::asset::ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask= nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.dstAccessMask= nbl::asset::ACCESS_FLAGS::SHADER_READ_BITS,
			}
		};
		cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .memBarriers = barriers } );

		cmdbuf->dispatch( WorkgroupCount, 1, 1 );
		cmdbuf->endDebugMarker();
		// Normally you'd want to perform a memory barrier when using the output of a compute shader or renderpass,
		// however waiting on a timeline semaphore (or fence) on the Host makes all Device writes visible.
		cmdbuf->end();
		
		{
			// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
			// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			// But we do need to signal completion by incrementing the Timeline Semaphore counter as soon as the compute shader is done
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
			// Default, we have no semaphores to wait on before we can start our workload
			IQueue::SSubmitInfo submitInfos[] = { { .commandBuffers = cmdbufs, .signalSemaphores = signals } };

			// We have a cool integration with RenderDoc that allows you to start and end captures programmatically.
			// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
			queue->startCapture();
			queue->submit( submitInfos );
			queue->endCapture();
		}
		// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
		const ISemaphore::SWaitInfo waitInfos[] = { { .semaphore = progress.get(), .value = FinishedValue } };
		m_device->blockForSemaphores( waitInfos );
			

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC( BoxBlurDemo )