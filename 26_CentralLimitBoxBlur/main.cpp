// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <nbl/builtin/hlsl/central_limit_blur/common.hlsl>

#include "CArchive.h"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#define _NBL_PLATFORM_WINDOWS_

class BoxBlurDemo final : public examples::BasicMultiQueueApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using base_t = examples::BasicMultiQueueApplication;
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
		if( !base_t::onAppInitialized( core::smart_refctd_ptr( system ) ) )
		{
			return false;
		}
		if( !asset_base_t::onAppInitialized( std::move( system ) ) )
		{
			return false;
		}

		constexpr uint32_t WorkgroupSize = 256;
		constexpr uint32_t AxisDimension = 3;
		constexpr uint32_t PassesPerAxis = 10;

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
		if( !textureToBlur )
		{
			return logFail( "Failed to load texture!\n" );
		}
		const auto& inCpuTexInfo = textureToBlur->getCreationParameters();
		
		auto createGPUImages = [ & ](
			core::bitflag<IGPUImage::E_USAGE_FLAGS> usageFlags,
			std::string_view name,
			smart_refctd_ptr<nbl::video::IGPUImage>& imgOut,
			smart_refctd_ptr<nbl::video::IGPUImageView>& imgViewOut
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
			imgView->setObjectDebugName( ( std::string{ name } + "_view" ).c_str() );
			imgOut = gpuImage;
			imgViewOut = imgView;
		};


		smart_refctd_ptr<nbl::video::IGPUImage> inputGpuImg;
		smart_refctd_ptr<nbl::video::IGPUImage> outputGpuImg;
		smart_refctd_ptr<nbl::video::IGPUImageView> inputGpuImgView;
		smart_refctd_ptr<nbl::video::IGPUImageView> outputGpuImgView;
		createGPUImages( IGPUImage::EUF_SAMPLED_BIT, "InputImg", inputGpuImg, inputGpuImgView );
		createGPUImages( IGPUImage::EUF_STORAGE_BIT, "OutputImg", outputGpuImg, outputGpuImgView );
		assert(inputGpuImg&&outputGpuImg&&inputGpuImgView&&outputGpuImgView);

		auto computeMain = checkedLoad.operator()< nbl::asset::ICPUShader >( "app_resources/main.comp.hlsl" );
		smart_refctd_ptr<ICPUShader> overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
			computeMain.get(), 
			"#define WORKGROUP_SIZE %s\n#define PASSES_PER_AXIS %d\n#define AXIS_DIM %d\n",
			std::to_string( WorkgroupSize ).c_str(), PassesPerAxis, AxisDimension
		);
		smart_refctd_ptr<IGPUShader> shader = m_device->createShader( overridenUnspecialized.get() );
		if( !shader )
		{
			return logFail( "Creation of a GPU Shader to from CPU Shader source failed!" );
		}


		// TODO: move to shaderd cpp/hlsl descriptors file 
		// No because this is a C++ only struct, only move the binding and count there
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
		const asset::SPushConstantRange pushConst[] = { {.stageFlags = IShader::ESS_COMPUTE, .offset = 0, .size = sizeof( nbl::hlsl::central_limit_blur::BoxBlurParams )} };
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
			if( !m_device->createComputePipelines( nullptr, { &params, 1 }, &pipeline ) )
			{
				return logFail( "Failed to create pipelines (compile & link shaders)!\n" );
			}
		}
		smart_refctd_ptr<video::IGPUSampler> sampler = m_device->createSampler( {} );
		smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts( IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 } );
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds = pool->createDescriptorSet( std::move( dsLayout ) );
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
			const bool success = m_device->updateDescriptorSets( writes, {} );
			assert( success );
		}

		ds->setObjectDebugName( "Box blur DS" );
		pplnLayout->setObjectDebugName( "Box Blur PPLN Layout" );

		// Transfer stage
		const bool needsOwnershipTransfer = getTransferUpQueue()->getFamilyIndex()!=getComputeQueue()->getFamilyIndex();
		auto transferSema = m_device->createSemaphore(0);
		IQueue::SSubmitInfo::SSemaphoreInfo transferDone[] = {
			{.semaphore = transferSema.get(),.value = 1,.stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT} };
		{
			IQueue* queue = getTransferUpQueue();

			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(
				queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT );
			if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
			{
				return logFail( "Failed to create Command Buffers!\n" );
			}

			
			cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );

			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgLayouts[] = {
				{
					.barrier = {
						.dep={
							// there's no need for a source synchronization because Host Ops become available and visible pre-submit
							.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, .srcAccessMask = ACCESS_FLAGS::NONE,
							.dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .dstAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT
						},
					},
					.image = inputGpuImg.get(),
					.subresourceRange = { .aspectMask = IGPUImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
					.newLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				}
			};
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = imgLayouts } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}

			queue->startCapture();
			IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			SIntendedSubmitInfo intendedSubmit = {
				.frontHalf = {.queue = queue, .waitSemaphores = {/*wait for no - one*/}, .commandBuffers = cmdbufs }, .signalSemaphores = transferDone };
			bool uploaded = m_utils->updateImageViaStagingBuffer(
				intendedSubmit, textureToBlur->getBuffer(), inCpuTexInfo.format,
				inputGpuImg.get(), IImage::LAYOUT::TRANSFER_DST_OPTIMAL, textureToBlur->getRegions()
			);
			if( !uploaded )
			{
				return logFail( "Failed to upload cpu tex!\n" );
			}

			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> releaseOwnership[] = {
				{
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT, .srcAccessMask = ACCESS_FLAGS::TRANSFER_WRITE_BIT,
							// there's no need for a source synchronization scope because the Submit implicit
							// Timeline Semaphore guarantees already sync us and make our writes available
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
						.otherQueueFamilyIndex = needsOwnershipTransfer ? getComputeQueue()->getFamilyIndex():IQueue::FamilyIgnored
					},
					.image = inputGpuImg.get(),
					.subresourceRange = {.aspectMask = IGPUImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
				    .newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL,
				}
			};
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = releaseOwnership } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}

			cmdbuf->end();
			const IQueue::SSubmitInfo info = intendedSubmit;
			queue->submit({&info,1});
			queue->endCapture();

			// WARNING : Depending on OVerflows, `transferDone->value!=1` so if you want to sync the compute submit against that,
			// use `transferDone` directly as the wait semaphore!
			const ISemaphore::SWaitInfo waitInfo = {transferDone->semaphore,transferDone->value};
			m_device->blockForSemaphores( { &waitInfo,1 } );
		}
		
		constexpr size_t StartedValue = 0;
		constexpr size_t FinishedValue = 45;
		static_assert( StartedValue < FinishedValue );
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore( StartedValue );
		IQueue* queue = getComputeQueue();

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(
			queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT );
		if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
		{
			return logFail( "Failed to create Command Buffers!\n" );
		}

		struct packed_data_t
		{
			uint32_t direction : 2;
			uint32_t channelCount : 3 = 3; // TODO: don't hardcode
			uint32_t wrapMode : 2 = hlsl::central_limit_blur::WrapMode::WRAP_MODE_CLAMP_TO_EDGE;
			uint32_t borderColor : 3 = hlsl::central_limit_blur::BorderColor::BORDER_COLOR_FLOAT_OPAQUE_BLACK;

			explicit operator uint32_t() const {
				return *reinterpret_cast< const uint32_t* >( this );
			}
		};

		hlsl::central_limit_blur::BoxBlurParams pushConstData = {
			.inputDimensions = {0,0,0,uint32_t( packed_data_t{} )}, .chosenAxis = {1, 0}, .radius = 4.f
		};

		auto gpuTexSize = inputGpuImg->getCreationParameters().extent;
		const uint32_t WorkgroupCount = ( gpuTexSize.width * gpuTexSize.height ) / WorkgroupSize;

		cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );
		cmdbuf->beginDebugMarker( "Box Blur dispatches", core::vectorSIMDf( 0, 1, 0, 1 ) );
		{
			const IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgLayouts[] = {
				{
					.barrier = {
						.dep = {
						//.srcStageMask = PIPELINE_STAGE_FLAGS::NONE, .srcAccessMask = ACCESS_FLAGS::NONE,
						.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT, 
						.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
						},
					},
					.image = outputGpuImg.get(),
					.subresourceRange = {.aspectMask = IGPUImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IGPUImage::LAYOUT::UNDEFINED,
					.newLayout = IGPUImage::LAYOUT::GENERAL,
				},
				// this is only for Ownership Acquire, the transfer queue does the layout xform,
				// so if `!needsOwnershipTransfer` we skip to prevent a double layout transition
				{
					.barrier = {
						// src flags are ignored by Acquire
						.dep = {
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.dstAccessMask = ACCESS_FLAGS::SAMPLED_READ_BIT,
						},
						.ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
						.otherQueueFamilyIndex =  getTransferUpQueue()->getFamilyIndex()
					},
					.image = inputGpuImg.get(),
					.subresourceRange = {.aspectMask = IGPUImage::EAF_COLOR_BIT, .levelCount = 1, .layerCount = 1 },
					.oldLayout = IGPUImage::LAYOUT::TRANSFER_DST_OPTIMAL,
					.newLayout = IGPUImage::LAYOUT::READ_ONLY_OPTIMAL
				}
			};

			const size_t barrierCount = needsOwnershipTransfer ? 2 : 1;
			if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .imgBarriers = {imgLayouts, barrierCount} } ) )
			{
				return logFail( "Failed to issue barrier!\n" );
			}
		}

		cmdbuf->bindComputePipeline( pipeline.get() );
		cmdbuf->bindDescriptorSets( nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &ds.get() );
		cmdbuf->pushConstants( pplnLayout.get(), IShader::ESS_COMPUTE, 0, sizeof( pushConstData ), &pushConstData );
		cmdbuf->dispatch( WorkgroupCount, 1, 1 );

		const nbl::asset::SMemoryBarrier barriers[] = {
			{
				.srcStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.srcAccessMask = nbl::asset::ACCESS_FLAGS::SHADER_WRITE_BITS,
				.dstStageMask = nbl::asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
				.dstAccessMask= nbl::asset::ACCESS_FLAGS::SHADER_READ_BITS,
			}
		};
		// TODO: you don't need a pipeline barrier just before the end of the last command buffer to be submitted
		// Timeline semaphore takes care of all the memory deps between a signal and a wait 
		if( !cmdbuf->pipelineBarrier( nbl::asset::EDF_NONE, { .memBarriers = barriers } ) )
		{
			return logFail( "Failed to issue barrier!\n" );
		}
		//cmdbuf->dispatch( WorkgroupCount, 1, 1 );
		cmdbuf->endDebugMarker();
		cmdbuf->end();
		
		{
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { 
				{.semaphore = progress.get(), .value = FinishedValue, .stageMask = asset::PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS} };
			IQueue::SSubmitInfo submitInfos[] = { 
				{ .waitSemaphores = transferDone, .commandBuffers = cmdbufs, .signalSemaphores = signals } };

			// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
			queue->startCapture();
			queue->submit( submitInfos );
			queue->endCapture();
		}
		const ISemaphore::SWaitInfo waitInfos[] = { { .semaphore = progress.get(), .value = FinishedValue } };
		m_device->blockForSemaphores( waitInfos );

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

	// Just to run destructors in a nice order
	bool onAppTerminated() override
	{
		return base_t::onAppTerminated();
	}
};


NBL_MAIN_FUNC( BoxBlurDemo )