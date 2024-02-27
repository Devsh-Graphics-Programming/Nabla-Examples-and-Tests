// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

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
	BoxBlurDemo( const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD 
	) : system::IApplicationFramework( _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD ) {}
	
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
		constexpr uint32_t WorkgroupCount = 2048;


		// load shader source from file
		auto getShaderSource = [ & ]( const char* filePath ) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams = {};
			lparams.logger = m_logger.get();
			lparams.workingDirectory = "";
			auto bundle = m_assetMgr->getAsset( filePath, lparams );
			if( bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER )
			{
				m_logger->log( "Shader %s not found!", ILogger::ELL_ERROR, filePath );
				exit( -1 );
			}
			auto firstAssetInBundle = bundle.getContents()[ 0 ];
			return smart_refctd_ptr_static_cast< ICPUShader >( firstAssetInBundle );
		};
		auto computeMain = getShaderSource( "app_resources/main.comp.hlsl" );

		smart_refctd_ptr<ICPUShader> overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
			computeMain.get(), 
			"#define WORKGROUP_SIZE %s\n#define PASSES_PER_AXIS %d\n#define AXIS_DIM %d\n",
			std::to_string( WorkgroupSize ).c_str(), 3, 4
		);
		smart_refctd_ptr<IGPUShader> shader = m_device->createShader( overridenUnspecialized.get() );
		if( !shader )
		{
			return logFail( "Creation of a GPU Shader to from CPU Shader source failed!" );
		}

		/*// the simplest example would have used push constants and BDA, but RenderDoc's debugging of that sucks, so I'll demonstrate "classical" binding of buffers with descriptors
		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[ 1 ] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, // not is not the time for descriptor indexing
				.stageFlags = IGPUShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr // irrelevant for a buffer
			}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = device->createDescriptorSetLayout( bindings );
		if( !dsLayout )
			return logFail( "Failed to create a Descriptor Layout!\n" );

		// Nabla actually has facilities for SPIR-V Reflection and "guessing" pipeline layouts for a given SPIR-V which we'll cover in a different example
		smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = device->createPipelineLayout( {}, smart_refctd_ptr( dsLayout ) );
		if( !pplnLayout )
			return logFail( "Failed to create a Pipeline Layout!\n" );

		// We use strong typing on the pipelines (Compute, Graphics, Mesh, RT), since there's no reason to polymorphically switch between different pipelines
		smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
		{
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = pplnLayout.get();
			// Theoretically a blob of SPIR-V can contain multiple named entry points and one has to be chosen, in practice most compilers only support outputting one (and glslang used to require it be called "main")
			params.shader.entryPoint = "main";
			params.shader.shader = shader.get();
			// we'll cover the specialization constant API in another example
			if( !device->createComputePipelines( nullptr, { &params,1 }, &pipeline ) )
				return logFail( "Failed to create pipelines (compile & link shaders)!\n" );
		}

		// Our Descriptor Sets track (refcount) resources written into them, so you can pretty much drop and forget whatever you write into them.
		// A later Descriptor Indexing example will test that this tracking is also correct for Update-After-Bind Descriptor Set bindings too.
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;

		// A `nbl::video::DeviceMemoryAllocator` is an interface to implement anything that can dish out free memory range to bind to back a `nbl::video::IGPUBuffer` or a `nbl::video::IGPUImage`
		// The Logical Device itself implements the interface and behaves as the most simple allocator, it will create a new `nbl::video::IDeviceMemoryAllocation` every single time.
		// We will cover allocators and suballocation in a later example.
		nbl::video::IDeviceMemoryAllocator::SAllocation allocation = {};
		{
			constexpr size_t BufferSize = sizeof( uint32_t ) * WorkgroupSize * WorkgroupCount;

			// Always default the creation parameters, there's a lot of extra stuff for DirectX/CUDA interop and slotting into external engines you don't usually care about. 
			nbl::video::IGPUBuffer::SCreationParams params = {};
			params.size = BufferSize;
			// While the usages on `ICPUBuffers` are mere hints to our automated CPU-to-GPU conversion systems which need to be patched up anyway,
			// the usages on an `IGPUBuffer` are crucial to specify correctly.
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
			smart_refctd_ptr<IGPUBuffer> outputBuff = device->createBuffer( std::move( params ) );
			if( !outputBuff )
				return logFail( "Failed to create a GPU Buffer of size %d!\n", params.size );

			// Naming objects is cool because not only errors (such as Vulkan Validation Layers) will show their names, but RenderDoc captures too.
			outputBuff->setObjectDebugName( "My Output Buffer" );

			// We don't want to bother explaining best staging buffer practices just yet, so we will create a buffer over
			// a memory type thats Host Visible (can be mapped and give the CPU a direct pointer to read from)
			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
			// you can simply constrain the memory requirements by AND-ing the type bits of the host visible memory types
			reqs.memoryTypeBits &= physDev->getHostVisibleMemoryTypeBits();

			// There are actually two `allocate` overloads, one which allocates memory if you already know the type you want.
			// And this one which is a utility which tries to allocate from every type that matches your requirements in some order of preference.
			// The other of preference (iteration over compatible types) can be controlled by the method's template parameter,
			// the default is from lowest index to highest, but skipping over incompatible types.
			allocation = device->allocate( reqs, outputBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE );
			if( !allocation.isValid() )
				return logFail( "Failed to allocate Device Memory compatible with our GPU Buffer!\n" );

			// Note that we performed a Dedicated Allocation above, so there's no need to bind the memory anymore (since the allocator knows the dedication, it can already bind).
			// This is a carryover from having an OpenGL backend, where you couldn't have a memory allocation separate from the resource, so all allocations had to be "dedicated".
			// In Vulkan dedicated allocations are the most performant and still make sense as long as you won't blow the 4096 allocation limit on windows.
			// You should always use dedicated allocations for images used for swapchains, framebuffer attachments (esp transient), as well as objects used in CUDA/DirectX interop.
			assert( outputBuff->getBoundMemory().memory == allocation.memory.get() );

			// This is a cool utility you can use instead of counting up how much of each descriptor type you need to N_i allocate descriptor sets with layout L_i from a single pool
			smart_refctd_ptr<nbl::video::IDescriptorPool> pool = device->createDescriptorPoolForDSLayouts( IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 } );

			// note how the pool will go out of scope but thanks for backreferences in each object to its parent/dependency it will be kept alive for as long as all the Sets it allocated
			ds = pool->createDescriptorSet( std::move( dsLayout ) );
			// we still use Vulkan 1.0 descriptor update style, could move to Update Templates but Descriptor Buffer ubiquity seems just around the corner
			{
				IGPUDescriptorSet::SDescriptorInfo info[ 1 ];
				info[ 0 ].desc = smart_refctd_ptr( outputBuff ); // bad API, too late to change, should just take raw-pointers since not consumed
				info[ 0 ].info.buffer = { .offset = 0,.size = BufferSize };
				IGPUDescriptorSet::SWriteDescriptorSet writes[ 1 ] = {
					{.dstSet = ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
				};
				device->updateDescriptorSets( writes, {} );
			}
		}

		// To be able to read the contents of the buffer we need to map its memory
		// P.S. Nabla mandates Persistent Memory Mappings on all backends (but not coherent memory types)
		auto ptr = allocation.memory->map( { 0ull,allocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ );
		if( !ptr )
			return logFail( "Failed to map the Device Memory!\n" );

		// Our commandbuffers are cool because they refcount the resources used by each command you record into them, so you can rely a commandbuffer on keeping them alive.
		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = device->createCommandPool( params.queueParams[ 0 ].familyIndex, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT );
			if( !cmdpool->createCommandBuffers( IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf ) )
				return logFail( "Failed to create Command Buffers!\n" );
		}

		cmdbuf->begin( IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT );
		// If you enable the `debugUtils` API Connection feature on a supported backend as we've done, you'll get these pretty debug sections in RenderDoc
		cmdbuf->beginDebugMarker( "My Compute Dispatch", core::vectorSIMDf( 0, 1, 0, 1 ) );
		// you want to bind the pipeline first to avoid accidental unbind of descriptor sets due to compatibility matching
		cmdbuf->bindComputePipeline( pipeline.get() );
		cmdbuf->bindDescriptorSets( nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &ds.get() );
		cmdbuf->dispatch( WorkgroupCount, 1, 1 );
		cmdbuf->endDebugMarker();
		// Normally you'd want to perform a memory barrier when using the output of a compute shader or renderpass,
		// however waiting on a timeline semaphore (or fence) on the Host makes all Device writes visible.
		cmdbuf->end();

		// Only Timeline Semaphores are supported in Nabla, there's no fences or binary semaphores.
		// Swapchains run on adaptors with empty submits that make them look like they work with Timeline Semaphores,
		// which has important side-effects we'll cover in another example.
		constexpr auto StartedValue = 0;
		constexpr auto FinishedValue = 45;
		static_assert( FinishedValue > StartedValue );
		smart_refctd_ptr<ISemaphore> progress = device->createSemaphore( StartedValue );
		{
			// queues are inherent parts of the device, ergo not refcounted (you refcount the device instead)
			IQueue* queue = device->getQueue( params.queueParams[ 0 ].familyIndex, 0 );

			// Default, we have no semaphores to wait on before we can start our workload
			IQueue::SSubmitInfo submitInfos[ 1 ] = {};
			// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
			// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
			const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
			submitInfos[ 0 ].commandBuffers = cmdbufs;
			// But we do need to signal completion by incrementing the Timeline Semaphore counter as soon as the compute shader is done
			const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
			submitInfos[ 0 ].signalSemaphores = signals;

			// We have a cool integration with RenderDoc that allows you to start and end captures programmatically.
			// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
			queue->startCapture();
			queue->submit( submitInfos );
			queue->endCapture();
		}
		// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
		const ISemaphore::SWaitInfo waitInfos[] = { {
			.semaphore = progress.get(),
			.value = FinishedValue
		} };
		device->blockForSemaphores( waitInfos );

		// You don't need to do this, but putting it here to demonstrate that its safe to drop a commandbuffer after GPU is done (try moving it above and see if you BSOD or just get a validation error). 
		cmdbuf = nullptr;

		// if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
		const ILogicalDevice::MappedMemoryRange memoryRange( allocation.memory.get(), 0ull, allocation.memory->getAllocationSize() );
		if( !allocation.memory->getMemoryPropertyFlags().hasFlags( IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT ) )
			device->invalidateMappedMemoryRanges( 1, &memoryRange );

		// a simple test to check we got the right thing back
		auto buffData = reinterpret_cast< const uint32_t* >( ptr );
		for( auto i = 0; i < WorkgroupSize * WorkgroupCount; i++ )
			if( buffData[ i ] != i )
				return logFail( "DWORD at position %d doesn't match!\n", i );
		// This allocation would unmap itself in the dtor anyway, but lets showcase the API usage
		allocation.memory->unmap();

		return true;*/
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC( BoxBlurDemo )