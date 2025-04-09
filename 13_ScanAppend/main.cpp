// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h


// I've moved out a tiny part of this example into a shared header for reuse, please open and read it.
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;


#include "app_resources/common.hlsl"

#include "nbl/builtin/hlsl/bit.hlsl"

// In this application we'll cover buffer streaming, Buffer Device Address (BDA) and push constants 
class ScanAppendUnitTestApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;


		smart_refctd_ptr<IGPUComputePipeline> m_pipeline;

		smart_refctd_ptr<IGPUBuffer> m_inputBuffer;
		smart_refctd_ptr<IGPUBuffer> m_outputBuffer;

		input_t* m_inputPtr;
		output_t* m_outputPtr;

		bool m_testFailed = false;

		// The pool cache is just a formalized way of round-robining command pools and resetting + reusing them after their most recent submit signals finished.
		// Its a little more ergonomic to use if you don't have a 1:1 mapping between frames and pools.
		smart_refctd_ptr<nbl::video::ICommandPoolCache> m_poolCache;

		// This example really lets the advantages of a timeline semaphore shine through!
		smart_refctd_ptr<ISemaphore> m_timeline;
		uint64_t m_iteration = 0;
		constexpr static inline uint64_t MaxIterations = 200;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		ScanAppendUnitTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD,_localOutputCWD,_sharedInputCWD,_sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			// this time we load a shader directly from a file
			smart_refctd_ptr<IGPUShader> shader;
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl",lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
					return logFail("Could not load shader!");

				// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
				auto source = IAsset::castDown<ICPUShader>(assets[0]);
				// The down-cast should not fail!
				assert(source);

				// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
				shader = m_device->createShader(source.get());
				if (!shader)
					return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
			}

			IGPUBuffer::SCreationParams inputBufferCreationParams = {};
			inputBufferCreationParams.size = ElementCount * sizeof(input_t);
			inputBufferCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			m_inputBuffer = m_device->createBuffer(std::move(inputBufferCreationParams));
			auto inputBufferMemoryReqs = m_inputBuffer->getMemoryReqs();
			inputBufferMemoryReqs.memoryTypeBits &= m_physicalDevice->getUpStreamingMemoryTypeBits();
			auto inputBufferAllocation = m_device->allocate(inputBufferMemoryReqs, m_inputBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			
			IGPUBuffer::SCreationParams outputBufferCreationParams = {};
			outputBufferCreationParams.size = (ElementCount+1u) * sizeof(output_t);
			outputBufferCreationParams.usage = IGPUBuffer::E_USAGE_FLAGS::EUF_SHADER_DEVICE_ADDRESS_BIT;
			m_outputBuffer = m_device->createBuffer(std::move(outputBufferCreationParams));
			auto outputBufferMemoryReqs = m_outputBuffer->getMemoryReqs();
			outputBufferMemoryReqs.memoryTypeBits &= m_physicalDevice->getDownStreamingMemoryTypeBits();
			auto outputBufferAllocation = m_device->allocate(outputBufferMemoryReqs, m_outputBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);

			
			const auto inputBoundMem = m_inputBuffer->getBoundMemory();
			const auto outputBoundMem = m_outputBuffer->getBoundMemory();
			const ILogicalDevice::MappedMemoryRange inputMappedRange(inputBoundMem.memory,inputBoundMem.offset,sizeof(input_t)*ElementCount);
			const ILogicalDevice::MappedMemoryRange outputMappedRange(inputBoundMem.memory,inputBoundMem.offset,sizeof(input_t)*ElementCount);
			m_inputPtr = reinterpret_cast<input_t*>(inputBoundMem.memory->map(inputMappedRange.range, IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE));
			m_outputPtr = reinterpret_cast<output_t*>(outputBoundMem.memory->map(outputMappedRange.range, IDeviceMemoryAllocation::EMCAF_READ_AND_WRITE));

			const nbl::asset::SPushConstantRange pcRange = {.stageFlags=IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset=0,.size=sizeof(PushConstantData)};

			{
				auto layout = m_device->createPipelineLayout({&pcRange,1});
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = shader.get();
				if (!m_device->createComputePipelines(nullptr,{&params,1},&m_pipeline))
					return logFail("Failed to create compute pipeline!\n");
			}

			// We'll allow subsequent iterations to overlap each other on the GPU, the only limiting factors are
			// the amount of memory in the streaming buffers and the number of commandpools we can use simultaenously.
			constexpr auto MaxConcurrency = 64;

			// Since this time we don't throw the Command Pools away and we'll reset them instead, we don't create the pools with the transient flag
			m_poolCache = ICommandPoolCache::create(core::smart_refctd_ptr(m_device),getComputeQueue()->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::NONE,MaxConcurrency);

			// In contrast to fences, we just need one semaphore to rule all dispatches
			m_timeline = m_device->createSemaphore(m_iteration);

			return true;
		}

		// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
		bool keepRunning() override { return m_iteration<MaxIterations && !m_testFailed; }

		// Finally the first actual work-loop
		void workLoopBody() override
		{
			IQueue* const queue = getComputeQueue();

			// Note that I'm using the sample struct with methods that have identical code which compiles as both C++ and HLSL
			auto rng = nbl::hlsl::Xoroshiro64StarStar::construct({m_iteration^0xdeadbeefu,std::hash<string>()(_NBL_APP_NAME_)});
			
			// we dynamically choose the number of elements for each iteration
			const size_t inputSize = sizeof(input_t)*ElementCount;
			const size_t outputSize = sizeof(output_t)*(ElementCount+1u);

			// Map and Copy to our input memory
			{
				const auto inputBoundMem = m_inputBuffer->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange inputMappedRange(inputBoundMem.memory,inputBoundMem.offset,sizeof(input_t)*ElementCount);

				for (uint32_t i = 0; i < ElementCount; ++i)
					m_inputPtr[i] = rng() % 1024u;

				if (inputBoundMem.memory->haveToMakeVisible())
					m_device->flushMappedMemoryRanges(1,&inputMappedRange);
			}

			// Obtain our command pool once one gets recycled
			uint32_t poolIx;
			do
			{
				poolIx = m_poolCache->acquirePool();
			} while (poolIx==ICommandPoolCache::invalid_index);

			smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
			{
				m_poolCache->getPool(poolIx)->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1},core::smart_refctd_ptr(m_logger));
				// lets record, its still a one time submit because we have to re-record with different push constants each time
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
				cmdbuf->bindComputePipeline(m_pipeline.get());
				static_assert(sizeof(output_t) == sizeof(uint64_t));
				PushConstantData pc = {
					.inputAddress=m_inputBuffer->getDeviceAddress(),
					.outputAddress=m_outputBuffer->getDeviceAddress(),
					.atomicBDA=m_outputBuffer->getDeviceAddress() + sizeof(uint64_t) * ElementCount,
					.dataElementCount=ElementCount,
					.isAtomicClearDispatch=true
				};
				cmdbuf->pushConstants(m_pipeline->getLayout(),IShader::E_SHADER_STAGE::ESS_COMPUTE,0u,sizeof(pc),&pc);
				
				cmdbuf->dispatch(1,1,1);

				// TODO[Erfan]: need a memory barrier between the two dispatches.

				pc.isAtomicClearDispatch = false;
				cmdbuf->pushConstants(m_pipeline->getLayout(),IShader::E_SHADER_STAGE::ESS_COMPUTE,0u,sizeof(pc),&pc);
				// Good old trick to get rounded up divisions, in case you're not familiar
				cmdbuf->dispatch((ElementCount-1)/WorkgroupSize+1,1,1);
				cmdbuf->end();
			}


			const auto savedIterNum = m_iteration++;
			{
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufInfo =
				{
					.cmdbuf = cmdbuf.get()
				};
				const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo =
				{
					.semaphore = m_timeline.get(),
					.value = m_iteration,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				};
				// Generally speaking we don't need to wait on any semaphore because in this example every dispatch gets its own clean piece of memory to use
				// from the point of view of the GPU. Implicit domain operations between Host and Device happen upon a submit and a semaphore/fence signal operation,
				// this ensures we can touch the input and get accurate values from the output memory using the CPU before and after respectively, each submit becoming PENDING.
				// If we actually cared about this submit seeing the memory accesses of a previous dispatch we could add a semaphore wait
				const IQueue::SSubmitInfo submitInfo = {
					.waitSemaphores = {},
					.commandBuffers = {&cmdbufInfo,1},
					.signalSemaphores = {&signalInfo,1}
				};

				m_api->startCapture();
				queue->submit({&submitInfo,1});
				m_api->endCapture();
			}
				
			// We let all latches know what semaphore and counter value has to be passed for the functors to execute
			const ISemaphore::SWaitInfo futureWait = {m_timeline.get(),m_iteration};

			// We can also actually latch our Command Pool reset and its return to the pool of free pools!
			m_poolCache->releasePool(futureWait,poolIx);

			ISemaphore::SWaitInfo semaphoreWait = { .semaphore = m_timeline.get(), .value = m_iteration };
			m_device->blockForSemaphores({&semaphoreWait, 1u}, true);

			// Map and read and output memory
			{
				const auto inputBoundMem = m_inputBuffer->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange inputMappedRange(inputBoundMem.memory,inputBoundMem.offset, inputSize);

				if (inputBoundMem.memory->haveToMakeVisible())
					m_device->invalidateMappedMemoryRanges(1,&inputMappedRange);

				const auto outputBoundMem = m_outputBuffer->getBoundMemory();
				const ILogicalDevice::MappedMemoryRange outputMappedRange(outputBoundMem.memory,outputBoundMem.offset, outputSize);

				if (outputBoundMem.memory->haveToMakeVisible())
					m_device->invalidateMappedMemoryRanges(1,&outputMappedRange);

				output_t* sortedOutputs = reinterpret_cast<output_t*>(::malloc(outputSize));

				for (auto j = 0; j < ElementCount; j++)
					sortedOutputs[j] = m_outputPtr[j];

				std::sort(sortedOutputs, sortedOutputs + ElementCount, [](output_t lhs, output_t rhs)
					{
						return lhs.getSecond() < rhs.getSecond();
					});

				uint64_t exclusivePrefixSum = 0ull;
				for (auto j=0; j<ElementCount; j++)
				{
					if (sortedOutputs[j].getFirst() != exclusivePrefixSum)
					{
						m_logger->log("PrefixSum Calculation Failed. sortedOutputs[%d].second != actualPrefixSum",ILogger::ELL_ERROR,j);
						m_testFailed = true;
						break;
					}
					exclusivePrefixSum += m_inputPtr[j];
				}

				::free(sortedOutputs);
			}
		}

		bool onAppTerminated() override
		{
			m_inputBuffer->getBoundMemory().memory->unmap();
			m_outputBuffer->getBoundMemory().memory->unmap();
			m_inputPtr = nullptr;
			m_outputPtr = nullptr;
			return device_base_t::onAppTerminated();
		}
};


NBL_MAIN_FUNC(ScanAppendUnitTestApp)