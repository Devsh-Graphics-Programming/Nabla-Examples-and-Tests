#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "app_resources/common.hlsl"

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;
using namespace video;

// method emulations on the CPU, to verify the results of the GPU methods
template<class Binop>
struct emulatedReduction
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		const type_t red = std::reduce(in,in+itemCount,Binop::identity,Binop());
		std::fill(out,out+itemCount,red);
	}

	static inline constexpr const char* name = "reduction";
};
template<class Binop>
struct emulatedScanInclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::inclusive_scan(in,in+itemCount,out,Binop());
	}
	static inline constexpr const char* name = "inclusive_scan";
};
template<class Binop>
struct emulatedScanExclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::exclusive_scan(in,in+itemCount,out,Binop::identity,Binop());
	}
	static inline constexpr const char* name = "exclusive_scan";
};

class ArithmeticUnitTestApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
	ArithmeticUnitTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		transferDownQueue = getTransferDownQueue();
		computeQueue = getComputeQueue();

		// TODO: get the element count from argv
		const uint32_t elementCount = Output<>::ScanElementCount;
		// populate our random data buffer on the CPU and create a GPU copy
		inputData = new uint32_t[elementCount];
		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			std::mt19937 randGenerator(0xdeadbeefu);
			for (uint32_t i = 0u; i < elementCount; i++)
				inputData[i] = randGenerator(); // TODO: change to using xoroshiro, then we can skip having the input buffer at all

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(Output<>::data[0]) * elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuinputDataBuffer = m_utils->createFilledDeviceLocalBufferOnDedMem(
				{.queue=getTransferUpQueue()},
				std::move(inputDataBufferCreationParams),
				inputData
			);
		}

		// create 8 buffers for 8 operations
		for (auto i=0u; i<OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t) + gpuinputDataBuffer->getSize();
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get());
			assert(bufferMem.isValid());
		}

		// create Descriptor Set and Pipeline Layout
		{
			// create Descriptor Set Layout
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[2];
				for (uint32_t i = 0u; i < 2; i++)
					binding[i] = { i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr };
				binding[1].count = OutputBufferCount;
				dsLayout = m_device->createDescriptorSetLayout(binding);
			}

			// set and transient pool
			auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,{&dsLayout.get(),1});
			descriptorSet = descPool->createDescriptorSet(smart_refctd_ptr(dsLayout));
			{
				IGPUDescriptorSet::SDescriptorInfo infos[1+OutputBufferCount];
				infos[0].desc = gpuinputDataBuffer;
				infos[0].info.buffer = { 0u,gpuinputDataBuffer->getSize() };
				for (uint32_t i = 1u; i <= OutputBufferCount; i++)
				{
					auto buff = outputBuffers[i - 1];
					infos[i].info.buffer = { 0u,buff->getSize() };
					infos[i].desc = std::move(buff); // save an atomic in the refcount

				}

				IGPUDescriptorSet::SWriteDescriptorSet writes[2];
				for (uint32_t i=0u; i<2; i++)
					writes[i] = {descriptorSet.get(),i,0u,1u,infos+i};
				writes[1].count = OutputBufferCount;

				m_device->updateDescriptorSets(2, writes, 0u, nullptr);
			}

			pipelineLayout = m_device->createPipelineLayout({},std::move(dsLayout));
		}

		const auto spirv_isa_cache_path = localOutputCWD/"spirv_isa_cache.bin";
		// enclose to make sure file goes out of scope and we can reopen it
		{
			smart_refctd_ptr<const IFile> spirv_isa_cache_input;
			// try to load SPIR-V to ISA cache
			{
				ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
				m_system->createFile(fileCreate,spirv_isa_cache_path,IFile::ECF_READ|IFile::ECF_MAPPABLE|IFile::ECF_COHERENT);
				if (auto lock=fileCreate.acquire())
					spirv_isa_cache_input = *lock;
			}
			// create the cache
			{
				std::span<const uint8_t> spirv_isa_cache_data = {};
				if (spirv_isa_cache_input)
					spirv_isa_cache_data = {reinterpret_cast<const uint8_t*>(spirv_isa_cache_input->getMappedPointer()),spirv_isa_cache_input->getSize()};
				else
					m_logger->log("Failed to load SPIR-V 2 ISA cache!",ILogger::ELL_PERFORMANCE);
				// Normally we'd deserialize a `ICPUPipelineCache` properly and pass that instead
				m_spirv_isa_cache = m_device->createPipelineCache(spirv_isa_cache_data);
			}
		}
		{
			// TODO: rename `deleteDirectory` to just `delete`? and a `IFile::setSize()` ?
			m_system->deleteDirectory(spirv_isa_cache_path);
			ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
			m_system->createFile(fileCreate,spirv_isa_cache_path,IFile::ECF_WRITE);
			// I can be relatively sure I'll succeed to acquire the future, the pointer to created file might be null though.
			m_spirv_isa_cache_output=*fileCreate.acquire();
			if (!m_spirv_isa_cache_output)
				logFail("Failed to Create SPIR-V to ISA cache file.");
		}

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams = {};
			lparams.logger = m_logger.get();
			lparams.workingDirectory = "";
			auto bundle = m_assetMgr->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SHADER)
			{
				m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
				exit(-1);
			}
			auto firstAssetInBundle = bundle.getContents()[0];
			return smart_refctd_ptr_static_cast<ICPUShader>(firstAssetInBundle);
		};

		auto subgroupTestSource = getShaderSource("app_resources/testSubgroup.comp.hlsl");
		auto workgroupTestSource = getShaderSource("app_resources/testWorkgroup.comp.hlsl");
		// now create or retrieve final resources to run our tests
		sema = m_device->createSemaphore(timelineValue);
		resultsBuffer = make_smart_refctd_ptr<ICPUBuffer>(outputBuffers[0]->getSize());
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(),IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY,{&cmdbuf,1}))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}

		const auto MaxWorkgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		for (auto subgroupSize=MinSubgroupSize; subgroupSize <= MaxSubgroupSize; subgroupSize *= 2u)
		{
			const uint8_t subgroupSizeLog2 = hlsl::findMSB(subgroupSize);
			for (uint32_t workgroupSize = subgroupSize; workgroupSize <= MaxWorkgroupSize; workgroupSize += subgroupSize)
			{
				// make sure renderdoc captures everything for debugging
				computeQueue->startCapture();
				m_logger->log("Testing Workgroup Size %u with Subgroup Size %u", ILogger::ELL_INFO, workgroupSize, subgroupSize);

				bool passed = true;
				// TODO async the testing
				passed = runTest<emulatedReduction, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanInclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanExclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				for (uint32_t itemsPerWG = workgroupSize; itemsPerWG > workgroupSize - subgroupSize; itemsPerWG--)
				{
					m_logger->log("Testing Item Count %u", ILogger::ELL_INFO, itemsPerWG);
					passed = runTest<emulatedReduction, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanInclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanExclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
				}
				computeQueue->endCapture();

				// save cache every now and then	
				{
					auto cpu = m_spirv_isa_cache->convertToCPUCache();
					// Normally we'd beautifully JSON serialize the thing, allow multiple devices & drivers + metadata
					auto bin = cpu->getEntries().begin()->second.bin;
					IFile::success_t success;
					m_spirv_isa_cache_output->write(success,bin->data(),0ull,bin->size());
					if (!success)
						logFail("Could not write Create SPIR-V to ISA cache to disk!");
				}
			}
		}

		return true;
	}

	virtual bool onAppTerminated() override
	{
		m_logger->log("==========Result==========", ILogger::ELL_INFO);
		m_logger->log("Fail Count: %u", ILogger::ELL_INFO, totalFailCount);
		delete[] inputData;
		return true;
	}

	// the unit test is carried out on init
	void workLoopBody() override {}

	//
	bool keepRunning() override { return false; }

private:
	void logTestOutcome(bool passed, uint32_t workgroupSize)
	{
		if (passed)
			m_logger->log("Passed test #%u", ILogger::ELL_INFO, workgroupSize);
		else
		{
			totalFailCount++;
			m_logger->log("Failed test #%u", ILogger::ELL_ERROR, workgroupSize);
		}
	}

	// create pipeline (specialized every test) [TODO: turn into a future/async]
	smart_refctd_ptr<IGPUComputePipeline> createPipeline(const ICPUShader* overridenUnspecialized, const uint8_t subgroupSizeLog2)
	{
		auto shader = m_device->createShader(overridenUnspecialized);
		IGPUComputePipeline::SCreationParams params = {};
		params.layout = pipelineLayout.get();
		params.shader = {
			.entryPoint = "main",
			.shader = shader.get(),
			.entries = nullptr,
			.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(subgroupSizeLog2),
			.requireFullSubgroups = true
		};
		core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		if (!m_device->createComputePipelines(m_spirv_isa_cache.get(),{&params,1},&pipeline))
			return nullptr;
		return pipeline;
	}

	/*template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u)
	{
		return true;
	}*/

	template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u)
	{
		std::string arith_name = Arithmetic<bit_xor<float>>::name;

		smart_refctd_ptr<ICPUShader> overridenUnspecialized;
		if constexpr (WorkgroupTest)
		{
			overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n#define ITEMS_PER_WG %d\n",
				(("workgroup::") + arith_name).c_str(), workgroupSize, itemsPerWG
			);
		}
		else
		{
			itemsPerWG = workgroupSize;
			overridenUnspecialized = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n",
				(("subgroup::") + arith_name).c_str(), workgroupSize
			);
		}
		auto pipeline = createPipeline(overridenUnspecialized.get(),subgroupSizeLog2);

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		const uint32_t workgroupCount = elementCount / itemsPerWG;
		cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
		cmdbuf->bindComputePipeline(pipeline.get());
		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &descriptorSet.get());
		cmdbuf->dispatch(workgroupCount, 1, 1);
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t memoryBarrier[OutputBufferCount];
			for (auto i=0u; i<OutputBufferCount; i++)
			{
				memoryBarrier[i] = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
							// in theory we don't need the HOST BITS cause we block on a semaphore but might as well add them
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT|PIPELINE_STAGE_FLAGS::HOST_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS|ACCESS_FLAGS::HOST_READ_BIT
						}
					},
					.range = {0ull,outputBuffers[i]->getSize(),outputBuffers[i]}
				};
			}
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = {.memBarriers={},.bufBarriers=memoryBarrier};
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE,info);
		}
		cmdbuf->end();

		const IQueue::SSubmitInfo::SSemaphoreInfo signal[1] = {{.semaphore=sema.get(),.value=++timelineValue}};
		const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = {{.cmdbuf=cmdbuf.get()}};
		const IQueue::SSubmitInfo submits[1] = {{.commandBuffers=cmdbufs,.signalSemaphores=signal}};
		computeQueue->submit(submits);
		const ISemaphore::SWaitInfo wait[1] = {{.semaphore=sema.get(),.value=timelineValue}};
		m_device->blockForSemaphores(wait);

		// check results
		bool passed = validateResults<Arithmetic, bit_and<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount);
		passed = validateResults<Arithmetic, bit_xor<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		passed = validateResults<Arithmetic, bit_or<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		passed = validateResults<Arithmetic, plus<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		passed = validateResults<Arithmetic, multiplies<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		passed = validateResults<Arithmetic, minimum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		passed = validateResults<Arithmetic, maximum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;
		if constexpr (WorkgroupTest)
			passed = validateResults<Arithmetic, ballot<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount) && passed;

		return passed;
	}

	//returns true if result matches
	template<template<class> class Arithmetic, class Binop, bool WorkgroupTest>
	bool validateResults(const uint32_t itemsPerWG, const uint32_t workgroupCount)
	{
		bool success = true;

		// download data
		const SBufferRange<IGPUBuffer> bufferRange = {0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex]};
		m_utils->downloadBufferRangeViaStagingBufferAutoSubmit({.queue=transferDownQueue},bufferRange,resultsBuffer->getPointer());

		using type_t = typename Binop::type_t;
		const auto dataFromBuffer = reinterpret_cast<const uint32_t*>(resultsBuffer->getPointer());
		const auto subgroupSize = dataFromBuffer[0];
		if (subgroupSize<nbl::hlsl::subgroup::MinSubgroupSize || subgroupSize>nbl::hlsl::subgroup::MaxSubgroupSize)
		{
			m_logger->log("Unexpected Subgroup Size %u", ILogger::ELL_ERROR, subgroupSize);
			return false;
		}

		const auto testData = reinterpret_cast<const type_t*>(dataFromBuffer + 1);
		// TODO: parallel for (the temporary values need to be threadlocal or what?)
		// now check if the data obtained has valid values
		type_t* tmp = new type_t[itemsPerWG];
		type_t* ballotInput = new type_t[itemsPerWG];
		for (uint32_t workgroupID = 0u; success && workgroupID < workgroupCount; workgroupID++)
		{
			const auto workgroupOffset = workgroupID * itemsPerWG;

			if constexpr (WorkgroupTest)
			{
				if constexpr (std::is_same_v<ballot<type_t>, Binop>)
				{
					for (auto i = 0u; i < itemsPerWG; i++)
						ballotInput[i] = inputData[i + workgroupOffset] & 0x1u;
					Arithmetic<Binop>::impl(tmp, ballotInput, itemsPerWG);
				}
				else
					Arithmetic<Binop>::impl(tmp, inputData + workgroupOffset, itemsPerWG);
			}
			else
			{
				for (uint32_t pseudoSubgroupID = 0u; pseudoSubgroupID < itemsPerWG; pseudoSubgroupID += subgroupSize)
					Arithmetic<Binop>::impl(tmp + pseudoSubgroupID, inputData + workgroupOffset + pseudoSubgroupID, subgroupSize);
			}

			for (uint32_t localInvocationIndex = 0u; localInvocationIndex < itemsPerWG; localInvocationIndex++)
			{
				const auto globalInvocationIndex = workgroupOffset + localInvocationIndex;
				const auto cpuVal = tmp[localInvocationIndex];
				const auto gpuVal = testData[globalInvocationIndex];
				if (cpuVal != gpuVal)
				{
					m_logger->log(
						"Failed test #%d  (%s)  (%s) Expected %u got %u for workgroup %d and localinvoc %d",
						ILogger::ELL_ERROR, itemsPerWG, WorkgroupTest ? "workgroup" : "subgroup", Binop::name,
						cpuVal, gpuVal, workgroupID, localInvocationIndex
					);
					success = false;
					break;
				}
			}
		}
		delete[] ballotInput;
		delete[] tmp;

		return success;
	}

	IQueue* transferDownQueue;
	IQueue* computeQueue;
	smart_refctd_ptr<IGPUPipelineCache> m_spirv_isa_cache;
	smart_refctd_ptr<IFile> m_spirv_isa_cache_output;

	uint32_t* inputData = nullptr;
	constexpr static inline uint32_t OutputBufferCount = 8u;
	smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];
	smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;

	smart_refctd_ptr<ISemaphore> sema;
	uint64_t timelineValue = 0;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;
};

NBL_MAIN_FUNC(ArithmeticUnitTestApp)