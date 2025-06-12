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
		const type_t red = std::reduce(in, in + itemCount, Binop::identity, Binop());
		std::fill(out, out + itemCount, red);
	}

	static inline constexpr const char* name = "reduction";
};
template<class Binop>
struct emulatedScanInclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::inclusive_scan(in, in + itemCount, out, Binop());
	}
	static inline constexpr const char* name = "inclusive_scan";
};
template<class Binop>
struct emulatedScanExclusive
{
	using type_t = typename Binop::type_t;

	static inline void impl(type_t* out, const type_t* in, const uint32_t itemCount)
	{
		std::exclusive_scan(in, in + itemCount, out, Binop::identity, Binop());
	}
	static inline constexpr const char* name = "exclusive_scan";
};

class ComputeScanApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = application_templates::BasicMultiQueueApplication;
    using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
    ComputeScanApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
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
		const uint32_t elementCount = 1024*12;
		// populate our random data buffer on the CPU and create a GPU copy
		inputData = new uint32_t[elementCount];
		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			std::mt19937 randGenerator(0xdeadbeefu);
			std::uniform_int_distribution rng(0, 100);
			for (uint32_t i = 0u; i < elementCount; i++)
				inputData[i] = 1;// rng(randGenerator); // TODO: change to using xoroshiro, then we can skip having the input buffer at all

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(uint32_t) * elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{ .queue = getTransferUpQueue() },
				std::move(inputDataBufferCreationParams),
				inputData
			).move_into(gpuinputDataBuffer);
		}

		// create 8 buffers for 8 operations
		for (auto i = 0u; i < OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t) + gpuinputDataBuffer->getSize();
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			assert(bufferMem.isValid());
		}

		// create scratch memory buffer (not the same as scratch shared memory)
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t) * (elementCount / MinSubgroupSize);
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			scratchBuffer = m_device->createBuffer(std::move(params));
			auto mreq = scratchBuffer->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, scratchBuffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			assert(bufferMem.isValid());
		}

		pc.pInputBuf = gpuinputDataBuffer->getDeviceAddress();
		for (auto i = 0u; i < OutputBufferCount; i++)
			pc.pOutputBuf[i] = outputBuffers[i]->getDeviceAddress();
		pc.pScratchBuf = scratchBuffer->getDeviceAddress();

		// create Pipeline Layout
		{
			SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0,.size = sizeof(PushConstantData) };
			pipelineLayout = m_device->createPipelineLayout({ &pcRange, 1 });
		}

		const auto spirv_isa_cache_path = localOutputCWD / "spirv_isa_cache.bin";
		// enclose to make sure file goes out of scope and we can reopen it
		{
			smart_refctd_ptr<const IFile> spirv_isa_cache_input;
			// try to load SPIR-V to ISA cache
			{
				ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
				m_system->createFile(fileCreate, spirv_isa_cache_path, IFile::ECF_READ | IFile::ECF_MAPPABLE | IFile::ECF_COHERENT);
				if (auto lock = fileCreate.acquire())
					spirv_isa_cache_input = *lock;
			}
			// create the cache
			{
				std::span<const uint8_t> spirv_isa_cache_data = {};
				if (spirv_isa_cache_input)
					spirv_isa_cache_data = { reinterpret_cast<const uint8_t*>(spirv_isa_cache_input->getMappedPointer()),spirv_isa_cache_input->getSize() };
				else
					m_logger->log("Failed to load SPIR-V 2 ISA cache!", ILogger::ELL_PERFORMANCE);
				// Normally we'd deserialize a `ICPUPipelineCache` properly and pass that instead
				m_spirv_isa_cache = m_device->createPipelineCache(spirv_isa_cache_data);
			}
		}
		{
			// TODO: rename `deleteDirectory` to just `delete`? and a `IFile::setSize()` ?
			m_system->deleteDirectory(spirv_isa_cache_path);
			ISystem::future_t<smart_refctd_ptr<IFile>> fileCreate;
			m_system->createFile(fileCreate, spirv_isa_cache_path, IFile::ECF_WRITE);
			// I can be relatively sure I'll succeed to acquire the future, the pointer to created file might be null though.
			m_spirv_isa_cache_output = *fileCreate.acquire();
			if (!m_spirv_isa_cache_output)
				logFail("Failed to Create SPIR-V to ISA cache file.");
		}

		//if (m_physicalDevice->getProperties().limits.vu)

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
			{
				IAssetLoader::SAssetLoadParams lparams = {};
				lparams.logger = m_logger.get();
				lparams.workingDirectory = "";
				auto bundle = m_assetMgr->getAsset(filePath, lparams);
				if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SHADER)
				{
					m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
					exit(-1);
				}
				auto firstAssetInBundle = bundle.getContents()[0];
				return smart_refctd_ptr_static_cast<ICPUShader>(firstAssetInBundle);
			};

		auto scanTestSource = getShaderSource("app_resources/testScans.comp.hlsl");
		// now create or retrieve final resources to run our tests
		sema = m_device->createSemaphore(timelineValue);
		statusBuffer = ICPUBuffer::create({ scratchBuffer->getSize() });
		resultsBuffer = ICPUBuffer::create({ outputBuffers[0]->getSize() });
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}

		const auto MaxWorkgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		for (auto subgroupSize = MinSubgroupSize; subgroupSize <= MaxSubgroupSize; subgroupSize *= 2u)
		{
			const uint8_t subgroupSizeLog2 = hlsl::findMSB(subgroupSize);
			for (uint32_t workgroupSize = subgroupSize; workgroupSize <= MaxWorkgroupSize; workgroupSize *= 2u)
			{
				// make sure renderdoc captures everything for debugging
				m_api->startCapture();
				m_logger->log("Testing Workgroup Size %u with Subgroup Size %u", ILogger::ELL_INFO, workgroupSize, subgroupSize);

				for (uint32_t j = 0; j < ItemsPerInvocations.size(); j++)
				{
					const uint32_t itemsPerInvocation = ItemsPerInvocations[j];
					const uint32_t itemsPerWorkgroup = calculateItemsPerWorkgroup(workgroupSize, subgroupSize, itemsPerInvocation);
					m_logger->log("Testing Items per Invocation %u", ILogger::ELL_INFO, itemsPerInvocation);
					bool passed = true;
					passed = runTest<emulatedReduction>(scanTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWorkgroup, itemsPerInvocation) && passed;
					logTestOutcome(passed, workgroupSize);
					//passed = runTest<emulatedScanInclusive>(scanTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWorkgroup, itemsPerInvocation) && passed;
					//logTestOutcome(passed, workgroupSize);
					//passed = runTest<emulatedScanExclusive>(scanTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWorkgroup, itemsPerInvocation) && passed;
					//logTestOutcome(passed, workgroupSize);
				}
				m_api->endCapture();

				// save cache every now and then	
				{
					auto cpu = m_spirv_isa_cache->convertToCPUCache();
					// Normally we'd beautifully JSON serialize the thing, allow multiple devices & drivers + metadata
					auto bin = cpu->getEntries().begin()->second.bin;
					IFile::success_t success;
					m_spirv_isa_cache_output->write(success, bin->data(), 0ull, bin->size());
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

	// reflects calculations in workgroup2::ArithmeticConfiguration
	uint32_t calculateItemsPerWorkgroup(const uint32_t workgroupSize, const uint32_t subgroupSize, const uint32_t itemsPerInvocation)
	{
		if (workgroupSize <= subgroupSize)
			return workgroupSize * itemsPerInvocation;

		const uint8_t subgroupSizeLog2 = hlsl::findMSB(subgroupSize);
		const uint8_t workgroupSizeLog2 = hlsl::findMSB(workgroupSize);

		const uint16_t levels = (workgroupSizeLog2 == subgroupSizeLog2) ? 1 :
			(workgroupSizeLog2 > subgroupSizeLog2 * 2 + 2) ? 3 : 2;

		const uint16_t itemsPerInvocationProductLog2 = max(workgroupSizeLog2 - subgroupSizeLog2 * levels, 0);
		uint16_t itemsPerInvocation1 = (levels == 3) ? min(itemsPerInvocationProductLog2, 2) : itemsPerInvocationProductLog2;
		itemsPerInvocation1 = uint16_t(1u) << itemsPerInvocation1;

		uint32_t virtualWorkgroupSize = 1u << max(subgroupSizeLog2 * levels, workgroupSizeLog2);

		return itemsPerInvocation * virtualWorkgroupSize;
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
		if (!m_device->createComputePipelines(m_spirv_isa_cache.get(), { &params,1 }, &pipeline))
			return nullptr;
		return pipeline;
	}

	template<template<class> class Arithmetic>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, const uint32_t itemsPerWG, const uint32_t itemsPerInvoc)
	{
		std::string arith_name = Arithmetic<arithmetic::bit_xor<float>>::name;
		const uint32_t workgroupSizeLog2 = hlsl::findMSB(workgroupSize);

		auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
		CHLSLCompiler::SOptions options = {};
		options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
		options.spirvOptimizer = nullptr;
#ifndef _NBL_DEBUG
		ISPIRVOptimizer::E_OPTIMIZER_PASS optPasses = ISPIRVOptimizer::EOP_STRIP_DEBUG_INFO;
		auto opt = make_smart_refctd_ptr<ISPIRVOptimizer>(std::span<ISPIRVOptimizer::E_OPTIMIZER_PASS>(&optPasses, 1));
		options.spirvOptimizer = opt.get();
#else
		options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
#endif
		options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
		options.preprocessorOptions.logger = m_logger.get();

		auto* includeFinder = compiler->getDefaultIncludeFinder();
		includeFinder->addSearchPath("nbl/builtin/hlsl/jit", core::make_smart_refctd_ptr<CJITIncludeLoader>(m_physicalDevice->getLimits(), m_device->getEnabledFeatures()));
		options.preprocessorOptions.includeFinder = includeFinder;

		const std::string definitions[5] = {
			"scan::" + arith_name,
			std::to_string(workgroupSizeLog2),
			std::to_string(itemsPerInvoc),
			std::to_string(subgroupSizeLog2),
			std::to_string(arith_name == "reduction")
		};

		const IShaderCompiler::SMacroDefinition defines[5] = {
			{ "OPERATION", definitions[0] },
			{ "WORKGROUP_SIZE_LOG2", definitions[1] },
			{ "ITEMS_PER_INVOCATION", definitions[2] },
			{ "SUBGROUP_SIZE_LOG2", definitions[3] },
			{ "IS_REDUCTION", definitions[4] }
		};
		options.preprocessorOptions.extraDefines = { defines, defines + 5 };

		auto overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		auto pipeline = createPipeline(overriddenUnspecialized.get(), subgroupSizeLog2);

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		uint32_t workgroupCount = elementCount / itemsPerWG;
		workgroupCount = min(workgroupCount, m_physicalDevice->getLimits().maxComputeWorkGroupCount[0]);

		cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);

		// clear buffers
		cmdbuf->fillBuffer({ .size = scratchBuffer->getSize(), .buffer = scratchBuffer }, 0u);
		for (uint32_t i = 0; i < OutputBufferCount; i++)
			cmdbuf->fillBuffer({ .size = outputBuffers[i]->getSize(), .buffer = outputBuffers[i] }, 0u);

		cmdbuf->bindComputePipeline(pipeline.get());
		cmdbuf->pushConstants(pipelineLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PushConstantData), &pc);
		cmdbuf->dispatch(workgroupCount, 1, 1);
		{
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t memoryBarrier[OutputBufferCount];
			for (auto i = 0u; i < OutputBufferCount; i++)
			{
				memoryBarrier[i] = {
					.barrier = {
						.dep = {
							.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
							.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
							// in theory we don't need the HOST BITS cause we block on a semaphore but might as well add them
							.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT | PIPELINE_STAGE_FLAGS::HOST_BIT,
							.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS | ACCESS_FLAGS::HOST_READ_BIT
						}
					},
					.range = {0ull,outputBuffers[i]->getSize(),outputBuffers[i]}
				};
			}
			IGPUCommandBuffer::SPipelineBarrierDependencyInfo info = { .memBarriers = {},.bufBarriers = memoryBarrier };
			cmdbuf->pipelineBarrier(asset::E_DEPENDENCY_FLAGS::EDF_NONE, info);
		}
		cmdbuf->end();

		const IQueue::SSubmitInfo::SSemaphoreInfo signal[1] = { {.semaphore = sema.get(),.value = ++timelineValue} };
		const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[1] = { {.cmdbuf = cmdbuf.get()} };
		const IQueue::SSubmitInfo submits[1] = { {.commandBuffers = cmdbufs,.signalSemaphores = signal} };
		computeQueue->submit(submits);
		const ISemaphore::SWaitInfo wait[1] = { {.semaphore = sema.get(),.value = timelineValue} };
		m_device->blockForSemaphores(wait);

		const uint32_t subgroupSize = 1u << subgroupSizeLog2;
		// check results
		bool passed = true;
		passed = validateResults<Arithmetic, arithmetic::bit_and<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize);
		passed = validateResults<Arithmetic, arithmetic::bit_xor<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;
		passed = validateResults<Arithmetic, arithmetic::bit_or<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;
		passed = validateResults<Arithmetic, arithmetic::plus<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;
		passed = validateResults<Arithmetic, arithmetic::multiplies<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;
		passed = validateResults<Arithmetic, arithmetic::minimum<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;
		passed = validateResults<Arithmetic, arithmetic::maximum<uint32_t> >(itemsPerWG, workgroupCount, subgroupSize) && passed;

		return passed;
	}

	//returns true if result matches
	template<template<class> class Arithmetic, class Binop>
	bool validateResults(const uint32_t itemsPerWG, const uint32_t workgroupCount, const uint32_t subgroupSize)
	{
		bool success = true;
		std::string arith_name = Arithmetic<arithmetic::bit_xor<float>>::name;
		const bool isReduction = arith_name == "reduction";

		// download data
	    {
	        const SBufferRange<IGPUBuffer> bufferRange = { 0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex] };
		    m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(SIntendedSubmitInfo{ .queue = transferDownQueue }, bufferRange, resultsBuffer->getPointer());
	    }
		{
			const SBufferRange<IGPUBuffer> bufferRange = { 0u, statusBuffer->getSize(), scratchBuffer };
			m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(SIntendedSubmitInfo{ .queue = transferDownQueue }, bufferRange, statusBuffer->getPointer());
		}

		using type_t = typename Binop::type_t;
		const auto dataFromBuffer = reinterpret_cast<const uint32_t*>(resultsBuffer->getPointer());

		const auto testData = reinterpret_cast<const type_t*>(dataFromBuffer + 1);
		const auto scratchData = reinterpret_cast<const uint32_t*>(statusBuffer->getPointer());

		// TODO: parallel for (the temporary values need to be threadlocal or what?)
			// now check if the data obtained has valid values
		const uint32_t elementCount = itemsPerWG * workgroupCount;
		type_t* tmp = new type_t[elementCount];

		Arithmetic<Binop>::impl(tmp, inputData, elementCount);

		if (isReduction)
		{
			const auto cpuVal = tmp[0];
			const auto gpuVal = testData[0];

			const auto gpuStatus = scratchData[Binop::BindingIndex];
			const uint32_t expectedStatus = workgroupCount;

			if (cpuVal != gpuVal || expectedStatus != gpuStatus)
			{
				m_logger->log(
					"Failed test #%d (%s) Expected value %u got %u, expected status %u got %u",
					ILogger::ELL_ERROR, itemsPerWG, Binop::name,
					cpuVal, gpuVal, expectedStatus, gpuStatus
				);
				success = false;
			}
		}
		else
		{
			for (uint32_t index = 0u; index < elementCount; index++)
			{
				const auto cpuVal = tmp[index];
				const auto gpuVal = testData[index];
				if (cpuVal != gpuVal)
				{
					m_logger->log(
						"Failed test #%d (%s) Expected %u got %u for item %d",
						ILogger::ELL_ERROR, itemsPerWG, Binop::name,
						cpuVal, gpuVal, index
					);
					success = false;
					break;
				}
			}
		}
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
	smart_refctd_ptr<IGPUBuffer> scratchBuffer;
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
	PushConstantData pc;

	smart_refctd_ptr<ISemaphore> sema;
	uint64_t timelineValue = 0;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> statusBuffer;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;

	constexpr static inline std::array<uint32_t, 4> ItemsPerInvocations = { 1, 2, 3, 4 };
};

NBL_MAIN_FUNC(ComputeScanApp)