// TODO: copyright notice


#include "nbl/examples/examples.hpp"

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/workgroup2/arithmetic_config.hlsl"
#include "nbl/builtin/hlsl/subgroup2/arithmetic_params.hlsl"


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

class Workgroup2ScanTestApp final : public application_templates::BasicMultiQueueApplication, public examples::BuiltinResourcesApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = examples::BuiltinResourcesApplication;

public:
	Workgroup2ScanTestApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
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
		const uint32_t elementCount = 1024 * 1024;
		// populate our random data buffer on the CPU and create a GPU copy
		inputData = new uint32_t[elementCount];
		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			std::mt19937 randGenerator(0xdeadbeefu);
			for (uint32_t i = 0u; i < elementCount; i++)
				inputData[i] = randGenerator(); // TODO: change to using xoroshiro, then we can skip having the input buffer at all

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(uint32_t) * elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
			m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{.queue=getTransferUpQueue()},
				std::move(inputDataBufferCreationParams),
				inputData
			).move_into(gpuinputDataBuffer);
		}

		// create 8 buffers for 8 operations
		for (auto i=0u; i<OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = gpuinputDataBuffer->getSize();
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			outputBuffers[i] = m_device->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = m_device->allocate(mreq, outputBuffers[i].get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
			assert(bufferMem.isValid());
		}
		pc.pInputBuf = gpuinputDataBuffer->getDeviceAddress();
		for (uint32_t i = 0; i < OutputBufferCount; i++)
			pc.pOutputBuf[i] = outputBuffers[i]->getDeviceAddress();

		// create Pipeline Layout
		{
			SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0,.size = sizeof(PushConstantData) };
			pipelineLayout = m_device->createPipelineLayout({&pcRange, 1});
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
			return smart_refctd_ptr_static_cast<IShader>(firstAssetInBundle);
		};

		auto subgroupTestSource = getShaderSource("app_resources/testSubgroup.comp.hlsl");
		auto workgroupTestSource = getShaderSource("app_resources/testWorkgroup.comp.hlsl");
		// now create or retrieve final resources to run our tests
		sema = m_device->createSemaphore(timelineValue);
		resultsBuffer = ICPUBuffer::create({ outputBuffers[0]->getSize() });
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
		for (uint32_t useNative = 0; useNative <= uint32_t(m_physicalDevice->getProperties().limits.shaderSubgroupArithmetic); useNative++)
		{
			if (useNative)
				m_logger->log("Testing with native subgroup arithmetic", ILogger::ELL_INFO);
			else
				m_logger->log("Testing with emulated subgroup arithmetic", ILogger::ELL_INFO);

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
						uint32_t itemsPerWG = workgroupSize * itemsPerInvocation;
						m_logger->log("Testing Items per Invocation %u", ILogger::ELL_INFO, itemsPerInvocation);
						bool passed = true;
						passed = runTest<emulatedReduction, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);
						passed = runTest<emulatedScanInclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);
						passed = runTest<emulatedScanExclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);

						hlsl::workgroup2::SArithmeticConfiguration wgConfig;
					    wgConfig.init(hlsl::findMSB(workgroupSize), subgroupSizeLog2, itemsPerInvocation);
						itemsPerWG = wgConfig.VirtualWorkgroupSize * wgConfig.ItemsPerInvocation_0;
						m_logger->log("Testing Item Count %u", ILogger::ELL_INFO, itemsPerWG);
						passed = runTest<emulatedReduction, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);
						passed = runTest<emulatedScanInclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);
						passed = runTest<emulatedScanExclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, bool(useNative), itemsPerWG, itemsPerInvocation) && passed;
						logTestOutcome(passed, itemsPerWG);
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
	smart_refctd_ptr<IGPUComputePipeline> createPipeline(const IShader* overridenUnspecialized, const uint8_t subgroupSizeLog2)
	{
		auto shader = m_device->compileShader({ overridenUnspecialized });
		IGPUComputePipeline::SCreationParams params = {};
		params.layout = pipelineLayout.get();
		params.shader = {
			.shader = shader.get(),
			.entryPoint = "main",
			.requiredSubgroupSize = static_cast<IPipelineBase::SUBGROUP_SIZE>(subgroupSizeLog2),
			.entries = nullptr,
		};
		params.cached.requireFullSubgroups = true;
		core::smart_refctd_ptr<IGPUComputePipeline> pipeline;
		if (!m_device->createComputePipelines(m_spirv_isa_cache.get(),{&params,1},&pipeline))
			return nullptr;
		return pipeline;
	}

	template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const IShader>& source, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, bool useNative, uint32_t itemsPerWG, uint32_t itemsPerInvoc = 1u)
	{
		std::string arith_name = Arithmetic<arithmetic::bit_xor<float>>::name;
		const uint32_t workgroupSizeLog2 = hlsl::findMSB(workgroupSize);

		auto compiler = make_smart_refctd_ptr<asset::CHLSLCompiler>(smart_refctd_ptr(m_system));
		CHLSLCompiler::SOptions options = {};
		options.stage = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
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
		options.preprocessorOptions.includeFinder = includeFinder;

		smart_refctd_ptr<IShader> overriddenUnspecialized;
		if constexpr (WorkgroupTest)
		{
			hlsl::workgroup2::SArithmeticConfiguration wgConfig;
			wgConfig.init(hlsl::findMSB(workgroupSize), subgroupSizeLog2, itemsPerInvoc);

			const std::string definitions[3] = {
				"workgroup2::" + arith_name,
				wgConfig.getConfigTemplateStructString(),
				std::to_string(arith_name=="reduction")
			};

			const IShaderCompiler::SMacroDefinition defines[4] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_CONFIG_T", definitions[1] },
				{ "IS_REDUCTION", definitions[2] },
				{ "TEST_NATIVE", "1" }
			};
			if (useNative)
				options.preprocessorOptions.extraDefines = { defines, defines + 4 };
			else
				options.preprocessorOptions.extraDefines = { defines, defines + 3 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}
		else
		{
			hlsl::subgroup2::SArithmeticParams sgParams;
			sgParams.init(subgroupSizeLog2, itemsPerInvoc);

			const std::string definitions[3] = { 
				"subgroup2::" + arith_name,
				std::to_string(workgroupSize),
				sgParams.getParamTemplateStructString()
			};

			const IShaderCompiler::SMacroDefinition defines[4] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_SIZE", definitions[1] },
				{ "SUBGROUP_CONFIG_T", definitions[2] },
				{ "TEST_NATIVE", "1" }
			};
			if (useNative)
				options.preprocessorOptions.extraDefines = { defines, defines + 4 };
			else
				options.preprocessorOptions.extraDefines = { defines, defines + 3 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}

		auto pipeline = createPipeline(overriddenUnspecialized.get(),subgroupSizeLog2);

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		uint32_t workgroupCount = 1;// min(elementCount / itemsPerWG, m_physicalDevice->getLimits().maxComputeWorkGroupCount[0]);

		cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
		cmdbuf->bindComputePipeline(pipeline.get());
		cmdbuf->pushConstants(pipelineLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PushConstantData), &pc);
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

		const uint32_t subgroupSize = 1u << subgroupSizeLog2;
		// check results
		bool passed = validateResults<Arithmetic, arithmetic::bit_and<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc);
		passed = validateResults<Arithmetic, arithmetic::bit_xor<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, arithmetic::bit_or<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, arithmetic::plus<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, arithmetic::multiplies<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, arithmetic::minimum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, arithmetic::maximum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, subgroupSize, itemsPerInvoc) && passed;

		return passed;
	}

	//returns true if result matches
	template<template<class> class Arithmetic, class Binop, bool WorkgroupTest>
	bool validateResults(const uint32_t itemsPerWG, const uint32_t workgroupCount, const uint32_t subgroupSize, const uint32_t itemsPerInvoc)
	{
		bool success = true;

		// download data
		const SBufferRange<IGPUBuffer> bufferRange = {0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex]};
		m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(SIntendedSubmitInfo{.queue=transferDownQueue},bufferRange,resultsBuffer->getPointer());

		using type_t = typename Binop::type_t;
		const auto testData = reinterpret_cast<const uint32_t*>(resultsBuffer->getPointer());

		// TODO: parallel for (the temporary values need to be threadlocal or what?)
		// now check if the data obtained has valid values
		type_t* tmp = new type_t[itemsPerWG];
		for (uint32_t workgroupID = 0u; success && workgroupID < workgroupCount; workgroupID++)
		{
			if constexpr (WorkgroupTest)
			{
				const auto workgroupOffset = workgroupID * itemsPerWG;
				Arithmetic<Binop>::impl(tmp, inputData + workgroupOffset, itemsPerWG);

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
			else
			{
				const auto workgroupOffset = workgroupID * itemsPerWG;
				const auto workgroupSize = itemsPerWG / itemsPerInvoc;
				for (uint32_t pseudoSubgroupID = 0u; pseudoSubgroupID < workgroupSize; pseudoSubgroupID += subgroupSize)
					Arithmetic<Binop>::impl(tmp + pseudoSubgroupID * itemsPerInvoc, inputData + workgroupOffset + pseudoSubgroupID * itemsPerInvoc, subgroupSize * itemsPerInvoc);

				for (uint32_t localInvocationIndex = 0u; localInvocationIndex < workgroupSize; localInvocationIndex++)
				{
					const auto localOffset = localInvocationIndex * itemsPerInvoc;
					const auto globalInvocationIndex = workgroupOffset + localOffset;

					for (uint32_t itemInvocationIndex = 0u; itemInvocationIndex < itemsPerInvoc; itemInvocationIndex++)
					{
						const auto cpuVal = tmp[localOffset + itemInvocationIndex];
						const auto gpuVal = testData[globalInvocationIndex + itemInvocationIndex];
						if (cpuVal != gpuVal)
						{
							m_logger->log(
								"Failed test #%d  (%s)  (%s) Expected %u got %u for workgroup %d and localinvoc %d and iteminvoc %d",
								ILogger::ELL_ERROR, itemsPerWG, WorkgroupTest ? "workgroup" : "subgroup", Binop::name,
								cpuVal, gpuVal, workgroupID, localInvocationIndex, itemInvocationIndex
							);
							success = false;
							break;
						}
					}
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
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
	PushConstantData pc;

	smart_refctd_ptr<ISemaphore> sema;
	uint64_t timelineValue = 0;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;

	constexpr static inline std::array<uint32_t, 4> ItemsPerInvocations = { 1, 2, 3, 4 };
};

NBL_MAIN_FUNC(Workgroup2ScanTestApp)