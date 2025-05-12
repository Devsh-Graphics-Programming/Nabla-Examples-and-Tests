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

class Workgroup2ScanTestApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

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
					binding[i] = {{},i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,IShader::E_SHADER_STAGE::ESS_COMPUTE,1u,nullptr };
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
		for (auto subgroupSize=MinSubgroupSize; subgroupSize <= MaxSubgroupSize; subgroupSize *= 2u)
		{
			const uint8_t subgroupSizeLog2 = hlsl::findMSB(subgroupSize);
			for (uint32_t i = 0; i < WorkgroupSizes.size(); i++)
			{
				const uint32_t workgroupSize = WorkgroupSizes[i];
				// make sure renderdoc captures everything for debugging
				m_api->startCapture();
				m_logger->log("Testing Workgroup Size %u with Subgroup Size %u", ILogger::ELL_INFO, workgroupSize, subgroupSize);

				for (uint32_t j = 0; j < ItemsPerInvocations.size(); j++)
				{
					const uint32_t itemsPerInvocation = ItemsPerInvocations[j];
					m_logger->log("Testing Items per Invocation %u", ILogger::ELL_INFO, itemsPerInvocation);
					bool passed = true;
					passed = runTest<emulatedReduction, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
					logTestOutcome(passed, workgroupSize);
					passed = runTest<emulatedScanInclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
					logTestOutcome(passed, workgroupSize);
					passed = runTest<emulatedScanExclusive, false>(subgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, ~0u, itemsPerInvocation) && passed;
					logTestOutcome(passed, workgroupSize);

					const uint32_t itemsPerWG = workgroupSize <= subgroupSize ? workgroupSize * itemsPerInvocation : itemsPerInvocation * max(workgroupSize >> subgroupSizeLog2, subgroupSize) << subgroupSizeLog2;	// TODO use Config somehow
					m_logger->log("Testing Item Count %u", ILogger::ELL_INFO, itemsPerWG);
					passed = runTest<emulatedReduction, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG, itemsPerInvocation) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanInclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG, itemsPerInvocation) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanExclusive, true>(workgroupTestSource, elementCount, subgroupSizeLog2, workgroupSize, itemsPerWG, itemsPerInvocation) && passed;
					logTestOutcome(passed, itemsPerWG);
				}
				m_api->endCapture();
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
		if (!m_device->createComputePipelines(nullptr,{&params,1},&pipeline))
			return nullptr;
		return pipeline;
	}

	template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint8_t subgroupSizeLog2, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u, uint32_t itemsPerInvoc = 1u)
	{
		std::string arith_name = Arithmetic<bit_xor<float>>::name;
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

		smart_refctd_ptr<ICPUShader> overriddenUnspecialized;
		if constexpr (WorkgroupTest)
		{
			const std::string definitions[5] = {
				"workgroup2::" + arith_name,
				std::to_string(workgroupSizeLog2),
				std::to_string(itemsPerWG),
				std::to_string(itemsPerInvoc),
				std::to_string(subgroupSizeLog2)
			};

			const IShaderCompiler::SMacroDefinition defines[5] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_SIZE_LOG2", definitions[1] },
				{ "ITEMS_PER_WG", definitions[2] },
				{ "ITEMS_PER_INVOCATION", definitions[3] },
				{ "SUBGROUP_SIZE_LOG2", definitions[4] }
			};
			options.preprocessorOptions.extraDefines = { defines, defines + 5 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}
		else
		{
			const std::string definitions[4] = { 
				"subgroup2::" + arith_name,
				std::to_string(workgroupSize),
				std::to_string(itemsPerInvoc),
				std::to_string(subgroupSizeLog2)
			};

			const IShaderCompiler::SMacroDefinition defines[4] = {
				{ "OPERATION", definitions[0] },
				{ "WORKGROUP_SIZE", definitions[1] },
				{ "ITEMS_PER_INVOCATION", definitions[2] },
				{ "SUBGROUP_SIZE_LOG2", definitions[3] }
			};
			options.preprocessorOptions.extraDefines = { defines, defines + 4 };

			overriddenUnspecialized = compiler->compileToSPIRV((const char*)source->getContent()->getPointer(), options);
		}

		auto pipeline = createPipeline(overriddenUnspecialized.get(),subgroupSizeLog2);

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		uint32_t workgroupCount;
		if constexpr (WorkgroupTest)
			workgroupCount = elementCount / itemsPerWG;
		else
		{
			itemsPerWG = workgroupSize;
			workgroupCount = elementCount / (itemsPerWG * itemsPerInvoc);
		}	
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
		bool passed = validateResults<Arithmetic, bit_and<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc);
		passed = validateResults<Arithmetic, bit_xor<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, bit_or<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, plus<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, multiplies<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, minimum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;
		passed = validateResults<Arithmetic, maximum<uint32_t>, WorkgroupTest>(itemsPerWG, workgroupCount, itemsPerInvoc) && passed;

		return passed;
	}

	//returns true if result matches
	template<template<class> class Arithmetic, class Binop, bool WorkgroupTest>
	bool validateResults(const uint32_t itemsPerWG, const uint32_t workgroupCount, const uint32_t itemsPerInvoc)
	{
		bool success = true;

		// download data
		const SBufferRange<IGPUBuffer> bufferRange = {0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex]};
		m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(SIntendedSubmitInfo{.queue=transferDownQueue},bufferRange,resultsBuffer->getPointer());

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
		type_t* tmp;
		if constexpr (WorkgroupTest)
			tmp = new type_t[itemsPerWG];
		else
			tmp = new type_t[itemsPerWG * itemsPerInvoc];
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
				const auto workgroupOffset = workgroupID * itemsPerWG * itemsPerInvoc;
				for (uint32_t pseudoSubgroupID = 0u; pseudoSubgroupID < itemsPerWG; pseudoSubgroupID += subgroupSize)
					Arithmetic<Binop>::impl(tmp + pseudoSubgroupID * itemsPerInvoc, inputData + workgroupOffset + pseudoSubgroupID * itemsPerInvoc, subgroupSize * itemsPerInvoc);

				for (uint32_t localInvocationIndex = 0u; localInvocationIndex < itemsPerWG; localInvocationIndex++)
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

	constexpr static inline std::array<uint32_t, 4> WorkgroupSizes = { 32, 256, 512, 1024 };
	constexpr static inline std::array<uint32_t, 3> ItemsPerInvocations = { 1, 2, 4 };
};

NBL_MAIN_FUNC(Workgroup2ScanTestApp)