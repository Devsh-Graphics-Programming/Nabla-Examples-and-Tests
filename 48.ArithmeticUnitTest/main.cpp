#include "../common/CommonAPI.h"
#include "hlsl/common.hlsl"

// Thanks to our unified HLSL/C++ STD lib we're able to remove a whole load of code
template<typename T>
struct bit_and : nbl::hlsl::bit_and<T>
{
	static inline constexpr const char* name = "and";
};
template<typename T>
struct bit_or : nbl::hlsl::bit_or<T>
{
	static inline constexpr const char* name = "xor";
};
template<typename T>
struct bit_xor : nbl::hlsl::bit_xor<T>
{
	static inline constexpr const char* name = "or";
};
template<typename T>
struct plus : nbl::hlsl::plus<T>
{
	static inline constexpr const char* name = "add";
};
template<typename T>
struct multiplies : nbl::hlsl::multiplies<T>
{
	static inline constexpr const char* name = "mul";
};
template<typename T>
struct minimum : nbl::hlsl::minimum<T>
{
	static inline constexpr const char* name = "min";
};
template<typename T>
struct maximum : nbl::hlsl::maximum<T>
{
	static inline constexpr const char* name = "max";
};

template<typename T>
struct ballot : nbl::hlsl::plus<T>
{
	static inline constexpr const char* name = "bitcount";
};


using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;

//subgroup method emulations on the CPU, to verify the results of the GPU methods
template<class CRTP, typename T>
struct emulatedSubgroupCommon
{
	using type_t = T;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		for (uint32_t pseudoSubgroupID=0u; pseudoSubgroupID<workgroupSize; pseudoSubgroupID+=subgroupSize)
		{
			type_t* outSubgroupData = outputData+pseudoSubgroupID;
			const type_t* subgroupData = workgroupData+pseudoSubgroupID;
			CRTP::impl(outSubgroupData,subgroupData,core::min<uint32_t>(subgroupSize,workgroupSize-pseudoSubgroupID));
		}
	}
};
template<class OP>
struct emulatedSubgroupReduction : emulatedSubgroupCommon<emulatedSubgroupReduction<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		type_t red = subgroupData[0];
		for (auto i=1u; i<clampedSubgroupSize; i++)
			red = OP()(red,subgroupData[i]);
		std::fill(outSubgroupData,outSubgroupData+clampedSubgroupSize,red);
	}
	static inline constexpr const char* name = "subgroup reduction";
};
template<class OP>
struct emulatedSubgroupScanExclusive : emulatedSubgroupCommon<emulatedSubgroupScanExclusive<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		outSubgroupData[0u] = OP::identity;
		for (auto i=1u; i<clampedSubgroupSize; i++)
			outSubgroupData[i] = OP()(outSubgroupData[i-1u],subgroupData[i-1u]);
	}
	static inline constexpr const char* name = "subgroup exclusive scan";
};
template<class OP>
struct emulatedSubgroupScanInclusive : emulatedSubgroupCommon<emulatedSubgroupScanInclusive<OP>,typename OP::type_t>
{
	using type_t = typename OP::type_t;

	static inline void impl(type_t* outSubgroupData, const type_t* subgroupData, const uint32_t clampedSubgroupSize)
	{
		outSubgroupData[0u] = subgroupData[0u];
		for (auto i=1u; i<clampedSubgroupSize; i++)
			outSubgroupData[i] = OP()(outSubgroupData[i-1u],subgroupData[i]);
	}
	static inline constexpr const char* name = "subgroup inclusive scan";
};

#if 0
//workgroup methods
template<class OP>
struct emulatedWorkgroupReduction
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		type_t red = workgroupData[0];
		for (auto i=1u; i<workgroupSize; i++)
			red = OP()(red,workgroupData[i]);
		std::fill(outputData,outputData+workgroupSize,red);
	}
	static inline constexpr const char* name = "workgroup reduction";
};
template<class OP>
struct emulatedWorkgroupScanExclusive
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		outputData[0u] = OP::identity;
		for (auto i=1u; i<workgroupSize; i++)
			outputData[i] = OP()(outputData[i-1u],workgroupData[i-1u]);
	}
	static inline constexpr const char* name = "workgroup exclusive scan";
};
template<class OP>
struct emulatedWorkgroupScanInclusive
{
	using type_t = typename OP::type_t;

	inline void operator()(type_t* outputData, const type_t* workgroupData, uint32_t workgroupSize, uint32_t subgroupSize)
	{
		outputData[0u] = workgroupData[0u];
		for (auto i=1u; i<workgroupSize; i++)
			outputData[i] = OP()(outputData[i-1u],workgroupData[i]);
	}
	static inline constexpr const char* name = "workgroup inclusive scan";
};
#endif

//returns true if result matches
template<template<class> class Arithmetic, template<class> class OP>
bool validateResults(ILogicalDevice* device, IUtilities* utilities, IGPUQueue* transferDownQueue, const uint32_t* inputData, const uint32_t workgroupSize, const uint32_t workgroupCount, video::IGPUBuffer* bufferToRead, asset::ICPUBuffer* resultsBuffer, system::ILogger* logger)
{
	bool success = true;

	SBufferRange<IGPUBuffer> bufferRange = {0u, sizeof(Output<>::output), core::smart_refctd_ptr<IGPUBuffer>(bufferToRead)};
	utilities->downloadBufferRangeViaStagingBufferAutoSubmit(bufferRange, resultsBuffer->getPointer(), transferDownQueue);

	auto dataFromBuffer = reinterpret_cast<Output<>*>(resultsBuffer->getPointer());
	const uint32_t subgroupSize = dataFromBuffer->subgroupSize;

	// TODO: parallel for
	// now check if the data obtained has valid values
	uint32_t* tmp = new uint32_t[workgroupSize];
	uint32_t* ballotInput = new uint32_t[workgroupSize];
	for (uint32_t workgroupID=0u; success&&workgroupID<workgroupCount; workgroupID++)
	{
		const auto workgroupOffset = workgroupID*workgroupSize;
		if constexpr (std::is_same_v<OP<uint32_t>,ballot<uint32_t>>)
		{
			for (auto i=0u; i<workgroupSize; i++)
				ballotInput[i] = inputData[i+workgroupOffset]&0x1u;
			Arithmetic<OP<uint32_t>>()(tmp,ballotInput,workgroupSize,subgroupSize);
		}
		else
			Arithmetic<OP<uint32_t>>()(tmp,inputData+workgroupOffset,workgroupSize,subgroupSize);
		for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
		if (tmp[localInvocationIndex]!=dataFromBuffer->output[workgroupOffset+localInvocationIndex])
		{
			logger->log(
				"Failed test #%d  (%s)  (%s) Expected %u got %u for workgroup %d and localinvoc %d",system::ILogger::ELL_ERROR,
				workgroupSize,Arithmetic<OP<uint32_t>>::name,OP<uint32_t>::name,
				tmp[localInvocationIndex], dataFromBuffer->output[workgroupOffset + localInvocationIndex], workgroupOffset, localInvocationIndex
			);
			success = false;
			break;
		}
	}
	delete[] ballotInput;
	delete[] tmp;

	return success;
}

constexpr const auto outputBufferCount = 8u;

template<template<class> class Arithmetic>
bool runTest(
	ILogicalDevice* device, IUtilities* utilities, IGPUQueue* transferDownQueue, IGPUQueue* queue,
	IGPUFence* reusableFence, IGPUCommandBuffer* cmdbuf, core::smart_refctd_ptr<IGPUComputePipeline>&& pipeline, const IGPUDescriptorSet* ds,
	const uint32_t* inputData, const uint32_t workgroupSize, core::smart_refctd_ptr<IGPUBuffer>* const buffers, system::ILogger* logger, bool is_workgroup_test = false)
{
	// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
	constexpr auto kBufferSize = sizeof(Output<>);

	cmdbuf->begin(IGPUCommandBuffer::EU_NONE);
	cmdbuf->bindComputePipeline(pipeline.get());
	cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&ds);
	const uint32_t workgroupCount = Output<>::ScanDwordCount/workgroupSize;
	cmdbuf->dispatch(workgroupCount,1,1);
	IGPUCommandBuffer::SBufferMemoryBarrier memoryBarrier[outputBufferCount];
	for (auto i=0u; i<outputBufferCount; i++)
	{
		memoryBarrier[i].barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
		memoryBarrier[i].barrier.dstAccessMask = static_cast<asset::E_ACCESS_FLAGS>(EAF_SHADER_WRITE_BIT|EAF_HOST_READ_BIT);
		memoryBarrier[i].srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
		memoryBarrier[i].dstQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
		memoryBarrier[i].buffer = buffers[i];
		memoryBarrier[i].offset = 0u;
		memoryBarrier[i].size = kBufferSize;
	}
	cmdbuf->pipelineBarrier(
		asset::EPSF_COMPUTE_SHADER_BIT,static_cast<asset::E_PIPELINE_STAGE_FLAGS>(asset::EPSF_COMPUTE_SHADER_BIT|asset::EPSF_HOST_BIT),asset::EDF_NONE,
		0u,nullptr,outputBufferCount,memoryBarrier,0u,nullptr
	);
	cmdbuf->end();

	IGPUQueue::SSubmitInfo submit = {};
	submit.commandBufferCount = 1u;
	submit.commandBuffers = &cmdbuf;
	queue->submit(1u,&submit,reusableFence);
	device->blockForFences(1u,&reusableFence);
	device->resetFences(1u,&reusableFence);
	
	auto resultsBuffer = core::make_smart_refctd_ptr<ICPUBuffer>(kBufferSize);
	//check results 
	bool passed = validateResults<Arithmetic,bit_and>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[0].get(), resultsBuffer.get(),logger);
	passed = validateResults<Arithmetic,bit_xor>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[1].get(), resultsBuffer.get(),logger)&&passed;
	passed = validateResults<Arithmetic,bit_or>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[2].get(), resultsBuffer.get(),logger)&&passed;
	passed = validateResults<Arithmetic,plus>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[3].get(), resultsBuffer.get(),logger)&&passed;
	passed = validateResults<Arithmetic,multiplies>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[4].get(), resultsBuffer.get(),logger)&&passed;
	passed = validateResults<Arithmetic,minimum>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[5].get(), resultsBuffer.get(),logger)&&passed;
	passed = validateResults<Arithmetic,maximum>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[6].get(), resultsBuffer.get(),logger)&&passed;
	if(is_workgroup_test)
	{
		passed = validateResults<Arithmetic,ballot>(device, utilities, transferDownQueue, inputData, workgroupSize, workgroupCount, buffers[7].get(), resultsBuffer.get(), logger)&&passed;
	}

	return passed;
}

class ArythmeticUnitTestApp : public NonGraphicalApplicationBase
{

public:
	void setSystem(nbl::core::smart_refctd_ptr<nbl::system::ISystem>&& s) override
	{
		system = std::move(s);
	}

	NON_GRAPHICAL_APP_CONSTRUCTOR(ArythmeticUnitTestApp)
	void onAppInitialized_impl() override
	{
#pragma region Init
		CommonAPI::InitParams initParams;
		initParams.apiType = video::EAT_VULKAN;
		initParams.appName = { "Subgroup Arithmetic Test" };
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));

		apiConnection = std::move(initOutput.apiConnection);
		gpuPhysicalDevice = std::move(initOutput.physicalDevice);
		logicalDevice = std::move(initOutput.logicalDevice);
		queues = std::move(initOutput.queues);
		commandPools = std::move(initOutput.commandPools);
		assetManager = std::move(initOutput.assetManager);
		logger = std::move(initOutput.logger);
		inputSystem = std::move(initOutput.inputSystem);
		system = std::move(initOutput.system);
		cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
		utilities = std::move(initOutput.utilities);
		
		auto transferDownQueue = queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN];

		nbl::video::IGPUObjectFromAssetConverter cpu2gpu;
#pragma endregion Init

		inputData = new uint32_t[Output<>::ScanDwordCount];
		{
			std::mt19937 randGenerator(std::time(0));
			for (uint32_t i=0u; i<Output<>::ScanDwordCount; i++)
				inputData[i] = randGenerator();
		}

		IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
		inputDataBufferCreationParams.size = sizeof(Output<>::output);
		inputDataBufferCreationParams.usage = core::bitflag<IGPUBuffer::E_USAGE_FLAGS>(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
		auto gpuinputDataBuffer = utilities->createFilledDeviceLocalBufferOnDedMem(queues[decltype(initOutput)::EQT_TRANSFER_UP], std::move(inputDataBufferCreationParams), inputData);

		//create 8 buffers.
		constexpr const auto totalBufferCount = outputBufferCount+1u; // output buffers for all ops +1 for the input buffer

		constexpr auto kBufferSize = sizeof(Output<>);
		core::smart_refctd_ptr<IGPUBuffer> buffers[outputBufferCount];
		for (auto i = 0; i<outputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params;
			params.size = kBufferSize;
			params.queueFamilyIndexCount = 0;
			params.queueFamilyIndices = nullptr;
			params.usage = core::bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT)|IGPUBuffer::EUF_TRANSFER_SRC_BIT;
			
			buffers[i] = logicalDevice->createBuffer(std::move(params));
			auto mreq = buffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= logicalDevice->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();

			assert(mreq.memoryTypeBits);
			auto bufferMem = logicalDevice->allocate(mreq, buffers[i].get());
			assert(bufferMem.isValid());
		}

		IGPUDescriptorSetLayout::SBinding binding[totalBufferCount];
		for (uint32_t i = 0u; i < totalBufferCount; i++)
			binding[i] = { i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr };
		auto gpuDSLayout = logicalDevice->createDescriptorSetLayout(binding, binding + totalBufferCount);

		constexpr uint32_t pushconstantSize = 8u * totalBufferCount;
		SPushConstantRange pcRange[1] = { IShader::ESS_COMPUTE,0u,pushconstantSize };
		auto pipelineLayout = logicalDevice->createPipelineLayout(pcRange, pcRange + 1u, core::smart_refctd_ptr(gpuDSLayout));

		auto descPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &gpuDSLayout.get(), &gpuDSLayout.get() + 1u);
		auto descriptorSet = descPool->createDescriptorSet(core::smart_refctd_ptr(gpuDSLayout));
		{
			IGPUDescriptorSet::SDescriptorInfo infos[totalBufferCount];
			infos[0].desc = gpuinputDataBuffer;
			infos[0].info.buffer = { 0u,kBufferSize };

			for (uint32_t i = 1u; i <= outputBufferCount; i++)
			{
				infos[i].desc = buffers[i - 1];
				infos[i].info.buffer = { 0u,kBufferSize };

			}
			IGPUDescriptorSet::SWriteDescriptorSet writes[totalBufferCount];
			for (uint32_t i = 0u; i < totalBufferCount; i++)
				writes[i] = { descriptorSet.get(),i,0u,1u,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,infos + i };
			logicalDevice->updateDescriptorSets(totalBufferCount, writes, 0u, nullptr);
		}

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams;
			lparams.workingDirectory = std::filesystem::current_path();
			auto bundle = assetManager->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SPECIALIZED_SHADER)
			{
				logger->log("Shader %s not found!", system::ILogger::ELL_ERROR, filePath);
				exit(-1);
			}
			auto firstAssetInBundle = bundle.getContents()[0];
			return core::smart_refctd_ptr<ICPUShader>(core::smart_refctd_ptr_static_cast<ICPUSpecializedShader>(firstAssetInBundle)->getUnspecialized());
		};
		auto subgroupShader = getShaderSource("../hlsl/testSubgroup.comp.hlsl");
//		auto workgroupTestSource = getShaderSource("../hlsl/testWorkgroup.comp.hlsl");

		// create pipeline (specialized every test) [TODO: turn into a future/async]
		auto createPipeline = [&](const core::smart_refctd_ptr<const ICPUShader>& source, const char* opName, const uint32_t workgroupSize) -> core::smart_refctd_ptr<IGPUComputePipeline>
		{
			auto overridenUnspecialized = CHLSLCompiler::createOverridenCopy(source.get(), "#define OPERATION %s\n#define _NBL_WORKGROUP_SIZE_ %d\n", opName, workgroupSize);
			auto shader = logicalDevice->createShader(std::move(overridenUnspecialized));
			auto specialized = logicalDevice->createSpecializedShader(shader.get(),ISpecializedShader::SInfo(nullptr,nullptr,"main"));
			return logicalDevice->createComputePipeline(nullptr,core::smart_refctd_ptr(pipelineLayout),std::move(specialized));
		};


		auto logTestOutcome = [this](bool passed, uint32_t workgroupSize)
		{
			if (passed)
				logger->log("Passed test #%u", system::ILogger::ELL_INFO, workgroupSize);
			else
			{
				totalFailCount++;
				logger->log("Failed test #%u", system::ILogger::ELL_ERROR, workgroupSize);
			}
		};

		// get stuff
		auto computeQueue = initOutput.queues[CommonAPI::InitOutput::EQT_COMPUTE];
		auto fence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
		auto cmdPools = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];
		core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		logicalDevice->createCommandBuffers(cmdPools[0].get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

		const auto MaxWorkgroupSize = gpuPhysicalDevice->getLimits().maxComputeWorkGroupInvocations;
		// TODO: redo the test with all subgroup sizes
		const auto MinSubgroupSize = gpuPhysicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = gpuPhysicalDevice->getLimits().maxSubgroupSize;
		for (auto subgroupSize=/*see TODO*/MaxSubgroupSize; subgroupSize<=MaxSubgroupSize; subgroupSize*=2u)
		{
			for (uint32_t workgroupSize=subgroupSize; workgroupSize<=MaxWorkgroupSize; workgroupSize+=subgroupSize)
			{
				// make sure renderdoc captures everything for debugging
				computeQueue->startCapture();
				logger->log("Testing Workgroup Size %u", system::ILogger::ELL_INFO, workgroupSize);
				logger->log("Testing Item Count %u", system::ILogger::ELL_INFO, workgroupSize);

				bool passed = true;

				// TODO async the testing
				const video::IGPUDescriptorSet* ds = descriptorSet.get();
				passed = runTest<emulatedSubgroupReduction>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), createPipeline(subgroupShader,"reduction",workgroupSize), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get()) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedSubgroupScanExclusive>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), createPipeline(subgroupShader,"inclusive_scan",workgroupSize), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get()) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedSubgroupScanInclusive>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), createPipeline(subgroupShader,"exclusive_scan",workgroupSize), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get()) && passed;
				logTestOutcome(passed, workgroupSize);
#if 0
				passed = runTest<emulatedWorkgroupReduction>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), pipelines[3u].get(), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get(), true) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedWorkgroupScanInclusive>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), pipelines[4u].get(), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get(), true) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedWorkgroupScanExclusive>(logicalDevice.get(), utilities.get(), transferDownQueue, computeQueue, fence.get(), cmdbuf.get(), pipelines[5u].get(), descriptorSet.get(), inputData, workgroupSize, buffers, logger.get(), true) && passed;
				logTestOutcome(passed, workgroupSize);
#endif
				computeQueue->endCapture();
			}
		}
	}

	void onAppTerminated_impl() override
	{
		logger->log("==========Result==========", system::ILogger::ELL_INFO);
		logger->log("Fail Count: %u", system::ILogger::ELL_INFO, totalFailCount);
		delete[] inputData;
	}

	void workLoopBody() override
	{
		//! the unit test is carried out on init
	}

	bool keepRunning() override
	{
		return false;
	}

	private:

		nbl::core::smart_refctd_ptr<nbl::video::IAPIConnection> apiConnection;
		nbl::core::smart_refctd_ptr<nbl::video::IUtilities> utilities;
		nbl::core::smart_refctd_ptr<nbl::video::ILogicalDevice> logicalDevice;
		nbl::video::IPhysicalDevice* gpuPhysicalDevice;
		std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> queues;
		nbl::core::smart_refctd_ptr<nbl::video::IGPURenderpass> renderpass;
		std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
		nbl::core::smart_refctd_ptr<nbl::system::ISystem> system;
		nbl::core::smart_refctd_ptr<nbl::asset::IAssetManager> assetManager;
		nbl::video::IGPUObjectFromAssetConverter::SParams cpu2gpuParams;
		nbl::core::smart_refctd_ptr<nbl::system::ILogger> logger;
		nbl::core::smart_refctd_ptr<CommonAPI::InputSystem> inputSystem;

		uint32_t* inputData = nullptr;
		uint32_t totalFailCount = 0;
};

NBL_COMMON_API_MAIN(ArythmeticUnitTestApp)
