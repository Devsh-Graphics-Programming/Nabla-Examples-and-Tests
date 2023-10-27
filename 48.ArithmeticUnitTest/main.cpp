#include "../common/CommonAPI.h"
#include "hlsl/common.hlsl"


using namespace nbl::core;
using namespace nbl::asset;
using namespace nbl::system;
using namespace nbl::video;

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
			CRTP::impl(outSubgroupData,subgroupData,min<uint32_t>(subgroupSize,workgroupSize-pseudoSubgroupID));
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


class ArithmeticUnitTestApp : public NonGraphicalApplicationBase
{

public:
	void setSystem(smart_refctd_ptr<ISystem>&& s) override
	{
		system = std::move(s);
	}

	NON_GRAPHICAL_APP_CONSTRUCTOR(ArithmeticUnitTestApp)
	void onAppInitialized_impl() override
	{
		// Initialize
		CommonAPI::InitParams initParams;
		initParams.apiType = EAT_VULKAN;
		initParams.appName = { "Subgroup Arithmetic Test" };
		initParams.physicalDeviceFilter.requiredFeatures.subgroupBroadcastDynamicId = true;
		initParams.physicalDeviceFilter.requiredFeatures.shaderSubgroupExtendedTypes = true;
		// TODO: actually need to implement this and set it on the pipelines
		initParams.physicalDeviceFilter.requiredFeatures.computeFullSubgroups = true;
		initParams.physicalDeviceFilter.requiredFeatures.subgroupSizeControl = true;
		auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));
		auto physicalDevice = initOutput.physicalDevice;
		{
			system = std::move(initOutput.system);
			logger = std::move(initOutput.logger);

			logicalDevice = std::move(initOutput.logicalDevice);
			transferDownQueue = initOutput.queues[CommonAPI::InitOutput::EQT_TRANSFER_DOWN];
			utilities = std::move(initOutput.utilities);
		}

		// TODO: get the element count from argv
		const uint32_t elementCount = Output<>::ScanElementCount;
		// populate our random data buffer on the CPU and create a GPU copy
		inputData = new uint32_t[elementCount];
		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			std::mt19937 randGenerator(std::time(0));
			for (uint32_t i=0u; i<elementCount; i++)
				inputData[i] = randGenerator();

			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(Output<>::data[0])*elementCount;
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT|IGPUBuffer::EUF_TRANSFER_DST_BIT;
			gpuinputDataBuffer = utilities->createFilledDeviceLocalBufferOnDedMem(
				initOutput.queues[decltype(initOutput)::EQT_TRANSFER_UP],
				std::move(inputDataBufferCreationParams),inputData
			);
		}

		// create 8 buffers for 8 operations
		for (auto i=0u; i<OutputBufferCount; i++)
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = sizeof(uint32_t)+gpuinputDataBuffer->getSize();
			params.usage = bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT)|IGPUBuffer::EUF_TRANSFER_SRC_BIT;
			
			outputBuffers[i] = logicalDevice->createBuffer(std::move(params));
			auto mreq = outputBuffers[i]->getMemoryReqs();
			mreq.memoryTypeBits &= physicalDevice->getDeviceLocalMemoryTypeBits();
			assert(mreq.memoryTypeBits);

			auto bufferMem = logicalDevice->allocate(mreq,outputBuffers[i].get());
			assert(bufferMem.isValid());
		}

		// create Descriptor Set and Pipeline Layout
		smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
		{
			// create Descriptor Set Layout
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout;
			{
				IGPUDescriptorSetLayout::SBinding binding[2];
				for (uint32_t i=0u; i<2; i++)
					binding[i] = { i,IDescriptor::E_TYPE::ET_STORAGE_BUFFER, IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, IShader::ESS_COMPUTE, 1u, nullptr };
				binding[1].count = OutputBufferCount;
				dsLayout = logicalDevice->createDescriptorSetLayout(binding,binding+2);
			}

			// set and transient pool
			auto descPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE,&dsLayout.get(),&dsLayout.get()+1u);
			descriptorSet = descPool->createDescriptorSet(smart_refctd_ptr(dsLayout));
			{
				IGPUDescriptorSet::SDescriptorInfo infos[1+OutputBufferCount];
				infos[0].desc = gpuinputDataBuffer;
				infos[0].info.buffer = { 0u,gpuinputDataBuffer->getSize() };
				for (uint32_t i=1u; i<=OutputBufferCount; i++)
				{
					auto buff = outputBuffers[i-1];
					infos[i].info.buffer = { 0u,buff->getSize() };
					infos[i].desc = std::move(buff); // save an atomic in the refcount

				}

				IGPUDescriptorSet::SWriteDescriptorSet writes[2];
				for (uint32_t i=0u; i<2; i++)
					writes[i] = { descriptorSet.get(),i,0u,1u,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,infos+i };
				writes[1].count = OutputBufferCount;
				
				logicalDevice->updateDescriptorSets(2,writes,0u,nullptr);
			}

			pipelineLayout = logicalDevice->createPipelineLayout(nullptr,nullptr,std::move(dsLayout));
		}

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams;
			lparams.workingDirectory = std::filesystem::current_path();
			auto bundle = initOutput.assetManager->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType()!=IAsset::ET_SPECIALIZED_SHADER)
			{
				logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
				exit(-1);
			}
			auto firstAssetInBundle = bundle.getContents()[0];
			return smart_refctd_ptr<ICPUShader>(smart_refctd_ptr_static_cast<ICPUSpecializedShader>(firstAssetInBundle)->getUnspecialized());
		};
		auto subgroupShader = getShaderSource("../hlsl/testSubgroup.comp.hlsl");
//		auto workgroupTestSource = getShaderSource("../hlsl/testWorkgroup.comp.hlsl");

		// create pipeline (specialized every test) [TODO: turn into a future/async]
		auto createPipeline = [&](const smart_refctd_ptr<const ICPUShader>& source, const char* opName, const uint32_t workgroupSize, const uint32_t itemsPerWG=0) -> smart_refctd_ptr<IGPUComputePipeline>
		{
			auto overridenUnspecialized = CHLSLCompiler::createOverridenCopy(source.get(), "#define OPERATION %s\n#define WORKGROUP_SIZE %d\n#define ITEMS_PER_WG %d",opName,workgroupSize,itemsPerWG);
			auto shader = logicalDevice->createShader(std::move(overridenUnspecialized));
			auto specialized = logicalDevice->createSpecializedShader(shader.get(),ISpecializedShader::SInfo(nullptr,nullptr,"main"));
			return logicalDevice->createComputePipeline(nullptr,smart_refctd_ptr(pipelineLayout),std::move(specialized));
		};

		// now create or retrieve final resources to run our tests
		computeQueue = initOutput.queues[CommonAPI::InitOutput::EQT_COMPUTE];
		fence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
		logicalDevice->createCommandBuffers(initOutput.commandPools[CommonAPI::InitOutput::EQT_COMPUTE].begin()->get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
		resultsBuffer = make_smart_refctd_ptr<ICPUBuffer>(outputBuffers[0]->getSize());

		const auto MaxWorkgroupSize = physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		// TODO: redo the test with all subgroup sizes
		const auto MinSubgroupSize = physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = physicalDevice->getLimits().maxSubgroupSize;
		for (auto subgroupSize=/*see TODO*/MaxSubgroupSize; subgroupSize<=MaxSubgroupSize; subgroupSize*=2u)
		{
			for (uint32_t workgroupSize=subgroupSize; workgroupSize<=MaxWorkgroupSize; workgroupSize+=subgroupSize)
			{
				// make sure renderdoc captures everything for debugging
				computeQueue->startCapture();
				logger->log("Testing Workgroup Size %u", ILogger::ELL_INFO, workgroupSize);
				logger->log("Testing Item Count %u", ILogger::ELL_INFO, workgroupSize);

				bool passed = true;

				// TODO async the testing
				const IGPUDescriptorSet* ds = descriptorSet.get();
				passed = runTest<emulatedSubgroupReduction>(elementCount,createPipeline(subgroupShader,"reduction",workgroupSize),workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedSubgroupScanInclusive>(elementCount,createPipeline(subgroupShader,"inclusive_scan",workgroupSize),workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedSubgroupScanExclusive>(elementCount,createPipeline(subgroupShader,"exclusive_scan",workgroupSize),workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				for (uint32_t itemsPerWG=workgroupSize; itemsPerWG>workgroupSize-subgroupSize; itemsPerWG--)
				{
					//passed = runTest<emulatedWorkgroupReduction>(pipelines[3u].get(), workgroupSize, true) && passed;
					logTestOutcome(passed, itemsPerWG);
					//passed = runTest<emulatedWorkgroupScanInclusive>(pipelines[4u].get(), workgroupSize, true) && passed;
					logTestOutcome(passed, itemsPerWG);
					//passed = runTest<emulatedWorkgroupScanExclusive>(pipelines[5u].get(), workgroupSize, true) && passed;
					logTestOutcome(passed, itemsPerWG);
				}
				computeQueue->endCapture();
			}
		}
	}

		void onAppTerminated_impl() override
		{
			logger->log("==========Result==========", ILogger::ELL_INFO);
			logger->log("Fail Count: %u", ILogger::ELL_INFO, totalFailCount);
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
		void logTestOutcome(bool passed, uint32_t workgroupSize)
		{
			if (passed)
				logger->log("Passed test #%u", ILogger::ELL_INFO, workgroupSize);
			else
			{
				totalFailCount++;
				logger->log("Failed test #%u", ILogger::ELL_ERROR, workgroupSize);
			}
		}


		template<template<class> class Arithmetic>
		bool runTest(const uint32_t elementCount, smart_refctd_ptr<IGPUComputePipeline>&& pipeline, const uint32_t workgroupSize, bool is_workgroup_test = false)
		{
			// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
			const uint32_t workgroupCount = elementCount/workgroupSize;

			cmdbuf->begin(IGPUCommandBuffer::EU_NONE);
			cmdbuf->bindComputePipeline(pipeline.get());
			cmdbuf->bindDescriptorSets(EPBP_COMPUTE,pipeline->getLayout(),0u,1u,&descriptorSet.get());
			cmdbuf->dispatch(workgroupCount,1,1);
			{
				IGPUCommandBuffer::SBufferMemoryBarrier memoryBarrier[OutputBufferCount];
				// in theory we don't need the HOST BITS cause we block on a fence but might as well add them
				for (auto i=0u; i<OutputBufferCount; i++)
				{
					memoryBarrier[i].barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
					memoryBarrier[i].barrier.dstAccessMask = EAF_SHADER_WRITE_BIT|EAF_HOST_READ_BIT;
					memoryBarrier[i].srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
					memoryBarrier[i].dstQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
					memoryBarrier[i].buffer = outputBuffers[i];
					memoryBarrier[i].offset = 0u;
					memoryBarrier[i].size = outputBuffers[i]->getSize();
				}
				cmdbuf->pipelineBarrier(
					EPSF_COMPUTE_SHADER_BIT,EPSF_COMPUTE_SHADER_BIT|EPSF_HOST_BIT,EDF_NONE,
					0u,nullptr,OutputBufferCount,memoryBarrier,0u,nullptr
				);
			}
			cmdbuf->end();

			IGPUQueue::SSubmitInfo submit = {};
			submit.commandBufferCount = 1u;
			submit.commandBuffers = &cmdbuf.get();
			computeQueue->submit(1u,&submit,fence.get());
			logicalDevice->blockForFences(1u,&fence.get());
			logicalDevice->resetFences(1u,&fence.get());
	
			//check results 
			bool passed = validateResults<Arithmetic,bit_and<uint32_t>>(workgroupSize, workgroupCount);
			passed = validateResults<Arithmetic,bit_xor<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			passed = validateResults<Arithmetic,bit_or<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			passed = validateResults<Arithmetic,plus<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			passed = validateResults<Arithmetic,multiplies<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			passed = validateResults<Arithmetic,minimum<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			passed = validateResults<Arithmetic,maximum<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			if(is_workgroup_test)
			{
				passed = validateResults<Arithmetic,ballot<uint32_t>>(workgroupSize, workgroupCount)&&passed;
			}

			return passed;
		}

		//returns true if result matches
		template<template<class> class Arithmetic, class Binop>
		bool validateResults(const uint32_t workgroupSize, const uint32_t workgroupCount)
		{
			bool success = true;

			SBufferRange<IGPUBuffer> bufferRange = {0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex]};
			utilities->downloadBufferRangeViaStagingBufferAutoSubmit(bufferRange, resultsBuffer->getPointer(), transferDownQueue);

			using type_t = typename Binop::type_t;
			const auto dataFromBuffer = reinterpret_cast<const uint32_t*>(resultsBuffer->getPointer());
			const auto subgroupSize = dataFromBuffer[0];
			if (subgroupSize<nbl::hlsl::subgroup::MinSubgroupSize || subgroupSize>nbl::hlsl::subgroup::MaxSubgroupSize)
			{
				logger->log("Unexpected Subgroup Size #%u", ILogger::ELL_ERROR, workgroupSize);
				return false;
			}
			const auto testData = reinterpret_cast<const type_t*>(dataFromBuffer+1);

			// TODO: parallel for
			// now check if the data obtained has valid values
			type_t* tmp = new type_t[workgroupSize];
			type_t* ballotInput = new type_t[workgroupSize];
			for (uint32_t workgroupID=0u; success&&workgroupID<workgroupCount; workgroupID++)
			{
				const auto workgroupOffset = workgroupID*workgroupSize;

				if constexpr (std::is_same_v<ballot<type_t>,Binop>)
				{
					for (auto i=0u; i<workgroupSize; i++)
						ballotInput[i] = inputData[i+workgroupOffset]&0x1u;
					Arithmetic<Binop>()(tmp,ballotInput,workgroupSize,subgroupSize);
				}
				else
					Arithmetic<Binop>()(tmp,inputData+workgroupOffset,workgroupSize,subgroupSize);

				for (uint32_t localInvocationIndex=0u; localInvocationIndex<workgroupSize; localInvocationIndex++)
				{
					const auto globalInvocationIndex = workgroupOffset+localInvocationIndex;
					if (tmp[localInvocationIndex]!=testData[globalInvocationIndex])
					{
						logger->log(
							"Failed test #%d  (%s)  (%s) Expected %u got %u for workgroup %d and localinvoc %d",
							ILogger::ELL_ERROR,workgroupSize,Arithmetic<Binop>::name,Binop::name,
							tmp[localInvocationIndex], testData[globalInvocationIndex], workgroupOffset, localInvocationIndex
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

		smart_refctd_ptr<ISystem> system;
		smart_refctd_ptr<ILogger> logger;

		smart_refctd_ptr<ILogicalDevice> logicalDevice;
		IGPUQueue* transferDownQueue;
		IGPUQueue* computeQueue;
		smart_refctd_ptr<IUtilities> utilities;

		uint32_t* inputData = nullptr;
		constexpr static inline uint32_t OutputBufferCount = 8u;
		smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];
		smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;

		smart_refctd_ptr<IGPUFence> fence;
		smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
		smart_refctd_ptr<ICPUBuffer> resultsBuffer;

		uint32_t totalFailCount = 0;
};

NBL_COMMON_API_MAIN(ArithmeticUnitTestApp)