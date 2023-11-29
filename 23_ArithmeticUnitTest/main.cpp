#include "../common/BasicMultiQueueApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "hlsl/common.hlsl"

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

class ArithmeticUnitTestApp final : public examples::BasicMultiQueueApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = examples::BasicMultiQueueApplication;
	using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

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
				getTransferUpQueue(),
				std::move(inputDataBufferCreationParams),
				inputData
			);
		}

		// create 8 buffers for 8 operations
		for (auto i = 0u; i < OutputBufferCount; i++)
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
				dsLayout = m_device->createDescriptorSetLayout(binding, binding + 2);
			}

			// set and transient pool
			auto descPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &dsLayout.get(), &dsLayout.get() + 1u);
			descriptorSet = descPool->createDescriptorSet(smart_refctd_ptr(dsLayout));
			{
				IGPUDescriptorSet::SDescriptorInfo infos[1 + OutputBufferCount];
				infos[0].desc = gpuinputDataBuffer;
				infos[0].info.buffer = { 0u,gpuinputDataBuffer->getSize() };
				for (uint32_t i = 1u; i <= OutputBufferCount; i++)
				{
					auto buff = outputBuffers[i - 1];
					infos[i].info.buffer = { 0u,buff->getSize() };
					infos[i].desc = std::move(buff); // save an atomic in the refcount

				}

				IGPUDescriptorSet::SWriteDescriptorSet writes[2];
				for (uint32_t i = 0u; i < 2; i++)
					writes[i] = { descriptorSet.get(),i,0u,1u,IDescriptor::E_TYPE::ET_STORAGE_BUFFER,infos + i };
				writes[1].count = OutputBufferCount;

				m_device->updateDescriptorSets(2, writes, 0u, nullptr);
			}

			pipelineLayout = m_device->createPipelineLayout(nullptr, nullptr, std::move(dsLayout));
		}

		// load shader source from file
		auto getShaderSource = [&](const char* filePath) -> auto
		{
			IAssetLoader::SAssetLoadParams lparams;
			lparams.workingDirectory = std::filesystem::current_path();
			auto bundle = m_assetMgr->getAsset(filePath, lparams);
			if (bundle.getContents().empty() || bundle.getAssetType() != IAsset::ET_SPECIALIZED_SHADER)
			{
				m_logger->log("Shader %s not found!", ILogger::ELL_ERROR, filePath);
				exit(-1);
			}
			auto firstAssetInBundle = bundle.getContents()[0];
			return smart_refctd_ptr<ICPUShader>(smart_refctd_ptr_static_cast<ICPUSpecializedShader>(firstAssetInBundle)->getUnspecialized());
		};
		auto subgroupTestSource = getShaderSource("../hlsl/testSubgroup.comp.hlsl");
		auto workgroupTestSource = getShaderSource("../hlsl/testWorkgroup.comp.hlsl");

		// now create or retrieve final resources to run our tests
		fence = m_device->createFence(IGPUFence::ECF_UNSIGNALED);
		resultsBuffer = make_smart_refctd_ptr<ICPUBuffer>(outputBuffers[0]->getSize());
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			if (!m_device->createCommandBuffers(cmdpool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}

		const auto MaxWorkgroupSize = m_physicalDevice->getLimits().maxComputeWorkGroupInvocations;
		// TODO: redo the test with all subgroup sizes
		const auto MinSubgroupSize = m_physicalDevice->getLimits().minSubgroupSize;
		const auto MaxSubgroupSize = m_physicalDevice->getLimits().maxSubgroupSize;
		for (auto subgroupSize =/*see TODO*/MaxSubgroupSize; subgroupSize <= MaxSubgroupSize; subgroupSize *= 2u)
		{
			for (uint32_t workgroupSize = subgroupSize; workgroupSize <= MaxWorkgroupSize; workgroupSize += subgroupSize)
			{
				// make sure renderdoc captures everything for debugging
				computeQueue->startCapture();
				m_logger->log("Testing Workgroup Size %u", ILogger::ELL_INFO, workgroupSize);

				bool passed = true;
				// TODO async the testing
				passed = runTest<emulatedReduction, false>(subgroupTestSource, elementCount, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanInclusive, false>(subgroupTestSource, elementCount, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				passed = runTest<emulatedScanExclusive, false>(subgroupTestSource, elementCount, workgroupSize) && passed;
				logTestOutcome(passed, workgroupSize);
				for (uint32_t itemsPerWG = workgroupSize; itemsPerWG > workgroupSize - subgroupSize; itemsPerWG--)
				{
					m_logger->log("Testing Item Count %u", ILogger::ELL_INFO, itemsPerWG);
					passed = runTest<emulatedReduction, true>(workgroupTestSource, elementCount, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanInclusive, true>(workgroupTestSource, elementCount, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
					passed = runTest<emulatedScanExclusive, true>(workgroupTestSource, elementCount, workgroupSize, itemsPerWG) && passed;
					logTestOutcome(passed, itemsPerWG);
				}
				computeQueue->endCapture();
			}
		}

		return true;
	}

	virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	{
		video::SPhysicalDeviceFeatures retval = {};

		retval.bufferDeviceAddress = true;
		retval.subgroupBroadcastDynamicId = true;
		retval.shaderSubgroupExtendedTypes = true;
		// TODO: actually need to implement this and set it on the pipelines
		retval.computeFullSubgroups = true;
		retval.subgroupSizeControl = true;

		return retval;
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
	smart_refctd_ptr<IGPUComputePipeline> createPipeline(smart_refctd_ptr<ICPUShader>&& overridenUnspecialized)
	{
		auto shader = m_device->createShader(std::move(overridenUnspecialized));
		auto specialized = m_device->createSpecializedShader(shader.get(), ISpecializedShader::SInfo(nullptr, nullptr, "main"));
		return m_device->createComputePipeline(nullptr, smart_refctd_ptr(pipelineLayout), std::move(specialized));
	}

	/*template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u)
	{
		return true;
	}*/

	template<template<class> class Arithmetic, bool WorkgroupTest>
	bool runTest(const smart_refctd_ptr<const ICPUShader>& source, const uint32_t elementCount, const uint32_t workgroupSize, uint32_t itemsPerWG = ~0u)
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
		auto pipeline = createPipeline(std::move(overridenUnspecialized));

		// TODO: overlap dispatches with memory readbacks (requires multiple copies of `buffers`)
		const uint32_t workgroupCount = elementCount / itemsPerWG;
		cmdbuf->begin(IGPUCommandBuffer::EU_NONE);
		cmdbuf->bindComputePipeline(pipeline.get());
		cmdbuf->bindDescriptorSets(EPBP_COMPUTE, pipeline->getLayout(), 0u, 1u, &descriptorSet.get());
		cmdbuf->dispatch(workgroupCount, 1, 1);
		{
			IGPUCommandBuffer::SBufferMemoryBarrier memoryBarrier[OutputBufferCount];
			// in theory we don't need the HOST BITS cause we block on a fence but might as well add them
			for (auto i = 0u; i < OutputBufferCount; i++)
			{
				memoryBarrier[i].barrier.srcAccessMask = EAF_SHADER_WRITE_BIT;
				memoryBarrier[i].barrier.dstAccessMask = EAF_SHADER_WRITE_BIT | EAF_HOST_READ_BIT;
				memoryBarrier[i].srcQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
				memoryBarrier[i].dstQueueFamilyIndex = cmdbuf->getQueueFamilyIndex();
				memoryBarrier[i].buffer = outputBuffers[i];
				memoryBarrier[i].offset = 0u;
				memoryBarrier[i].size = outputBuffers[i]->getSize();
			}
			cmdbuf->pipelineBarrier(
				EPSF_COMPUTE_SHADER_BIT, EPSF_COMPUTE_SHADER_BIT | EPSF_HOST_BIT, EDF_NONE,
				0u, nullptr, OutputBufferCount, memoryBarrier, 0u, nullptr
			);
		}
		cmdbuf->end();

		IGPUQueue::SSubmitInfo submit = {};
		submit.commandBufferCount = 1u;
		submit.commandBuffers = &cmdbuf.get();
		computeQueue->submit(1u, &submit, fence.get());
		m_device->blockForFences(1u, &fence.get());
		m_device->resetFences(1u, &fence.get());

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
		SBufferRange<IGPUBuffer> bufferRange = { 0u, resultsBuffer->getSize(), outputBuffers[Binop::BindingIndex] };
		m_utils->downloadBufferRangeViaStagingBufferAutoSubmit(bufferRange, resultsBuffer->getPointer(), transferDownQueue);

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

	IGPUQueue* transferDownQueue;
	IGPUQueue* computeQueue;

	uint32_t* inputData = nullptr;
	constexpr static inline uint32_t OutputBufferCount = 8u;
	smart_refctd_ptr<IGPUBuffer> outputBuffers[OutputBufferCount];
	smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;

	smart_refctd_ptr<IGPUFence> fence;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	uint32_t totalFailCount = 0;
};

NBL_MAIN_FUNC(ArithmeticUnitTestApp)