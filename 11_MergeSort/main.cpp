#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"

// A high level overview of the algorithm implementation (merge sort).
// This will be a X pass algorithm, where X is log2(number_of_elements).
// In each phase, you have to 'merge' 2 sorted list of elements into a single list.
// Each of these 'lists' will have thier number of elements roughly doubled in each phase (after the merge step of previous phase).
class MergeSortApp final : public nbl::application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
	MergeSortApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		// Initialize base classes.
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
		{
			return false;
		}
		if (!asset_base_t::onAppInitialized(std::move(system)))
		{
			return false;
		}

		// Get the maxinum bytes of shared memory we can use.
		// In case the number of elements shared memory array can hold exceeds the limit, then switch to fetching data from global memory directly (for as few memory elements as possible).
		const auto physicalDeviceLimits = m_device->getPhysicalDevice()->getLimits();
		const uint32_t maxNumberOfArrayElementsSharedMemoryCanHold = physicalDeviceLimits.maxComputeSharedMemorySize / sizeof(int);

		// Compute and input related constants.
		static constexpr uint32_t WorkgroupSize = 64;
		static constexpr uint32_t NumberOfElementsToSort = 1024 * 633u;
		static const uint32_t WorkgroupCount = (uint32_t)ceil(NumberOfElementsToSort / (float)WorkgroupSize);

		const size_t bufferSize = sizeof(int32_t) * NumberOfElementsToSort;

		std::vector<int32_t> inputBufferData(NumberOfElementsToSort);
		{
			// Setup the input data (random) that is to be sorted.
			const auto seed = 2532344;

			// Setup the random number generator.
			std::mt19937 randomNumberGenerator(seed);

			for (auto& elem : inputBufferData)
			{
				elem = randomNumberGenerator();
			}
		}

		// Create and setup the compute shader.
		{
			IAssetLoader::SAssetLoadParams assetLoadParams = {};
			assetLoadParams.logger = m_logger.get();
			assetLoadParams.workingDirectory = ""; // virutal root.

			auto shaderAsset = m_assetMgr->getAsset("app_resources/merge_sort.hlsl", assetLoadParams);
			const auto shaderAssetContents = shaderAsset.getContents();
			if (shaderAssetContents.empty())
			{
				return logFail("Failed to load shader asset with file name app_resources/merge_sort.hlsl");
			}

			auto cpuShaderAsset = IAsset::castDown<ICPUShader>(shaderAssetContents[0]);
			cpuShaderAsset->setShaderStage(IShader::ESS_COMPUTE);

			// Asset to ensure downcasting did not fail.
			assert(cpuShaderAsset);

			// Now, compile the shader.
			// During compilation, replace the value of WORKGROUP_SIZE in the shader source.

			const auto overridenSource = CHLSLCompiler::createOverridenCopy(
				cpuShaderAsset.get(), "#define WORKGROUP_SIZE %d\n#define MaxNumberOfArrayElementsSharedMemoryCanHold %d\n",
				WorkgroupSize, std::min(maxNumberOfArrayElementsSharedMemoryCanHold, NumberOfElementsToSort)
			);

			m_mergeSortShader = m_device->createShader(overridenSource.get());
			if (!m_mergeSortShader)
			{
				return logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");
			}
		}

		// Since BDA is used, the only binding the shader will have is a single push constant from which all buffers and data can be accessed.
		const nbl::asset::SPushConstantRange pushConstantRange = nbl::asset::SPushConstantRange{
			.stageFlags = nbl::asset::IShader::ESS_COMPUTE, .offset = 0, .size = sizeof(MergeSortPushData),
		};

		// Create the pipeline layout.
		m_pipelineLayout = m_device->createPipelineLayout({ &pushConstantRange, 1 });

		IGPUComputePipeline::SCreationParams params = {};
		params.layout = m_pipelineLayout.get();
		params.shader.entryPoint = "main";
		params.shader.shader = m_mergeSortShader.get();
		if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_computePipeline))
		{
			return logFail("Failed to create pipelines (compile & link shaders)!\n");
		}

		// Allocate the memory for buffer A and B.
		{
			// Always default the creation parameters, there's a lot of extra stuff for DirectX/CUDA interop and slotting into external engines you don't usually care about. 
			nbl::video::IGPUBuffer::SCreationParams params = {};
			params.size = bufferSize;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;

			// Create the buffer A.
			m_bufferA = m_device->createBuffer(std::move(params));
			if (!m_bufferA)
			{
				return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);
			}

			m_bufferA->setObjectDebugName("Buffer A");
			m_bufferAAddress = m_bufferA->getDeviceAddress();

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_bufferA->getMemoryReqs();
			reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();

			m_bufferAAllocation = m_device->allocate(reqs, m_bufferA.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_bufferAAllocation.isValid())
			{
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			}

			// Create the B buffer.
			m_bufferB = m_device->createBuffer(std::move(params));
			if (!m_bufferB)
			{
				return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);
			}

			m_bufferB->setObjectDebugName("Buffer B");
			m_bufferBAddress = m_bufferB->getDeviceAddress();

			m_bufferBAllocation = m_device->allocate(reqs, m_bufferB.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_bufferBAllocation.isValid())
			{
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			}

			if (!m_bufferAAllocation.memory->map({ 0ull,m_bufferAAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
			{
				return logFail("Failed to map the Device Memory!\n");
			}

			if (!m_bufferBAllocation.memory->map({ 0ull,m_bufferBAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
			{
				return logFail("Failed to map the Device Memory!\n");
			}
		}

		// Copy the random generated array to buffer A.
		const int32_t* bufferAMappedPointer = reinterpret_cast<const int32_t*>(m_bufferAAllocation.memory->getMappedPointer());
		memcpy((void*)bufferAMappedPointer, inputBufferData.data(), sizeof(int32_t) * inputBufferData.size());

		// Start of the merge sort algorithm.
		const size_t numberOfPhases = (size_t)ceil(log2(NumberOfElementsToSort));

		// A monotonically increasing counter value (for use by timeline semaphore).
		uint64_t monotonicallyIncreasingCounter = 0u;
		// Create the Semaphore (required since to start execution of phase X, phase X - 1 must complete execution on the GPU.
		m_phaseSemaphore = m_device->createSemaphore(monotonicallyIncreasingCounter);

		for (size_t phaseIndex = 1; phaseIndex <= numberOfPhases; phaseIndex++)
		{
			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(m_computeQueueFamily, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
				{
					return logFail("Failed to create Command Buffers!\n");
				}
			}

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdbuf->bindComputePipeline(m_computePipeline.get());

			const MergeSortPushData pushConstantData = {
				.buffer_a_address = phaseIndex % 2 == 1 ? m_bufferAAddress : m_bufferBAddress,
				.buffer_b_address = phaseIndex % 2 == 1 ? m_bufferBAddress : m_bufferBAddress,
				.num_elements_per_array = (uint32_t)pow(2, phaseIndex - 1),
				.buffer_length = static_cast<uint32_t>(NumberOfElementsToSort),
			};

			cmdbuf->pushConstants(m_pipelineLayout.get(), nbl::asset::IShader::ESS_COMPUTE, 0u, sizeof(pushConstantData), &pushConstantData);

			cmdbuf->dispatch((uint32_t)ceilf(NumberOfElementsToSort / powf(2.0f, phaseIndex)), 1, 1);
			cmdbuf->endDebugMarker();

			cmdbuf->end();

			{
				// queues are inherent parts of the device, ergo not refcounted (you refcount the device instead)
				IQueue* queue = m_device->getQueue(m_computeQueueFamily, 0);

				IQueue::SSubmitInfo submitInfos[1] = {};
				// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
				// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
				submitInfos[0].commandBuffers = cmdbufs;

				const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = m_phaseSemaphore.get(),.value = ++monotonicallyIncreasingCounter,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
				submitInfos[0].signalSemaphores = signals;

				queue->startCapture();
				queue->submit(submitInfos);
				queue->endCapture();

			}
			const ISemaphore::SWaitInfo wait_infos[] = { {
		.semaphore = m_phaseSemaphore.get(),
		.value = monotonicallyIncreasingCounter,
	}, };

			m_device->blockForSemaphores(wait_infos);
		}

		// Now, get the current pointer to output buffer (the output may be in either buffer A or in B).
		// Perform merge sort on the CPU to test whether GPU compute version computed the result correctly.
		auto outputBufferData = reinterpret_cast<const int32_t*>(m_bufferBAllocation.memory->getMappedPointer());

		if (numberOfPhases % 2 == 0)
		{
			outputBufferData = reinterpret_cast<const int32_t*>(m_bufferAAllocation.memory->getMappedPointer());
		}

		// Sorted input (which should match the output buffer).
		std::sort(inputBufferData.begin(), inputBufferData.end());

		for (auto i = 0; i < NumberOfElementsToSort; i++)
		{
			printf("output buffer -> %d input buffer -> %d\n", outputBufferData[i], inputBufferData[i]);
			if (outputBufferData[i] != inputBufferData[i])
			{
				return logFail("%d != %d\n", outputBufferData[i], inputBufferData[i]);
			}
		}

		printf("Merge sort succesfull!\n");

		m_device->waitIdle();
		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

private:
	uint32_t m_computeQueueFamily{};

	// Note : In this implementation of merge sort, the buffers are 'swapped' at the end of each phase (i.e buffer A / B of previous phase becomes buffer B / A buffer of current phase).
	// The data to be sorted is initially present in buffer A.
	smart_refctd_ptr<IGPUBuffer> m_bufferA{};
	smart_refctd_ptr<IGPUBuffer> m_bufferB{};

	uint64_t m_bufferAAddress{};
	uint64_t m_bufferBAddress{};

	// Allocator is a interface to anything that can dish out free memory range to bind to a buffer or image.
	nbl::video::IDeviceMemoryAllocator::SAllocation m_bufferAAllocation{};
	nbl::video::IDeviceMemoryAllocator::SAllocation m_bufferBAllocation{};

	// Semaphore is used because before completion of merge sort phase X, phase X + 1 cannot start.
	smart_refctd_ptr<ISemaphore> m_phaseSemaphore{};

	smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout{};
	smart_refctd_ptr<IGPUComputePipeline> m_computePipeline{};

	smart_refctd_ptr<IGPUShader> m_mergeSortShader{};
};

NBL_MAIN_FUNC(MergeSortApp)