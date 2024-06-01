#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// A high level overview of the algorithm implementation (merge sort).
// This will be a X pass algorithm, where X is log2(number_of_elements).
// In each phase, you have to 'merge' 2 sorted list of elements into a single list.
// Each of these 'lists' will have thier number of elements roughly doubled in each phase (after the merge step of previous phase).
class MergeSortApp final : public nbl::application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoSystemMonoLoggerApplication;
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
		if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
		{

			return false;
		}

		IPhysicalDevice* physicalDevice{};

		// Compute and input related constants.
		static constexpr uint32_t WorkgroupSize = 64;
		static constexpr uint32_t NumberOfElementsToSort = 64 * 64 * 64;
		static const uint32_t WorkgroupCount = (uint32_t)ceil(NumberOfElementsToSort / (float)WorkgroupSize);

		const size_t BufferSize = sizeof(int32_t) * WorkgroupSize * WorkgroupCount;

		std::vector<int32_t> inputBufferData(WorkgroupSize * WorkgroupCount);
		{
			// Setup the input data (random) that is to be sorted.
			const auto seed = std::chrono::system_clock::now().time_since_epoch().count();

			// Setup the random number generator.
			std::mt19937 randomNumberGenerator(seed);

			for (auto& elem : inputBufferData)
			{
				elem = randomNumberGenerator();
			}
		}

		// Setup API connection, select appropriate physical device and create logical device. 
		{
			smart_refctd_ptr<nbl::video::CVulkanConnection> api;
			{
				nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable = {};
				apiFeaturesToEnable.validations = true;
				apiFeaturesToEnable.synchronizationValidation = true;
				apiFeaturesToEnable.debugUtils = true;

				if (!(api = CVulkanConnection::create(smart_refctd_ptr(m_system), 0, _NBL_APP_NAME_, smart_refctd_ptr(device_base_t::m_logger), apiFeaturesToEnable)))
					return logFail("Failed to crate an IAPIConnection!");
			}

			ILogicalDevice::SCreationParams params = {};
			for (auto physDevIt = api->getPhysicalDevices().begin(); physDevIt != api->getPhysicalDevices().end(); physDevIt++)
			{
				const auto familyProps = (*physDevIt)->getQueueFamilyProperties();
				for (auto i = 0; i < familyProps.size(); i++)
				{
					if (familyProps[i].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT))
					{
						physicalDevice = *physDevIt;
						m_computeQueueFamily = i;
						params.queueParams[m_computeQueueFamily].count = 1;
						break;
					}
				}
			}

			if (!physicalDevice)
			{
				return logFail("Failed to find any Physical Devices with Compute capable Queue Families!");
			}

			// logical devices need to be created form physical devices which will actually let us create vulkan objects and use the physical device
			m_device = physicalDevice->createLogicalDevice(std::move(params));
			if (!m_device)
			{
				return logFail("Failed to create a Logical Device!");
			}
		}

		// Create and setup the compute shader.
		{
			IAssetLoader::SAssetLoadParams assetLoadParams = {};
			assetLoadParams.logger = m_logger.get();
			assetLoadParams.workingDirectory = "";

			const auto shaderAsset = m_assetMgr->getAsset("app_resources/merge_sort.hlsl", assetLoadParams);
			const auto shaderAssetContents = shaderAsset.getContents();
			if (shaderAssetContents.empty())
			{
				return logFail("Failed to load shader asset with file name app_resources/merge_sort.hlsl");
			}

			const auto cpuShaderAsset = IAsset::castDown<ICPUShader>(shaderAssetContents[0]);

			// Asset to ensure downcasting did not fail.
			assert(cpuShaderAsset);

			// Now, compile the shader.
			// During compilation, replace the value of WORKGROUP_SIZE in the shader source.
			smart_refctd_ptr<nbl::asset::IShaderCompiler> compiler = make_smart_refctd_ptr<nbl::asset::CHLSLCompiler>(smart_refctd_ptr(m_system));

			// Yes we know workgroup sizes can come from specialization constants, however DXC has a problem with that https://github.com/microsoft/DirectXShaderCompiler/issues/3092
			const string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
			const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE",WorkgroupSizeAsStr };

			const auto overridenSource = CHLSLCompiler::createOverridenCopy(
				cpuShaderAsset.get(), "WORKGROUP_SIZE",
				WorkgroupSize
			);

			m_mergeSortShader = m_device->createShader(overridenSource.get());
			if (!m_mergeSortShader)
			{
				return logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");
			}
		}

		// The shader has 2 bindings, both of which are storage buffers.
		// This is because it is not *guarenteed* which buffer the output is in as the buffers are swapped each phase.
		const nbl::video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
			{
				.binding = 0,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IGPUShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr,
			},
			{
				.binding = 1,
				.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
				.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
				.stageFlags = IGPUShader::ESS_COMPUTE,
				.count = 1,
				.samplers = nullptr,
			},
		};

		m_descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
		if (!m_descriptorSetLayout)
		{
			return logFail("Failed to create a Descriptor Layout!\n");
		}

		// The shader excepts 2 uint32_t's in the push constant. One for the length of array, one for current phase index.
		const nbl::asset::SPushConstantRange pushConstantRange = nbl::asset::SPushConstantRange{
			.stageFlags = nbl::asset::IShader::ESS_COMPUTE, .offset = 0, .size = sizeof(uint32_t) * 2,
		};

		m_pipelineLayout = m_device->createPipelineLayout({ &pushConstantRange, 1 }, smart_refctd_ptr(m_descriptorSetLayout));
		if (!m_pipelineLayout)
		{
			return logFail("Failed to create a Pipeline Layout!\n");
		}

		IGPUComputePipeline::SCreationParams params = {};
		params.layout = m_pipelineLayout.get();
		params.shader.entryPoint = "main";
		params.shader.shader = m_mergeSortShader.get();
		if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_computePipeline))
		{
			return logFail("Failed to create pipelines (compile & link shaders)!\n");
		}

		// Allocate the memory for output and input buffer.
		{
			// Always default the creation parameters, there's a lot of extra stuff for DirectX/CUDA interop and slotting into external engines you don't usually care about. 
			nbl::video::IGPUBuffer::SCreationParams params = {};
			params.size = BufferSize;
			params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;

			// Create the output buffer.
			m_outputBuffer = m_device->createBuffer(std::move(params));
			if (!m_outputBuffer)
			{
				return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);
			}

			m_outputBuffer->setObjectDebugName("My Output Buffer");

			nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = m_outputBuffer->getMemoryReqs();
			reqs.memoryTypeBits &= physicalDevice->getHostVisibleMemoryTypeBits();

			m_outputBufferAllocation = m_device->allocate(reqs, m_outputBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_outputBufferAllocation.isValid())
			{
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			}

			// Create the input buffer.
			m_inputBuffer = m_device->createBuffer(std::move(params));
			if (!m_inputBuffer)
			{
				return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);
			}

			m_inputBuffer->setObjectDebugName("My Input Buffer");

			m_inputBufferAllocation = m_device->allocate(reqs, m_inputBuffer.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
			if (!m_inputBufferAllocation.isValid())
			{
				return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");
			}

			// Create 2 descriptor pools, one for each descriptor set.
			smart_refctd_ptr<nbl::video::IDescriptorPool> pool1 = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &m_descriptorSetLayout.get(),1 });
			smart_refctd_ptr<nbl::video::IDescriptorPool> pool2 = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &m_descriptorSetLayout.get(),1 });

			// Create the descriptor sets.
			m_descriptorSet1 = pool1->createDescriptorSet(m_descriptorSetLayout);
			m_descriptorSet2 = pool2->createDescriptorSet(m_descriptorSetLayout);

			{
				IGPUDescriptorSet::SDescriptorInfo info[2];
				info[0].desc = smart_refctd_ptr(m_outputBuffer);
				info[0].info.buffer = { .offset = 0,.size = BufferSize };
				info[1].desc = smart_refctd_ptr(m_inputBuffer);
				info[1].info.buffer = { .offset = 0,.size = BufferSize };
				IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
					{.dstSet = m_descriptorSet1.get(),.binding = 0,.arrayElement = 0,.count = 2,.info = info}
				};
				m_device->updateDescriptorSets(writes, {});
			}

			{
				IGPUDescriptorSet::SDescriptorInfo info[2];
				info[1].desc = smart_refctd_ptr(m_outputBuffer);
				info[1].info.buffer = { .offset = 0,.size = BufferSize };
				info[0].desc = smart_refctd_ptr(m_inputBuffer);
				info[0].info.buffer = { .offset = 0,.size = BufferSize };
				IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
					{.dstSet = m_descriptorSet2.get(),.binding = 0,.arrayElement = 0,.count = 2,.info = info}
				};
				m_device->updateDescriptorSets(writes, {});
			}

			if (!m_outputBufferAllocation.memory->map({ 0ull,m_outputBufferAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
			{
				return logFail("Failed to map the Device Memory!\n");
			}

			if (!m_inputBufferAllocation.memory->map({ 0ull,m_inputBufferAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
			{
				return logFail("Failed to map the Device Memory!\n");
			}
		}

		// Copy the random generated array to input buffer.
		const int32_t* inputBufferMappedPointer = reinterpret_cast<const int32_t*>(m_inputBufferAllocation.memory->getMappedPointer());
		memcpy((void*)inputBufferMappedPointer, inputBufferData.data(), sizeof(int32_t) * inputBufferData.size());

		// Start of the merge sort algorithm.
		const size_t numberOfPhases = (size_t)log2(WorkgroupCount * WorkgroupSize);
		const size_t numberOfElements = static_cast<size_t>(WorkgroupCount) * WorkgroupSize;

		for (size_t phaseIndex = 1; phaseIndex <= numberOfPhases; phaseIndex++)
		{
			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(m_computeQueueFamily, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
					return logFail("Failed to create Command Buffers!\n");
			}

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdbuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdbuf->bindComputePipeline(m_computePipeline.get());

			// If phase index is ODD, the inpt and output buffer are as usual. 
			// If phase index is EVEN, input buffer is the output buffer of last phase.
			if (phaseIndex % 2 == 1)
			{
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, &m_descriptorSet1.get());
			}
			else
			{
				cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pipelineLayout.get(), 0, 1, &m_descriptorSet2.get());
			}

			struct SortingPhaseData
			{
				uint32_t numElementsPerArray;
				uint32_t bufferLength;
			};

			const SortingPhaseData pushConstantData = {
				.numElementsPerArray = (uint32_t)pow(2, phaseIndex - 1),
				.bufferLength = static_cast<uint32_t>(numberOfElements),
			};

			cmdbuf->pushConstants(m_pipelineLayout.get(), nbl::asset::IShader::ESS_COMPUTE, 0u, sizeof(pushConstantData), &pushConstantData);

			cmdbuf->dispatch(numberOfElements / (uint32_t)ceilf(numberOfElements / powf(2, phaseIndex)), 1, 1);
			cmdbuf->endDebugMarker();

			cmdbuf->end();

			// Create the Semaphore (required since to start execution of phase X, phase X - 1 must complete execution on the GPU.
			const auto SemaphoreValue = phaseIndex;
			m_currentPhaseSemaphore = m_device->createSemaphore(SemaphoreValue);
			{
				// queues are inherent parts of the device, ergo not refcounted (you refcount the device instead)
				IQueue* queue = m_device->getQueue(m_computeQueueFamily, 0);

				IQueue::SSubmitInfo submitInfos[1] = {};
				// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
				// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
				submitInfos[0].commandBuffers = cmdbufs;

				const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = m_currentPhaseSemaphore.get(),.value = phaseIndex,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
				submitInfos[0].signalSemaphores = signals;

				queue->startCapture();
				queue->submit(submitInfos);
				queue->endCapture();

				if (m_previousPhaseSemaphore)
				{
					const ISemaphore::SWaitInfo wait_infos[] = { {
				.semaphore = m_previousPhaseSemaphore.get(),
				.value = phaseIndex - 1,
			} };

					m_device->blockForSemaphores(wait_infos);
				}

				// Swap the semaphores.
				std::swap(m_previousPhaseSemaphore, m_currentPhaseSemaphore);
			}
		}


		// For the both semaphores to complete execution.
		const ISemaphore::SWaitInfo waitInfos[] = { {
			.semaphore = m_currentPhaseSemaphore.get(),
			.value = numberOfPhases,
		}, {
			.semaphore = m_previousPhaseSemaphore.get(),
			.value = numberOfPhases - 1,
		} };

		m_device->blockForSemaphores(waitInfos);

		// Now, get the current pointer to output buffer (the output may be in either input or output buffer).
		// Perform merge sort on the CPU to test whether GPU compute version computed the result correctly.
		auto outputBufferData = reinterpret_cast<const int32_t*>(m_outputBufferAllocation.memory->getMappedPointer());

		if (numberOfPhases % 2 == 0)
		{
			outputBufferData = reinterpret_cast<const int32_t*>(m_inputBufferAllocation.memory->getMappedPointer());
		}

		// Sorted input (which should match the output buffer).
		std::sort(inputBufferData.begin(), inputBufferData.end());

		for (auto i = 0; i < WorkgroupSize * WorkgroupCount; i++)
		{
			printf("%d %d\n", outputBufferData[i], inputBufferData[i]);
			if (outputBufferData[i] != inputBufferData[i])
			{
				return logFail("%d != %d\n", outputBufferData[i], inputBufferData[i]);
			}
		}

		printf("Merge sort succesfull!");

		m_device->waitIdle();

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

private:
	smart_refctd_ptr<nbl::video::ILogicalDevice> m_device{};
	uint32_t m_computeQueueFamily;

	// Note : In this implementation of merge sort, the buffers are 'swapped' at the end of each phase (i.e input / output buffer of previous phase becomes the output / input of current phase).
	// The names reflect the fact that initially the input data is present in m_inputBuffer.
	smart_refctd_ptr<IGPUBuffer> m_outputBuffer{};
	smart_refctd_ptr<IGPUBuffer> m_inputBuffer{};

	// Allocator is a interface to anything that can dish out free memory range to bind to a buffer or image.
	nbl::video::IDeviceMemoryAllocator::SAllocation m_outputBufferAllocation = {};
	nbl::video::IDeviceMemoryAllocator::SAllocation m_inputBufferAllocation = {};

	// Semaphore is used because before completion of merge sort phase X, phase X + 1 cannot start.
	smart_refctd_ptr<ISemaphore> m_currentPhaseSemaphore{};
	smart_refctd_ptr<ISemaphore> m_previousPhaseSemaphore{};

	smart_refctd_ptr<IGPUDescriptorSetLayout> m_descriptorSetLayout;

	// There are 2 descriptor sets for merge sort. One has input buffer / output buffer at binding 0, 1, and one has the inverse.
	// This is because in each merge sort phase, the inputs and output buffers are *swapped*.
	smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_descriptorSet1;
	smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_descriptorSet2;

	smart_refctd_ptr<IGPUPipelineLayout> m_pipelineLayout;
	smart_refctd_ptr<IGPUComputePipeline> m_computePipeline;

	smart_refctd_ptr<IGPUShader> m_mergeSortShader{};
};

NBL_MAIN_FUNC(MergeSortApp)