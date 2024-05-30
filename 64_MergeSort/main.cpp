#include "nbl/application_templates/MonoSystemMonoLoggerApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

// A high level overview of the algorithm implementation (merge sort) in mind.
// This will be a X pass algorithm, where X is log2(number_of_elements).
// In each phase, you have to 'merge' 2 sorted list of elements into a single list.
// Each of these 'lists' will have thier number of elements roughly doubled in each phase (after the merge step of previous phase).
class MergeSortApp final : public nbl::application_templates::MonoSystemMonoLoggerApplication
{
	using base_t = application_templates::MonoSystemMonoLoggerApplication;
public:
	using base_t::base_t;

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!base_t::onAppInitialized(std::move(system)))
			return false;

		smart_refctd_ptr<nbl::video::ILogicalDevice> device;

		constexpr auto FinishedValue = 45;
		smart_refctd_ptr<ISemaphore> progress;

		// A `nbl::video::DeviceMemoryAllocator` is an interface to implement anything that can dish out free memory range to bind to back a `nbl::video::IGPUBuffer` or a `nbl::video::IGPUImage`
		// The Logical Device itself implements the interface and behaves as the most simple allocator, it will create a new `nbl::video::IDeviceMemoryAllocation` every single time.
		nbl::video::IDeviceMemoryAllocator::SAllocation outputBufferAllocation = {};
		nbl::video::IDeviceMemoryAllocator::SAllocation inputBufferAllocation = {};

		// For our Compute Shader
		constexpr uint32_t WorkgroupSize = 32;
		constexpr uint32_t WorkgroupCount = 2;

		// Setup the random unsorted merge sort input data.
		const auto seed = std::chrono::system_clock::now().time_since_epoch().count();

		// Setup the random number generator.
		std::mt19937 randomNumberGenerator(seed);

		std::vector<int32_t> inputBufferData(WorkgroupSize * WorkgroupCount);

		for (auto& elem : inputBufferData)
		{
			elem = randomNumberGenerator();
		}

		{
			smart_refctd_ptr<nbl::video::CVulkanConnection> api;
			{
				nbl::video::IAPIConnection::SFeatures apiFeaturesToEnable = {};
				apiFeaturesToEnable.validations = true;
				apiFeaturesToEnable.synchronizationValidation = true;
				apiFeaturesToEnable.debugUtils = true;

				if (!(api = CVulkanConnection::create(smart_refctd_ptr(m_system), 0, _NBL_APP_NAME_, smart_refctd_ptr(base_t::m_logger), apiFeaturesToEnable)))
					return logFail("Failed to crate an IAPIConnection!");
			}

			uint8_t queueFamily;
			nbl::video::IPhysicalDevice* physDev = nullptr;
			ILogicalDevice::SCreationParams params = {};
			for (auto physDevIt = api->getPhysicalDevices().begin(); physDevIt != api->getPhysicalDevices().end(); physDevIt++)
			{
				const auto familyProps = (*physDevIt)->getQueueFamilyProperties();
				// this is the only "complicated" part, we want to create a queue that supports compute pipelines
				for (auto i = 0; i < familyProps.size(); i++)
					if (familyProps[i].queueFlags.hasFlags(IQueue::FAMILY_FLAGS::COMPUTE_BIT))
					{
						physDev = *physDevIt;
						queueFamily = i;
						params.queueParams[queueFamily].count = 1;
						break;
					}
			}
			if (!physDev)
				return logFail("Failed to find any Physical Devices with Compute capable Queue Families!");

			// logical devices need to be created form physical devices which will actually let us create vulkan objects and use the physical device
			device = physDev->createLogicalDevice(std::move(params));
			if (!device)
				return logFail("Failed to create a Logical Device!");

			smart_refctd_ptr<nbl::asset::ICPUShader> cpuShader;
			{
				smart_refctd_ptr<nbl::asset::IShaderCompiler> compiler = make_smart_refctd_ptr<nbl::asset::CHLSLCompiler>(smart_refctd_ptr(m_system));

				constexpr const char* source = R"===(
						#pragma wave shader_stage(compute)

						[[vk::binding(0,0)]] RWStructuredBuffer<int32_t> output_buffer;
						[[vk::binding(1,0)]] RWStructuredBuffer<int32_t> input_buffer;

						struct SortingPhaseData
						{
							uint current_sorting_phase;
						};


						[[vk::push_constant]] SortingPhaseData phase_data;

						// This shader will be invoked in log(X) phases (x = number of elements in input buffer). In each phase, number of threads launched is 
						// X / (phase_index * 2). Each of these threads map to the arrays thread index.x * phase_index * 2, and this thread will sort
						// the elements ????? some index ?????
						[numthreads(WORKGROUP_SIZE,1,1)]
						void main(uint32_t3 ID : SV_DispatchThreadID)
						{
							const uint mapped_thread_index = ID.x * phase_data.current_sorting_phase * 2;

							uint left = ID.x * phase_data.current_sorting_phase * 2;
							uint right = left + phase_data.current_sorting_phase;
							uint end = left + phase_data.current_sorting_phase * 2 - 1;

							// Copy to output buffer.
							for (uint i = left; i < end ; i++) output_buffer[i] = input_buffer[i];

							// Sort the buffers.
							uint index = left;
							while (left <= right && right <= end)
							{
								if (input_buffer[left] < input_buffer[right])
								{
									output_buffer[index++] = input_buffer[left];
									++left;
								}	
								else 
								{
									output_buffer[index++] = input_buffer[right];
									++right;
								}	
							}
						}
					)===";

				// Yes we know workgroup sizes can come from specialization constants, however DXC has a problem with that https://github.com/microsoft/DirectXShaderCompiler/issues/3092
				const string WorkgroupSizeAsStr = std::to_string(WorkgroupSize);
				const IShaderCompiler::SMacroDefinition WorkgroupSizeDefine = { "WORKGROUP_SIZE",WorkgroupSizeAsStr };

				CHLSLCompiler::SOptions options = {};
				options.stage = asset::IShader::E_SHADER_STAGE::ESS_COMPUTE;
				options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_LINE_BIT;
				if (physDev->getLimits().shaderNonSemanticInfo)
					options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_NON_SEMANTIC_BIT;

				options.preprocessorOptions.sourceIdentifier = "embedded.comp.hlsl";
				options.preprocessorOptions.logger = m_logger.get();
				options.preprocessorOptions.extraDefines = { &WorkgroupSizeDefine,&WorkgroupSizeDefine + 1 };
				if (!(cpuShader = compiler->compileToSPIRV(source, options)))
					return logFail("Failed to compile following HLSL Shader:\n%s\n", source);
			}

			smart_refctd_ptr<nbl::video::IGPUShader> shader = device->createShader(cpuShader.get());
			if (!shader)
				return logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

			nbl::video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
				{
					.binding = 0,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, // not is not the time for descriptor indexing
					.stageFlags = IGPUShader::ESS_COMPUTE,
					.count = 1,
					.samplers = nullptr // irrelevant for a buffer
				},
				{
					.binding = 1,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, // not is not the time for descriptor indexing
					.stageFlags = IGPUShader::ESS_COMPUTE,
					.count = 1,
					.samplers = nullptr // irrelevant for a buffer
				},
			};
			smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = device->createDescriptorSetLayout(bindings);
			if (!dsLayout)
				return logFail("Failed to create a Descriptor Layout!\n");

			const nbl::asset::SPushConstantRange pushConstantRange = nbl::asset::SPushConstantRange{
				.stageFlags = nbl::asset::IShader::ESS_COMPUTE, .offset = 0, .size = sizeof(uint32_t),
			};

			smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = device->createPipelineLayout({ &pushConstantRange, 1 }, smart_refctd_ptr(dsLayout));
			if (!pplnLayout)
				return logFail("Failed to create a Pipeline Layout!\n");

			smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
			{
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = pplnLayout.get();
				params.shader.entryPoint = "main";
				params.shader.shader = shader.get();
				if (!device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
					return logFail("Failed to create pipelines (compile & link shaders)!\n");
			}

			smart_refctd_ptr<nbl::video::IGPUDescriptorSet> descriptorSet;

			// Allocate the memory for output and input buffer.
			{
				constexpr size_t BufferSize = sizeof(int32_t) * WorkgroupSize * WorkgroupCount;

				// Always default the creation parameters, there's a lot of extra stuff for DirectX/CUDA interop and slotting into external engines you don't usually care about. 
				nbl::video::IGPUBuffer::SCreationParams params = {};
				params.size = BufferSize;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;

				// Create the output buffer.
				smart_refctd_ptr<IGPUBuffer> outputBuff = device->createBuffer(std::move(params));
				if (!outputBuff)
					return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

				outputBuff->setObjectDebugName("My Output Buffer");

				nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
				reqs.memoryTypeBits &= physDev->getHostVisibleMemoryTypeBits();

				outputBufferAllocation = device->allocate(reqs, outputBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
				if (!outputBufferAllocation.isValid())
					return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

				smart_refctd_ptr<IGPUBuffer> inputBuff = device->createBuffer(std::move(params));
				if (!inputBuff)
					return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

				inputBuff->setObjectDebugName("My Input Buffer");

				inputBufferAllocation = device->allocate(reqs, inputBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
				if (!inputBufferAllocation.isValid())
					return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

				smart_refctd_ptr<nbl::video::IDescriptorPool> pool = device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

				descriptorSet = pool->createDescriptorSet(std::move(dsLayout));

				{
					IGPUDescriptorSet::SDescriptorInfo info[2];
					info[0].desc = smart_refctd_ptr(outputBuff);
					info[0].info.buffer = { .offset = 0,.size = BufferSize };
					info[1].desc = smart_refctd_ptr(inputBuff);
					info[1].info.buffer = { .offset = 0,.size = BufferSize };
					IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
						{.dstSet = descriptorSet.get(),.binding = 0,.arrayElement = 0,.count = 2,.info = info}
					};
					device->updateDescriptorSets(writes, {});
				}

				if (!outputBufferAllocation.memory->map({ 0ull,outputBufferAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
					return logFail("Failed to map the Device Memory!\n");

				if (!inputBufferAllocation.memory->map({ 0ull,inputBufferAllocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
					return logFail("Failed to map the Device Memory!\n");

			}

			// Copy the random generated array to input buffer.
			const int32_t* inputBufferMappedPointer = reinterpret_cast<const int32_t*>(inputBufferAllocation.memory->getMappedPointer());
			memcpy((void*)inputBufferMappedPointer, inputBufferData.data(), sizeof(int32_t) * inputBufferData.size());

			// Our commandbuffers are cool because they refcount the resources used by each command you record into them, so you can rely a commandbuffer on keeping them alive.
			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = device->createCommandPool(queueFamily, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
					return logFail("Failed to create Command Buffers!\n");
			}

			cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			// If you enable the `debugUtils` API Connection feature on a supported backend as we've done, you'll get these pretty debug sections in RenderDoc
			cmdbuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdbuf->bindComputePipeline(pipeline.get());
			cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &descriptorSet.get());

			// Merge sort occurs in log(X) - 1 passes (where X = number of elements).
			// Each thread index maps to index (ID.x * (phase_index * 2)) and sorts 2 inplace subararys, which start from ID.x * (phase_index * 2) and 
			// ID.x * (phase_index * 2) and ID.x * (phase_index * 2) + phase_index;

			const size_t numberOfPhases = (size_t)log2(WorkgroupCount * WorkgroupSize);
			constexpr size_t numberOfElements = WorkgroupCount * WorkgroupSize;
			for (size_t phaseIndex = 1; phaseIndex <= numberOfPhases; phaseIndex++)
			{
				const uint32_t pushConstantData = phaseIndex;
				cmdbuf->pushConstants(pplnLayout.get(), nbl::asset::IShader::ESS_COMPUTE, 0u, sizeof(uint32_t), &pushConstantData);

				cmdbuf->dispatch(numberOfElements / (phaseIndex * 2), 1, 1);
			}

			cmdbuf->endDebugMarker();

			// Normally you'd want to perform a memory barrier when using the output of a compute shader or renderpass,
			// however signalling a timeline semaphore with the COMPUTE stage mask and waiting for it on the Host makes all Device writes visible.
			cmdbuf->end();

			// Create the Semaphore
			constexpr auto StartedValue = 0;
			static_assert(StartedValue < FinishedValue);
			progress = device->createSemaphore(StartedValue);
			{
				// queues are inherent parts of the device, ergo not refcounted (you refcount the device instead)
				IQueue* queue = device->getQueue(queueFamily, 0);

				// Default, we have no semaphores to wait on before we can start our workload
				IQueue::SSubmitInfo submitInfos[1] = {};
				// The IGPUCommandBuffer is the only object whose usage does not get automagically tracked internally, you're responsible for holding onto it as long as the GPU needs it.
				// So this is why our commandbuffer, even though its transient lives in the scope equal or above the place where we wait for the submission to be signalled as complete.
				const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
				submitInfos[0].commandBuffers = cmdbufs;
				// But we do need to signal completion by incrementing the Timeline Semaphore counter as soon as the compute shader is done
				const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
				submitInfos[0].signalSemaphores = signals;

				// We have a cool integration with RenderDoc that allows you to start and end captures programmatically.
				// This is super useful for debugging multi-queue workloads and by default RenderDoc delimits captures only by Swapchain presents.
				queue->startCapture();
				queue->submit(submitInfos);
				queue->endCapture();
			}
		}


		// As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
		const ISemaphore::SWaitInfo waitInfos[] = { {
			.semaphore = progress.get(),
			.value = FinishedValue
		} };
		device->blockForSemaphores(waitInfos);


		// Perform merge sort on the CPU to test whether GPU compute version computed the result correctly.
		auto outputBuffData = reinterpret_cast<const int32_t*>(outputBufferAllocation.memory->getMappedPointer());

		// Sorted input (which should match the output buffer).
		//std::sort(inputBufferData.begin(), inputBufferData.end());

		for (auto i = 0; i < WorkgroupSize * WorkgroupCount; i++)
		{
			printf("%d != %d\n", outputBuffData[i], inputBufferData[i]);
			//if (outputBuffData[i] != inputBufferData[i])
				//return logFail("%d != %d\n", outputBuffData[i], inputBufferData[i]);
		}

		// This allocation would unmap itself in the dtor anyway, but lets showcase the API usage
		outputBufferAllocation.memory->unmap();

		// There's just one caveat, the Queues tracking what resources get used in a submit do it via an event queue that needs to be polled to clear.
		// The tracking causes circular references from the resource back to the device, so unless we poll at the end of the application, they resources used by last submit will leak.
		// We could of-course make a very lazy thread that wakes up every second or so and runs this GC on the queues, but we think this is enough book-keeping for the users.
		device->waitIdle();

		return true;
	}

	// Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
	void workLoopBody() override {}

	// Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
	bool keepRunning() override { return false; }

};


NBL_MAIN_FUNC(MergeSortApp)