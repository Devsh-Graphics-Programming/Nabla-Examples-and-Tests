#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
#include "CommonPCH/PCH.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

class CountingSortApp final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
		using device_base_t = application_templates::MonoDeviceApplication;
		using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

	public:
		// Yay thanks to multiple inheritance we cannot forward ctors anymore
		CountingSortApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
			system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

		// we stuff all our work here because its a "single shot" app
		bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
		{
			// Remember to call the base class initialization!
			if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
				return false;
			if (!asset_base_t::onAppInitialized(std::move(system)))
				return false;

			auto limits = m_physicalDevice->getLimits();
			const uint32_t WorkgroupSize = limits.maxComputeWorkGroupInvocations;
			const uint32_t MaxBucketCount = (limits.maxComputeSharedMemorySize / sizeof(uint32_t)) / 2;
			constexpr uint32_t element_count = 100000;
			const uint32_t bucket_count = std::min((uint32_t)3000, MaxBucketCount);
			const uint32_t elements_per_thread = ceil((float)ceil((float)element_count / limits.computeUnits) / WorkgroupSize);

			auto prepShader = [&](const core::string& path) -> smart_refctd_ptr<IGPUShader>
			{
				// this time we load a shader directly from a file
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = ""; // virtual root
				auto assetBundle = m_assetMgr->getAsset(path,lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					logFail("Could not load shader!");
					return nullptr;
				}

				auto source = IAsset::castDown<ICPUShader>(assets[0]);
				// The down-cast should not fail!
				assert(source);
			
				// There's two ways of doing stuff like this:
				// 1. this - modifying the asset after load
				// 2. creating a short shader source file that includes the asset you would have wanted to load
				auto overrideSource = CHLSLCompiler::createOverridenCopy(
					source.get(), "#define WorkgroupSize %d\n#define BucketCount %d\n",
					WorkgroupSize, bucket_count
				);

				// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
				auto shader = m_device->createShader(overrideSource.get());
				if (!shader)
				{
					logFail("Creation of Prefix Sum Shader from CPU Shader source failed!");
					return nullptr;
				}
				return shader;
			};
			auto prefixSumShader = prepShader("app_resources/prefix_sum_shader.comp.hlsl");
			auto scatterShader = prepShader("app_resources/scatter_shader.comp.hlsl");

			// People love Reflection but I prefer Shader Sources instead!
			const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset = 0,.size = sizeof(CountingPushData) };

			// This time we'll have no Descriptor Sets or Layouts because our workload has a widely varying size
			// and using traditional SSBO bindings would force us to update the Descriptor Set every frame.
			// I even started writing this sample with the use of Dynamic SSBOs, however the length of the buffer range is not dynamic
			// only the offset. This means that we'd have to write the "worst case" length into the descriptor set binding.
			// Then this has a knock-on effect that we couldn't allocate closer to the end of the streaming buffer than the "worst case" size.
			smart_refctd_ptr<IGPUPipelineLayout> layout;
			smart_refctd_ptr<IGPUComputePipeline> prefixSumPipeline;
			smart_refctd_ptr<IGPUComputePipeline> scatterPipeline;
			{
				layout = m_device->createPipelineLayout({ &pcRange,1 });
				IGPUComputePipeline::SCreationParams params = {};
				params.layout = layout.get();
				params.shader.shader = prefixSumShader.get();
				params.shader.entryPoint = "main";
				params.shader.entries = nullptr;
				params.shader.requireFullSubgroups = true;
				params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
				if (!m_device->createComputePipelines(nullptr, { &params,1 }, &prefixSumPipeline))
					return logFail("Failed to create compute pipeline!\n");
				params.shader.shader = scatterShader.get();
				if (!m_device->createComputePipelines(nullptr, { &params,1 }, &scatterPipeline))
					return logFail("Failed to create compute pipeline!\n");
			}

			// Allocate memory
			nbl::video::IDeviceMemoryAllocator::SAllocation allocation[5] = {};
			smart_refctd_ptr<IGPUBuffer> buffers[5];
			//smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;
			{
				auto build_buffer = [this](
					smart_refctd_ptr<ILogicalDevice> m_device,
					nbl::video::IDeviceMemoryAllocator::SAllocation *allocation,
					smart_refctd_ptr<IGPUBuffer>& buffer,
					size_t buffer_size,
					const char *label
				) -> void {
					IGPUBuffer::SCreationParams params;
					params.size = buffer_size;
					params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
					buffer = m_device->createBuffer(std::move(params));
					if (!buffer)
						logFail("Failed to create GPU buffer of size %d!\n", buffer_size);

					buffer->setObjectDebugName(label);

					auto reqs = buffer->getMemoryReqs();
					reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

					*allocation = m_device->allocate(reqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
					if (!allocation->isValid())
						logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

					assert(allocation->memory.get() == buffer->getBoundMemory().memory);
				};

				build_buffer(m_device,	allocation,		buffers[0], sizeof(uint32_t) * element_count,	"Input Key Buffer");
				build_buffer(m_device,	allocation + 1,	buffers[1], sizeof(uint32_t) * element_count,	"Input Value Buffer");
				build_buffer(m_device,	allocation + 2, buffers[2], sizeof(uint32_t) * bucket_count,	"Scratch Buffer");
				build_buffer(m_device,	allocation + 3,	buffers[3], sizeof(uint32_t) * element_count,	"Output Key Buffer");
				build_buffer(m_device,	allocation + 4, buffers[4], sizeof(uint32_t) * element_count,	"Output Value Buffer");
			}
			uint64_t buffer_device_address[] = {
				buffers[0]->getDeviceAddress(),
				buffers[1]->getDeviceAddress(),
				buffers[2]->getDeviceAddress(),
				buffers[3]->getDeviceAddress(),
				buffers[4]->getDeviceAddress()
			};

			void* mapped_memory[] = {
				allocation[0].memory->map({0ull,allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
				allocation[1].memory->map({0ull,allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
				allocation[2].memory->map({0ull,allocation[2].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
				allocation[3].memory->map({0ull,allocation[3].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
				allocation[4].memory->map({0ull,allocation[3].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			};
			if (!mapped_memory[0] || !mapped_memory[1] || !mapped_memory[2] || !mapped_memory[3] || !mapped_memory[4])
				return logFail("Failed to map the Device Memory!\n");

			// Generate random data
			constexpr uint32_t minimum = 0;
			const uint32_t range = bucket_count;
			unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
			std::mt19937 g(seed);

			auto bufferData = new uint32_t[2][element_count];
			for (uint32_t i = 0; i < element_count; i++) {
				bufferData[0][i] = minimum + g() % range;
			}

			memcpy(mapped_memory[0], bufferData[0], sizeof(uint32_t) * element_count);

			for (uint32_t i = 0; i < element_count; i++) {
				bufferData[1][i] = g() % std::numeric_limits<uint32_t>::max();
			}

			memcpy(mapped_memory[1], bufferData[1], sizeof(uint32_t) * element_count);

			std::string outBuffer;
			for (auto i = 0; i < element_count; i++) {
				outBuffer.append("{");
				outBuffer.append(std::to_string(bufferData[0][i]));
				outBuffer.append(", ");
				outBuffer.append(std::to_string(bufferData[1][i]));
				outBuffer.append("} ");
			}
			outBuffer.append("\n");
			outBuffer.append("Count: ");
			outBuffer.append(std::to_string(element_count));
			outBuffer.append("\n");
			m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

			auto pc = CountingPushData {
				.inputKeyAddress = buffer_device_address[0],
				.inputValueAddress = buffer_device_address[1],
				.histogramAddress = buffer_device_address[2],
				.outputKeyAddress = buffer_device_address[3],
				.outputValueAddress = buffer_device_address[4],
				.dataElementCount = element_count,
				.elementsPerWT = elements_per_thread,
				.minimum = minimum,
				.maximum = minimum + bucket_count - 1,
			};

			smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
			{
				smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
				if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
					return logFail("Failed to create Command Buffers!\n");
			}

			// Create the Semaphore for prefix sum
			constexpr uint64_t started_value = 0;
			uint64_t timeline = started_value;
			smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(started_value);

			cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdBuf->beginDebugMarker("Prefix Sum Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdBuf->bindComputePipeline(prefixSumPipeline.get());
			cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
			cmdBuf->dispatch(ceil((float)element_count / (elements_per_thread * WorkgroupSize)), 1, 1);
			cmdBuf->endDebugMarker();
			cmdBuf->end();

			{
				auto queue = getComputeQueue();

				IQueue::SSubmitInfo submit_infos[1];
				IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = {
					{
						.cmdbuf = cmdBuf.get()
					}
				};
				submit_infos[0].commandBuffers = cmdBufs;
				IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
					{
						.semaphore = progress.get(),
						.value = ++timeline,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
					}
				};
				submit_infos[0].signalSemaphores = signals;

				m_api->startCapture();
				queue->submit(submit_infos);
				m_api->endCapture();
			}

			const ISemaphore::SWaitInfo wait_infos[] = { {
					.semaphore = progress.get(),
					.value = timeline
				} };
			m_device->blockForSemaphores(wait_infos);

			// Create the Semaphore for Scatter
			uint64_t timeline2 = started_value;
			smart_refctd_ptr<ISemaphore> progress2 = m_device->createSemaphore(started_value);

			cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
			cmdBuf->beginDebugMarker("Scatter Dispatch", core::vectorSIMDf(0, 1, 0, 1));
			cmdBuf->bindComputePipeline(scatterPipeline.get());
			cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
			cmdBuf->dispatch(ceil((float)element_count / (elements_per_thread * WorkgroupSize)), 1, 1);
			cmdBuf->endDebugMarker();
			cmdBuf->end();

			{
				auto queue = getComputeQueue();

				IQueue::SSubmitInfo submit_infos[1];
				IQueue::SSubmitInfo::SCommandBufferInfo cmdBufs[] = {
					{
						.cmdbuf = cmdBuf.get()
					}
				};
				submit_infos[0].commandBuffers = cmdBufs;
				IQueue::SSubmitInfo::SSemaphoreInfo waits[] = {
					{
						.semaphore = progress.get(),
						.value = timeline,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
					}
				};
				submit_infos[0].waitSemaphores = waits;
				IQueue::SSubmitInfo::SSemaphoreInfo signals[] = {
					{
						.semaphore = progress2.get(),
						.value = ++timeline2,
						.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
					}
				};
				submit_infos[0].signalSemaphores = signals;

				m_api->startCapture();
				queue->submit(submit_infos);
				m_api->endCapture();
			}

			const ISemaphore::SWaitInfo wait_infos2[] = {{
					.semaphore = progress2.get(),
					.value = timeline2
				} };
			m_device->blockForSemaphores(wait_infos2);

			const ILogicalDevice::MappedMemoryRange memory_range[] = {
				ILogicalDevice::MappedMemoryRange(allocation[0].memory.get(), 0ull, allocation[0].memory->getAllocationSize()),
				ILogicalDevice::MappedMemoryRange(allocation[1].memory.get(), 0ull, allocation[1].memory->getAllocationSize()),
				ILogicalDevice::MappedMemoryRange(allocation[2].memory.get(), 0ull, allocation[2].memory->getAllocationSize()),
				ILogicalDevice::MappedMemoryRange(allocation[3].memory.get(), 0ull, allocation[3].memory->getAllocationSize()),
				ILogicalDevice::MappedMemoryRange(allocation[4].memory.get(), 0ull, allocation[4].memory->getAllocationSize())
			};

			if (!allocation[0].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memory_range[0]);
			if (!allocation[1].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memory_range[1]);
			if (!allocation[2].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memory_range[2]);
			if (!allocation[3].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memory_range[3]);
			if (!allocation[4].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
				m_device->invalidateMappedMemoryRanges(1, &memory_range[4]);

			const uint32_t* buffData[] = {
				reinterpret_cast<const uint32_t*>(allocation[2].memory->getMappedPointer()),
				reinterpret_cast<const uint32_t*>(allocation[3].memory->getMappedPointer()),
				reinterpret_cast<const uint32_t*>(allocation[4].memory->getMappedPointer())
			};

			assert(allocation[2].offset == 0); // simpler than writing out all the pointer arithmetic
			assert(allocation[3].offset == 0); // simpler than writing out all the pointer arithmetic
			assert(allocation[4].offset == 0); // simpler than writing out all the pointer arithmetic

			outBuffer.clear();
			for (auto i = 0; i < bucket_count; i++) {
				outBuffer.append(std::to_string(buffData[0][i]));
				outBuffer.append(" ");
			}
			outBuffer.append("\n");

			m_logger->log("Scratch buffer is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

			outBuffer.clear();
			for (auto i = 0; i < element_count; i++) {
				outBuffer.append("{");
				outBuffer.append(std::to_string(buffData[1][i]));
				outBuffer.append(", ");
				outBuffer.append(std::to_string(buffData[2][i]));
				outBuffer.append("} ");
			}
			outBuffer.append("\n");
			outBuffer.append("Count: ");
			outBuffer.append(std::to_string(element_count));
			outBuffer.append("\n");
			m_logger->log("Your output array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

			allocation[0].memory->unmap();
			allocation[1].memory->unmap();
			allocation[2].memory->unmap();
			allocation[3].memory->unmap();
			allocation[4].memory->unmap();

			m_device->waitIdle();

			delete[] bufferData;

			return true;
		}

		// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
		bool keepRunning() override { return false; }

		// Finally the first actual work-loop
		void workLoopBody() override {}

		bool onAppTerminated() override { return true; }
};


NBL_MAIN_FUNC(CountingSortApp)