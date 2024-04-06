#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

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
		constexpr size_t element_count = 100000;
		const size_t bucket_count = std::min((uint32_t)3000, MaxBucketCount);
		const uint32_t elements_per_thread = ceil((float)ceil((float)element_count / limits.computeUnits) / WorkgroupSize);

		// this time we load a shader directly from a file
		smart_refctd_ptr<IGPUShader> prefixSumShader;
		smart_refctd_ptr<IGPUShader> scatterShader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/prefix_sum_shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			// The down-cast should not fail!
			assert(source);

			auto overrideSource = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define WorkgroupSize %d\n#define BucketCount %d\n",
				WorkgroupSize, bucket_count
			);

			// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
			prefixSumShader = m_device->createShader(overrideSource.get());
			if (!prefixSumShader)
				return logFail("Creation of Prefix Sum Shader from CPU Shader source failed!");
		}
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/scatter_shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			// The down-cast should not fail!
			assert(source);

			auto overrideSource = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define WorkgroupSize %d\n",
				WorkgroupSize
			);

			// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
			scatterShader = m_device->createShader(overrideSource.get());
			if (!scatterShader)
				return logFail("Creation of Scatter Shader from CPU Shader source failed!");
		}

		nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
				{
					.binding = 0,
					.type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
					.createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE, // not is not the time for descriptor indexing
					.stageFlags = IGPUShader::ESS_COMPUTE,
					.count = 1,
					.samplers = nullptr // irrelevant for a buffer
				}
		};
		smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings);
		if (!dsLayout)
			return logFail("Failed to create a Descriptor Layout!\n");

		// People love Reflection but I prefer Shader Sources instead!
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

		// This time we'll have no Descriptor Sets or Layouts because our workload has a widely varying size
		// and using traditional SSBO bindings would force us to update the Descriptor Set every frame.
		// I even started writing this sample with the use of Dynamic SSBOs, however the length of the buffer range is not dynamic
		// only the offset. This means that we'd have to write the "worst case" length into the descriptor set binding.
		// Then this has a knock-on effect that we couldn't allocate closer to the end of the streaming buffer than the "worst case" size.
		smart_refctd_ptr<IGPUPipelineLayout> layout;
		smart_refctd_ptr<IGPUComputePipeline> prefixSumPipeline;
		smart_refctd_ptr<IGPUComputePipeline> scatterPipeline;
		{
			layout = m_device->createPipelineLayout({ &pcRange,1 }, smart_refctd_ptr(dsLayout));
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
		nbl::video::IDeviceMemoryAllocator::SAllocation allocation[3] = {};
		smart_refctd_ptr<IGPUBuffer> buffers[3];
		smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;
		{
			auto build_buffer = [this](
				smart_refctd_ptr<ILogicalDevice> m_device,
				nbl::video::IDeviceMemoryAllocator::SAllocation *allocation,
				smart_refctd_ptr<IGPUBuffer>& buffer,
				size_t buffer_size,
				const char *label) {
				IGPUBuffer::SCreationParams params;
				params.size = buffer_size;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				buffer = m_device->createBuffer(std::move(params));
				if (!buffer)
					return logFail("Failed to create GPU buffer of size %d!\n", buffer_size);

				buffer->setObjectDebugName(label);

				auto reqs = buffer->getMemoryReqs();
				reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

				*allocation = m_device->allocate(reqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				if (!allocation->isValid())
					return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

				assert(allocation->memory.get() == buffer->getBoundMemory().memory);
			};

			build_buffer(m_device,	allocation,		buffers[0], sizeof(uint32_t) * element_count,	"Input Buffer");
			build_buffer(m_device,	allocation + 1,	buffers[1], sizeof(uint32_t) * bucket_count,	"Scratch Buffer");
			build_buffer(m_device,	allocation + 2,	buffers[2], sizeof(uint32_t) * element_count,	"Output Buffer");

			smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

			// note how the pool will go out of scope but thanks for backreferences in each object to its parent/dependency it will be kept alive for as long as all the Sets it allocated
			ds = pool->createDescriptorSet(std::move(dsLayout));
			// we still use Vulkan 1.0 descriptor update style, could move to Update Templates but Descriptor Buffer ubiquity seems just around the corner
			{
				IGPUDescriptorSet::SDescriptorInfo info[1];
				info[0].desc = buffers[1]; // bad API, too late to change, should just take raw-pointers since not consumed
				info[0].info.buffer = { .offset = 0,.size = sizeof(uint32_t) * bucket_count };
				IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
					{.dstSet = ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
				};
				m_device->updateDescriptorSets(writes, {});
			}
		}
		uint64_t buffer_device_address[] = {
			buffers[0]->getDeviceAddress(),
			buffers[2]->getDeviceAddress()
		};

		void* mapped_memory[3] = {
			allocation[0].memory->map({0ull,allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[1].memory->map({0ull,allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[2].memory->map({0ull,allocation[2].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
		};
		if (!mapped_memory[0] || !mapped_memory[1] || !mapped_memory[2])
			return logFail("Failed to map the Device Memory!\n");

		// Generate random data
		constexpr uint32_t minimum = 0;
		const uint32_t range = bucket_count;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);
		auto bufferData = new uint32_t[element_count];
		for (uint32_t i = 0; i < element_count; i++) {
			bufferData[i] = minimum + g() % range;
		}

		std::string outBuffer;
		for (auto i = 0; i < element_count; i++) {
			outBuffer.append(std::to_string(bufferData[i]));
			outBuffer.append(" ");
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(element_count));
		outBuffer.append("\n");
		m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		memcpy(mapped_memory[0], bufferData, sizeof(uint32_t) * element_count);

		auto pc = PushConstantData{
			.inputAddress = buffer_device_address[0],
			.outputAddress = buffer_device_address[1],
			.dataElementCount = element_count,
			.minimum = minimum,
			.elementsPerWT = elements_per_thread
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
		cmdBuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, layout.get(), 0, 1, &ds.get());
		cmdBuf->pushConstants(layout.get(), IShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
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

			queue->startCapture();
			queue->submit(submit_infos);
			queue->endCapture();
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
		cmdBuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, layout.get(), 0, 1, &ds.get());
		cmdBuf->pushConstants(layout.get(), IShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
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

			queue->startCapture();
			queue->submit(submit_infos);
			queue->endCapture();
		}

		const ISemaphore::SWaitInfo wait_infos2[] = {{
				.semaphore = progress2.get(),
				.value = timeline2
			} };
		m_device->blockForSemaphores(wait_infos2);

		const ILogicalDevice::MappedMemoryRange memory_range[2] = {
			ILogicalDevice::MappedMemoryRange(allocation[0].memory.get(), 0ull, allocation[0].memory->getAllocationSize()),
			ILogicalDevice::MappedMemoryRange(allocation[1].memory.get(), 0ull, allocation[1].memory->getAllocationSize())
		};

		if (!allocation[0].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[0]);
		if (!allocation[1].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[1]);
		
		auto buffData = reinterpret_cast<const uint32_t*>(allocation[1].memory->getMappedPointer());
		assert(allocation[1].offset == 0); // simpler than writing out all the pointer arithmetic

		outBuffer.clear();
		uint32_t count = 0;
		int c = 0;
		for (auto i = 0; i < bucket_count; i++) {
			outBuffer.append(std::to_string(buffData[i]));
			outBuffer.append(" ");
			count += buffData[i];
			if (i > 0 && buffData[i] > buffData[i-1])
				c++;
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(bucket_count));
		outBuffer.append("\n");
		outBuffer.append("True Count: ");
		outBuffer.append(std::to_string(c));
		outBuffer.append("\n");
		outBuffer.append("Sum: ");
		outBuffer.append(std::to_string(count));
		outBuffer.append("\n");

		m_logger->log("Scratch buffer is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		buffData = reinterpret_cast<const uint32_t*>(allocation[2].memory->getMappedPointer());
		assert(allocation[2].offset == 0); // simpler than writing out all the pointer arithmetic

		outBuffer.clear();
		for (auto i = 0; i < element_count; i++) {
			outBuffer.append(std::to_string(buffData[i]));
			outBuffer.append(" ");
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(element_count));
		outBuffer.append("\n");
		m_logger->log("Your ordered array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		allocation[0].memory->unmap();
		allocation[1].memory->unmap();

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