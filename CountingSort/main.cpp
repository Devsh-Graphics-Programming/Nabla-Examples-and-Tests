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

		// this time we load a shader directly from a file
		smart_refctd_ptr<IGPUShader> shader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			// lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			// The down-cast should not fail!
			assert(source);

			// this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
			shader = m_device->createShader(source.get());
			if (!shader)
				return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
		}

		// People love Reflection but I prefer Shader Sources instead!
		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::ESS_COMPUTE,.offset = 0,.size = sizeof(PushConstantData) };

		// This time we'll have no Descriptor Sets or Layouts because our workload has a widely varying size
		// and using traditional SSBO bindings would force us to update the Descriptor Set every frame.
		// I even started writing this sample with the use of Dynamic SSBOs, however the length of the buffer range is not dynamic
		// only the offset. This means that we'd have to write the "worst case" length into the descriptor set binding.
		// Then this has a knock-on effect that we couldn't allocate closer to the end of the streaming buffer than the "worst case" size.
		smart_refctd_ptr<IGPUPipelineLayout> layout;
		smart_refctd_ptr<IGPUComputePipeline> pipeline;
		{
			layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = layout.get();
			params.shader.shader = shader.get();
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		// Allocate memory
		nbl::video::IDeviceMemoryAllocator::SAllocation allocation[2] = {};
		constexpr size_t element_count = 100;
		uint64_t buffer_device_address[2];
		{
			auto build_buffer = [this](
				smart_refctd_ptr<ILogicalDevice> m_device,
				nbl::video::IDeviceMemoryAllocator::SAllocation *allocation,
				size_t buffer_size,
				uint64_t *bda,
				const char *label) {
				IGPUBuffer::SCreationParams params;
				params.size = buffer_size;
				params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_SHADER_DEVICE_ADDRESS_BIT;
				auto buffer = m_device->createBuffer(std::move(params));
				if (!buffer)
					return logFail("Failed to create GPU buffer of size %d!\n", buffer_size);

				buffer->setObjectDebugName(label);

				auto reqs = buffer->getMemoryReqs();
				reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

				*allocation = m_device->allocate(reqs, buffer.get(), IDeviceMemoryAllocation::EMAF_DEVICE_ADDRESS_BIT);
				if (!allocation->isValid())
					return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

				assert(allocation->memory.get() == buffer->getBoundMemory().memory);

				*bda = buffer->getDeviceAddress();
			};

			build_buffer(m_device, allocation, sizeof(uint32_t) * element_count, buffer_device_address, "Input Buffer");
			build_buffer(m_device, allocation + 1, sizeof(uint32_t) * element_count, buffer_device_address + 1, "Output Buffer");
		}

		void* mapped_memory[2] = {
			allocation[0].memory->map({0ull,allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[1].memory->map({0ull,allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
		};
		if (!mapped_memory[0] || !mapped_memory[1])
			return logFail("Failed to map the Device Memory!\n");

		// Generate random data
		constexpr uint32_t minimum = 0;
		constexpr uint32_t maximum = 100;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);
		uint32_t bufferData[element_count];
		for (uint32_t i = 0; i < element_count; i++) {
			bufferData[i] = minimum + g() % (maximum - minimum + 1);
		}

		std::string outBuffer;
		for (auto i = 0; i < element_count; i++) {
			outBuffer.append(std::to_string(bufferData[i]));
			outBuffer.append(" ");
		}
		outBuffer.append("\n");
		m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		memcpy(mapped_memory[0], bufferData, sizeof(uint32_t) * element_count);

		auto pc = PushConstantData{
			.inputAddress = buffer_device_address[0],
			.outputAddress = buffer_device_address[1],
			.dataElementCount = element_count,
			.minimum = minimum,
			.maximum = maximum
		};

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		// Create the Semaphore
		constexpr uint64_t started_value = 0;
		uint64_t timeline = started_value;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(started_value);

		cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdBuf->beginDebugMarker("My Compute Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		cmdBuf->bindComputePipeline(pipeline.get());
		cmdBuf->pushConstants(layout.get(), IShader::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		cmdBuf->dispatch(ceil((float)(maximum - minimum + 1) / WorkgroupSize), 1, 1);
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
					.semaphore = progress.get(),
					.value = timeline + 1,
					.stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT
				}
			};
			submit_infos[0].signalSemaphores = signals;

			queue->startCapture();
			queue->submit(submit_infos);
			queue->endCapture();

			timeline++;
		}

		const ISemaphore::SWaitInfo wait_infos[] = { {
				.semaphore = progress.get(),
				.value = timeline
			} };
		m_device->blockForSemaphores(wait_infos);

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

		uint32_t index = 0;
		uint32_t count = 0;
		for (auto i = 0; i <= maximum - minimum; i++) {
			while (buffData[i] && count < buffData[i]) {
				bufferData[index++] = i;
				count++;
			}
			count = 0;
		}

		outBuffer.clear();
		for (auto i = 0; i < element_count; i++) {
			//if (element_count % 10 == 0)
			//	outBuffer.append("\n");
			outBuffer.append(std::to_string(bufferData[i]));
			outBuffer.append(" ");
		}
		outBuffer.append("\n");
		m_logger->log("Your ordered array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		allocation[0].memory->unmap();
		allocation[1].memory->unmap();

		m_device->waitIdle();

		return true;
	}

	// Ok this time we'll actually have a work loop (maybe just for the sake of future WASM so we don't timeout a Browser Tab with an unresponsive script)
	bool keepRunning() override { return false; }

	// Finally the first actual work-loop
	void workLoopBody() override {}

	bool onAppTerminated() override { return true; }
};


NBL_MAIN_FUNC(CountingSortApp)