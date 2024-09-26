#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

class BitonicSort final : public application_templates::MonoDeviceApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
	BitonicSort(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		//auto limits = m_physicalDevice->getLimits();
		//constexpr uint32_t WorkgroupSize = 256;
		constexpr uint32_t n = 1024;
		uint32_t max_workgroup_size = 1024;
		uint32_t WorkgroupSize = 256;


		if (n < max_workgroup_size * 2) {
			WorkgroupSize = n / 2;
		}
		else {
			WorkgroupSize = max_workgroup_size;
		}


		smart_refctd_ptr<IGPUShader> bitonicShader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/bitonic_sort.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			assert(source);

			auto overrideSource = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define WorkgroupSize %d\n",
				WorkgroupSize
			);
			overrideSource->setShaderStage(nbl::asset::IShader::E_SHADER_STAGE::ESS_COMPUTE);
			bitonicShader = m_device->createShader(overrideSource.get());
			if (!bitonicShader)
				return logFail("Creation of Bitonic Sort Bitonic Shader from CPU Shader source failed!");
		}


		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE, .offset = 0, .size = sizeof(BitonicPushData) };

		smart_refctd_ptr<IGPUPipelineLayout> layout;
		smart_refctd_ptr<IGPUComputePipeline> bitonicShaderPipeline;

		{
			layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = layout.get();
			params.shader.shader = bitonicShader.get();
			params.shader.entryPoint = "main";
			params.shader.entries = nullptr;
			params.shader.requireFullSubgroups = true;
			params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(5);
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &bitonicShaderPipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		// Allocate memory
		nbl::video::IDeviceMemoryAllocator::SAllocation allocation[4] = {};
		smart_refctd_ptr<IGPUBuffer> buffers[4];
		{
			auto build_buffer = [this](
				smart_refctd_ptr<ILogicalDevice> m_device,
				nbl::video::IDeviceMemoryAllocator::SAllocation* allocation,
				smart_refctd_ptr<IGPUBuffer>& buffer,
				size_t buffer_size,
				const char* label) {
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

			build_buffer(m_device, allocation, buffers[0], sizeof(uint32_t) * n, "Input Key Buffer");
			build_buffer(m_device, allocation + 1, buffers[1], sizeof(uint32_t) * n, "Input Value Buffer");
			build_buffer(m_device, allocation + 2, buffers[2], sizeof(uint32_t) * n, "Output Key Buffer");
			build_buffer(m_device, allocation + 3, buffers[3], sizeof(uint32_t) * n, "Output Value Buffer");
		}
		uint64_t buffer_device_address[] = {
			buffers[0]->getDeviceAddress(),
			buffers[1]->getDeviceAddress(),
			buffers[2]->getDeviceAddress(),
			buffers[3]->getDeviceAddress(),
		};

		void* mapped_memory[] = {
			allocation[0].memory->map({0ull,allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[1].memory->map({0ull,allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[2].memory->map({0ull,allocation[2].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[3].memory->map({0ull,allocation[3].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
		};
		if (!mapped_memory[0] || !mapped_memory[1] || !mapped_memory[2] || !mapped_memory[3])
			return logFail("Failed to map the Device Memory!\n");

		//Generating data
		const uint32_t min = 0;
		const uint32_t range = n;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);

		auto bufferData = new uint32_t[2][n];
		for (uint32_t i = 0; i < n; i++)
		{
			bufferData[0][i] = min + g() % range;
		}
		memcpy(mapped_memory[0], bufferData[0], sizeof(uint32_t) * n);


		for (uint32_t i = 0; i < n; i++) {
			bufferData[1][i] = g() % std::numeric_limits<uint32_t>::max();
		}

		memcpy(mapped_memory[1], bufferData[1], sizeof(uint32_t) * n);

		std::string outBuffer;
		for (auto i = 0; i < n; i++) {
			outBuffer.append("{");
			outBuffer.append(std::to_string(bufferData[0][i]));
			outBuffer.append(", ");
			outBuffer.append(std::to_string(bufferData[1][i]));
			outBuffer.append("} ");
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(n));
		outBuffer.append("\n");
		m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		auto pc = BitonicPushData{
			.inputKeyAddress = buffer_device_address[0],
			.inputValueAddress = buffer_device_address[1],
			.outputKeyAddress = buffer_device_address[2],
			.outputValueAddress = buffer_device_address[3],
			.dataElementCount = n,
		};

		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		constexpr uint64_t started_value = 0;
		uint64_t timeline = started_value;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(started_value);

		cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdBuf->beginDebugMarker("Bitonic Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		cmdBuf->bindComputePipeline(bitonicShaderPipeline.get());
		cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		cmdBuf->dispatch(WorkgroupSize, 1, 1);
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


		const ILogicalDevice::MappedMemoryRange memory_range[] = {
		ILogicalDevice::MappedMemoryRange(allocation[0].memory.get(), 0ull, allocation[0].memory->getAllocationSize()),
		ILogicalDevice::MappedMemoryRange(allocation[1].memory.get(), 0ull, allocation[1].memory->getAllocationSize()),
		ILogicalDevice::MappedMemoryRange(allocation[2].memory.get(), 0ull, allocation[2].memory->getAllocationSize()),
		ILogicalDevice::MappedMemoryRange(allocation[3].memory.get(), 0ull, allocation[3].memory->getAllocationSize()),
		};

		if (!allocation[0].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[0]);
		if (!allocation[1].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[1]);
		if (!allocation[2].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[2]);
		if (!allocation[3].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[3]);

		const uint32_t* buffData[] = {
			reinterpret_cast<const uint32_t*>(allocation[2].memory->getMappedPointer()),
			reinterpret_cast<const uint32_t*>(allocation[3].memory->getMappedPointer()),
		};

		assert(allocation[2].offset == 0); // simpler than writing out all the pointer arithmetic
		assert(allocation[3].offset == 0); // simpler than writing out all the pointer arithmetic

		outBuffer.clear();
		for (auto i = 0; i < range; i++) {
			outBuffer.append(std::to_string(buffData[0][i]));
			outBuffer.append(" ");
		}
		outBuffer.append("\n");

		m_logger->log("Your output array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		allocation[0].memory->unmap();
		allocation[1].memory->unmap();
		allocation[2].memory->unmap();
		allocation[3].memory->unmap();

		m_device->waitIdle();

		delete[] bufferData;
		return true;
	}	bool keepRunning() override { return false; }

	// Finally the first actual work-loop
	void workLoopBody() override {}

	bool onAppTerminated() override { return true; }
};



NBL_MAIN_FUNC(BitonicSort)