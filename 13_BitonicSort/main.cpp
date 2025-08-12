#include "nbl/examples/examples.hpp"
#include <cmath>
#include <chrono>

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"

class BitonicSortApp final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
	using device_base_t = application_templates::MonoDeviceApplication;
	using asset_base_t = BuiltinResourcesApplication;

public:
	BitonicSortApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		auto limits = m_physicalDevice->getLimits();
		const uint32_t max_shared_memory_size = limits.maxComputeSharedMemorySize;
		const uint32_t max_workgroup_size = limits.maxComputeWorkGroupInvocations; // Get actual GPU limit
		const uint32_t bytes_per_elements = sizeof(uint32_t) * 2; // 2 uint32_t per element (key and value)
		const uint32_t max_element_in_shared_memory = max_shared_memory_size / bytes_per_elements;

		// For bitonic sort: element count MUST be power of 2
		uint32_t element_count = core::roundDownToPoT(max_element_in_shared_memory);
		
		const uint32_t log2_element_count = static_cast<uint32_t>(log2(element_count));

		m_logger->log("GPU Limits:", ILogger::ELL_INFO);
		m_logger->log("  Max Workgroup Size: " + std::to_string(max_workgroup_size), ILogger::ELL_INFO);
		m_logger->log("  Max Shared Memory: " + std::to_string(max_shared_memory_size) + " bytes", ILogger::ELL_INFO);
		m_logger->log("  Max elements in shared memory: " + std::to_string(max_element_in_shared_memory), ILogger::ELL_INFO);
		m_logger->log("  Using element count (power of 2): " + std::to_string(element_count), ILogger::ELL_INFO);

		auto prepShader = [&](const core::string& path) -> smart_refctd_ptr<IShader>
			{
				IAssetLoader::SAssetLoadParams lp = {};
				lp.logger = m_logger.get();
				lp.workingDirectory = "";
				auto assetBundle = m_assetMgr->getAsset(path, lp);
				const auto assets = assetBundle.getContents();
				if (assets.empty())
				{
					logFail("Could not load shader!");
					return nullptr;
				}

				auto source = IAsset::castDown<IShader>(assets[0]);
				assert(source);

				auto overrideSource = CHLSLCompiler::createOverridenCopy(
					source.get(), "#define ElementCount %d\n#define Log2ElementCount %d\n#define WorkgroupSize %d\n",
					element_count, log2_element_count, max_workgroup_size
				);

				auto shader = m_device->compileShader({ overrideSource.get() });
				if (!shader)
				{
					logFail("Creation of Bitonic Sort Shader from CPU Shader source failed!");
					return nullptr;
				}
				return shader;
			};

		auto bitonicSortShader = prepShader("app_resources/bitonic_sort_shader.comp.hlsl");

		if (!bitonicSortShader)
			return logFail("Failed to compile bitonic sort shader!");


		const nbl::asset::SPushConstantRange pcRange = { .stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE,.offset = 0,.size = sizeof(BitonicPushData) };

		smart_refctd_ptr<IGPUPipelineLayout> layout;
		smart_refctd_ptr<IGPUComputePipeline> bitonicSortPipeline;
		{
			layout = m_device->createPipelineLayout({ &pcRange,1 });
			IGPUComputePipeline::SCreationParams params = {};
			params.layout = layout.get();
			params.shader.shader = bitonicSortShader.get();
			params.shader.entryPoint = "main";
			params.shader.entries = nullptr;
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &bitonicSortPipeline))
				return logFail("Failed to create compute pipeline!\n");
		}

		nbl::video::IDeviceMemoryAllocator::SAllocation allocation[4] = {};
		smart_refctd_ptr<IGPUBuffer> buffers[4];

		auto build_buffer = [this](
			smart_refctd_ptr<ILogicalDevice> m_device,
			nbl::video::IDeviceMemoryAllocator::SAllocation* allocation,
			smart_refctd_ptr<IGPUBuffer>& buffer,
			size_t buffer_size,
			const char* label
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

		build_buffer(m_device, allocation, buffers[0], sizeof(uint32_t) * element_count, "Input Key Buffer");
		build_buffer(m_device, allocation + 1, buffers[1], sizeof(uint32_t) * element_count, "Input Value Buffer");
		build_buffer(m_device, allocation + 2, buffers[2], sizeof(uint32_t) * element_count, "Output Key Buffer");
		build_buffer(m_device, allocation + 3, buffers[3], sizeof(uint32_t) * element_count, "Output Value Buffer");

		uint64_t buffer_device_address[] = {
			buffers[0]->getDeviceAddress(),
			buffers[1]->getDeviceAddress(),
			buffers[2]->getDeviceAddress(),
			buffers[3]->getDeviceAddress()
		};

	
		void* mapped_memory[] = {
			allocation[0].memory->map({0ull,allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[1].memory->map({0ull,allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[2].memory->map({0ull,allocation[2].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[3].memory->map({0ull,allocation[3].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ)
		};
		if (!mapped_memory[0] || !mapped_memory[1] || !mapped_memory[2] || !mapped_memory[3])
			return logFail("Failed to map the Device Memory!\n");

		// Generate random data
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);

		auto bufferData = new uint32_t * [2];
		for (int i = 0; i < 2; ++i) {
			bufferData[i] = new uint32_t[element_count];
		}
		for (uint32_t i = 0; i < element_count; i++) {
			bufferData[0][i] = g() % 10000; 
		}

		memcpy(mapped_memory[0], bufferData[0], sizeof(uint32_t) * element_count);

		for (uint32_t i = 0; i < element_count; i++) {
			bufferData[1][i] = i; // Values are indices for verification
		}

		memcpy(mapped_memory[1], bufferData[1], sizeof(uint32_t) * element_count);

		std::string outBuffer;

		outBuffer.append("ALL ELEMENTS: ");
		for (auto i = 0; i < element_count; i++) {
			outBuffer.append("{");
			outBuffer.append(std::to_string(bufferData[0][i]));
			outBuffer.append(", ");
			outBuffer.append(std::to_string(bufferData[1][i]));
			outBuffer.append("} ");

			// Add newline every 20 elements for readability
			if ((i + 1) % 20 == 0) {
				outBuffer.append("\n");
			}
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(element_count));
		outBuffer.append("\n");
		m_logger->log("Your input array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);


		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdBuf))
				return logFail("Failed to create Command Buffers!\n");
		}

		constexpr uint64_t started_value = 0;
		uint64_t timeline = started_value;
		smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(started_value);

		auto pc = BitonicPushData{
			.inputKeyAddress = buffer_device_address[0],
			.inputValueAddress = buffer_device_address[1],
			.outputKeyAddress = buffer_device_address[2],
			.outputValueAddress = buffer_device_address[3],
			.dataElementCount = element_count
		};

		cmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
		cmdBuf->beginDebugMarker("Bitonic Sort Single Dispatch", core::vectorSIMDf(0, 1, 0, 1));
		cmdBuf->bindComputePipeline(bitonicSortPipeline.get());
		cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
		cmdBuf->dispatch(1, 1, 1);
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

		const ILogicalDevice::MappedMemoryRange memory_range[] = {
			ILogicalDevice::MappedMemoryRange(allocation[0].memory.get(), 0ull, allocation[0].memory->getAllocationSize()),
			ILogicalDevice::MappedMemoryRange(allocation[1].memory.get(), 0ull, allocation[1].memory->getAllocationSize()),
			ILogicalDevice::MappedMemoryRange(allocation[2].memory.get(), 0ull, allocation[2].memory->getAllocationSize()),
			ILogicalDevice::MappedMemoryRange(allocation[3].memory.get(), 0ull, allocation[3].memory->getAllocationSize())
		};

		if (!allocation[0].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[0]);
		if (!allocation[1].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[1]);
		if (!allocation[2].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[2]);
		if (!allocation[3].memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
			m_device->invalidateMappedMemoryRanges(1, &memory_range[3]);

		const uint32_t* sortedKeys = reinterpret_cast<const uint32_t*>(allocation[2].memory->getMappedPointer());
		const uint32_t* sortedValues = reinterpret_cast<const uint32_t*>(allocation[3].memory->getMappedPointer());

		assert(allocation[2].offset == 0);
		assert(allocation[3].offset == 0);

		outBuffer.clear();

		outBuffer.append("ALL SORTED ELEMENTS: ");
		for (auto i = 0; i < element_count; i++) {
			outBuffer.append("{");
			outBuffer.append(std::to_string(sortedKeys[i]));
			outBuffer.append(", ");
			outBuffer.append(std::to_string(sortedValues[i]));
			outBuffer.append("} ");

			if ((i + 1) % 20 == 0) {
				outBuffer.append("\n");
			}
		}
		outBuffer.append("\n");
		outBuffer.append("Count: ");
		outBuffer.append(std::to_string(element_count));
		outBuffer.append("\n");
		m_logger->log("Your sorted array is: \n" + outBuffer, ILogger::ELL_PERFORMANCE);

		bool is_sorted = true;
		for (uint32_t i = 1; i < element_count; i++) {
			if (sortedKeys[i] < sortedKeys[i - 1]) {
				is_sorted = false;
				break;
			}
		}
		m_logger->log(is_sorted ? "Array is correctly sorted!" : "Array is NOT sorted correctly!",
			is_sorted ? ILogger::ELL_PERFORMANCE : ILogger::ELL_ERROR);

		allocation[0].memory->unmap();
		allocation[1].memory->unmap();
		allocation[2].memory->unmap();
		allocation[3].memory->unmap();

		m_device->waitIdle();

		for (int i = 0; i < 2; ++i) {
			delete[] bufferData[i];
		}
		delete[] bufferData;

		return true;
	}

	bool keepRunning() override { return false; }
	void workLoopBody() override {}
	bool onAppTerminated() override { return true; }
};

NBL_MAIN_FUNC(BitonicSortApp)