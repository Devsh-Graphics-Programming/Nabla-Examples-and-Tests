#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"
using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#include "app_resources/common.hlsl"
#include "nbl/builtin/hlsl/bitonic_sort.hlsl"

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
		uint32_t WorkgroupSize = 1;


		if (n < max_workgroup_size * 2) {
			WorkgroupSize = n / 2;
		}
		else {
			WorkgroupSize = max_workgroup_size;
		}

		static auto pipeline =
			LeComputePipelineBuilder(encoder.getPipelineManager())
			.setShaderStage(
				LeShaderModuleBuilder(encoder.getPipelineManager())
				.setShaderStage(le::ShaderStage::eCompute)
				.setSourceFilePath("./local_resources/shaders/compute.glsl")
				.setSpecializationConstant(1, workgroup_size_x)
				.build())
			.build();

		struct Parameters {
			enum eAlgorithmVariant : uint32_t {
				eLocalBitonicMergeSortExample = 0,
				eLocalDisperse = 1,
				eBigFlip = 2,
				eBigDisperse = 3,
			};
			uint32_t          h;
			eAlgorithmVariant algorithm;
		};

		Parameters params{};

		params.h = 0;

		encoder
			.bindComputePipeline(pipeline)
			.bindArgumentBuffer(LE_ARGUMENT_NAME("SortData"), app->pixels_data->handle);

		const uint32_t workgroup_count = n / (workgroup_size_x * 2);

		auto dispatch = [&](uint32_t h) {
			params.h = h;
			encoder
				.setArgumentData(LE_ARGUMENT_NAME("Parameters"), &params, sizeof(params))
				.dispatch(workgroup_count)
				.bufferMemoryBarrier(le::PipelineStageFlags2(le::PipelineStageFlagBits2::eComputeShader),
					le::PipelineStageFlags2(le::PipelineStageFlagBits2::eComputeShader),
					le::AccessFlags2(le::AccessFlagBits2::eShaderRead),
					app->pixels_data->handle);
			};

		auto local_bitonic_merge_sort_example = [&](uint32_t h) {
			params.algorithm = Parameters::eAlgorithmVariant::eLocalBitonicMergeSortExample;
			dispatch(h);
			};

		auto big_flip = [&](uint32_t h) {
			params.algorithm = Parameters::eAlgorithmVariant::eBigFlip;
			dispatch(h);
			};

		auto local_disperse = [&](uint32_t h) {
			params.algorithm = Parameters::eAlgorithmVariant::eLocalDisperse;
			dispatch(h);
			};

		auto big_disperse = [&](uint32_t h) {
			params.algorithm = Parameters::eAlgorithmVariant::eBigDisperse;
			dispatch(h);
			};

		// Fully optimised version of bitonic merge sort.
		// Uses workgroup local memory whenever possible.

		uint32_t h = workgroup_size_x * 2;
		assert(h <= n);
		assert(h % 2 == 0);

		local_bitonic_merge_sort_example(h);
		// we must now double h, as this happens before every flip
		h *= 2;

		for (; h <= n; h *= 2) {
			big_flip(h);

			for (uint32_t hh = h / 2; hh > 1; hh /= 2) {

				if (hh <= workgroup_size_x * 2) {
					// We can fit all elements for a disperse operation into continuous shader
					// workgroup local memory, which means we can complete the rest of the
					// cascade using a single shader invocation.
					local_disperse(hh);
					break;
				}
				else {
					big_disperse(hh);
				}
			}
		}

		smart_refctd_ptr<IGPUShader> bitonicShader;
		{
			IAssetLoader::SAssetLoadParams lp = {};
			lp.logger = m_logger.get();
			lp.workingDirectory = ""; // virtual root
			auto assetBundle = m_assetMgr->getAsset("app_resources/bitonic_sort_shader.comp.hlsl", lp);
			const auto assets = assetBundle.getContents();
			if (assets.empty())
				return logFail("Could not load shader!");

			auto source = IAsset::castDown<ICPUShader>(assets[0]);
			assert(source);

			auto overrideSource = CHLSLCompiler::createOverridenCopy(
				source.get(), "#define WorkgroupSize %d\n#define ElementCount %d\n#define bitonicShaderLocal\n",
				WorkgroupSize, n
			);

			bitonicShader = m_device->createShader(overrideSource.get());
			if (!bitonicShader)
				return logFail("Creation of Bitonic Sort LocalBitonic Shader from CPU Shader source failed!");
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
			params.shader.requiredSubgroupSize = static_cast<IGPUShader::SSpecInfo::SUBGROUP_SIZE>(2);
			if (!m_device->createComputePipelines(nullptr, { &params,1 }, &bitonicShaderPipeline))
				return logFail("Failed to create compute pipeline!\n");
		}
		// Allocate memory
		nbl::video::IDeviceMemoryAllocator::SAllocation allocation[2] = {};
		smart_refctd_ptr<IGPUBuffer> buffers[2];
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

			build_buffer(m_device, allocation, buffers[0], sizeof(uint32_t) * element_count, "Input Buffer");
			build_buffer(m_device, allocation + 1, buffers[1], sizeof(uint32_t) * element_count, "Output Buffer");
		}

		void* mapped_memory[] = {
			allocation[0].memory->map({0ull, allocation[0].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
			allocation[1].memory->map({0ull, allocation[1].memory->getAllocationSize()}, IDeviceMemoryAllocation::EMCAF_READ),
		};
		if (!mapped_memory[0] || !mapped_memory[1])
			return logFail("Failed to map the Device Memory!\n");

		// Generate random data
		constexpr uint32_t minimum = 0;
		constexpr uint32_t range = 1000;
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::mt19937 g(seed);

		auto bufferData = new uint32_t[element_count];
		for (uint32_t i = 0; i < element_count; i++) {
			bufferData[i] = minimum + g() % range;
		}

		memcpy(mapped_memory[0], bufferData, sizeof(uint32_t) * element_count);

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

		auto pc = BitonicPushData{
			.inputAddress = buffers[0]->getDeviceAddress(),
			.outputAddress = buffers[1]->getDeviceAddress(),
			.elementCount = element_count,
		};


		smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdBuf;
		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(getComputeQueue()->getFamilyIndex(), IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool)
				return logFail("Failed to create Command Pool!\n");
		}

		{
			struct Parameters {
				enum eAlgorithmVariant : uint32_t {
					eLocalBitonicMergeSortExample = 0,
					eLocalDisperse = 1,
					eBigFlip = 2,
					eBigDisperse = 3,
				};
				uint32_t          h;
				eAlgorithmVariant algorithm;
			};
			Parameters parameters{};
			parameters.h = 0;


			auto dispatch = [&](uint32_t h) {
				parameters.h = h;
				cmdBuf->begin({ IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT });
				cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);
				cmdBuf->dispatch((element_count + WorkgroupSize - 1) / WorkgroupSize, 1, 1);
				cmdBuf->end();

				};


			auto local_bitonic_merge_sort_example = [&](uint32_t h) {
				parameters.algorithm = Parameters::eAlgorithmVariant::eLocalBitonicMergeSortExample;
				dispatch(h);
				};

			auto big_flip = [&](uint32_t h) {
				parameters.algorithm = Parameters::eAlgorithmVariant::eBigFlip;
				dispatch(h);
				};

			auto local_disperse = [&](uint32_t h) {
				parameters.algorithm = Parameters::eAlgorithmVariant::eLocalDisperse;
				dispatch(h);
				};

			auto big_disperse = [&](uint32_t h) {
				parameters.algorithm = Parameters::eAlgorithmVariant::eBigDisperse;
				dispatch(h);
				};
		}



		cmdBuf->begin({ IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT });
		cmdBuf->bindComputePipeline(bitonicPipeline.get());

		cmdBuf->pushConstants(layout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0u, sizeof(pc), &pc);

		cmdBuf->dispatch((element_count + WorkgroupSize - 1) / WorkgroupSize, 1, 1);
		cmdBuf->end();

		// Submit the command buffer
		{
			IGPUQueue::SSubmitInfo submit_infos[1] = {};
			submit_infos[0].commandBufferCount = 1u;
			submit_infos[0].pCommandBuffers = &cmdBuf.get();
			submit_infos[0].waitSemaphoreCount = 0u;
			submit_infos[0].signalSemaphoreCount = 0u;
			submit_infos[0].pWaitSemaphores = nullptr;
			submit_infos[0].pSignalSemaphores = nullptr;

			getComputeQueue()->submit(1u, submit_infos, nullptr);
			getComputeQueue()->waitIdle();
		}

		// Read back data
		memcpy(bufferData, mapped_memory[1], sizeof(uint32_t) * element_count);

		std::string sortedBuffer;
		for (auto i = 0; i < element_count; i++) {
			sortedBuffer.append(std::to_string(bufferData[i]));
			sortedBuffer.append(" ");
		}
		sortedBuffer.append("\n");
		sortedBuffer.append("Count: ");
		sortedBuffer.append(std::to_string(element_count));
		sortedBuffer.append("\n");
		m_logger->log("Your sorted array is: \n" + sortedBuffer, ILogger::ELL_PERFORMANCE);

		// Cleanup
		delete[] bufferData;
		allocation[0].memory->unmap();
		allocation[1].memory->unmap();

		return true;
	}	bool keepRunning() override { return false; }

	// Finally the first actual work-loop
	void workLoopBody() override {}

	bool onAppTerminated() override { return true; }
};



NBL_MAIN_FUNC(BitonicSort)