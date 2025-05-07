#include "nbl/application_templates/BasicMultiQueueApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace asset;
using namespace system;
using namespace video;

class ComputeScanApp final : public application_templates::BasicMultiQueueApplication, public application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
	using device_base_t = application_templates::BasicMultiQueueApplication;
	using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;

public:
	ComputeScanApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
		system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

	bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
	{
		if (!device_base_t::onAppInitialized(std::move(system)))
			return false;
		if (!asset_base_t::onAppInitialized(std::move(system)))
			return false;

		computeQueue = getComputeQueue();

		// Create (an almost) 128MB input buffer
		constexpr auto in_size = 128u << 20u;
		constexpr auto in_count = in_size / sizeof(uint32_t) - 24u;

		m_logger->log("Input element count: %d", ILogger::ELL_PERFORMANCE, in_count);

		inputData = new uint32_t[in_count];
		{
			std::random_device random_device;
			std::mt19937 generator(random_device());
			std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);
			for (auto i = 0u; i < in_count; i++)
				inputData[i] = 1u;//distribution(generator) % 128;
		}
		auto minSSBOAlign = m_physicalDevice->getLimits().minSSBOAlignment;
		constexpr auto begin = in_count / 4 + 118;
		assert(((begin * sizeof(uint32_t)) & (minSSBOAlign - 1u)) == 0u);
		constexpr auto end = in_count * 3 / 4 - 78;
		assert(((end * sizeof(uint32_t)) & (minSSBOAlign - 1u)) == 0u);
		constexpr auto elementCount = end - begin;

		// Set Semaphores to control GPU synchronization
		core::smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);
		IQueue::SSubmitInfo::SSemaphoreInfo semInfo[1] = { {
			.semaphore = semaphore.get(),
			.value = 1,
			.stageMask = PIPELINE_STAGE_FLAGS::ALL_TRANSFER_BITS
		} };

		smart_refctd_ptr<IGPUBuffer> gpuinputDataBuffer;
		{
			IGPUBuffer::SCreationParams inputDataBufferCreationParams = {};
			inputDataBufferCreationParams.size = sizeof(uint32_t) * in_count; // TODO Declare the element data type in the shader?
			inputDataBufferCreationParams.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT | IGPUBuffer::EUF_TRANSFER_SRC_BIT | IGPUBuffer::EUF_TRANSFER_DST_BIT;
			auto temp = m_utils->createFilledDeviceLocalBufferOnDedMem(
				SIntendedSubmitInfo{ .queue = getTransferUpQueue() },
				std::move(inputDataBufferCreationParams),
				inputData,
				{ semInfo, 1 }
			);

			const ISemaphore::SWaitInfo semWaitInfo[] = { {
				.semaphore = semaphore.get(),
				.value = 1
			} };
			if (m_device->blockForSemaphores(semWaitInfo) != ISemaphore::WAIT_RESULT::SUCCESS) {
				m_logger->log("Blocking for operation semaphore failed during input data buffer creation", ILogger::ELL_ERROR);
				return false;
			}
			gpuinputDataBuffer = *temp.get();
		}
		SBufferRange<IGPUBuffer> in_gpu_range = { begin * sizeof(uint32_t), elementCount * sizeof(uint32_t), gpuinputDataBuffer };

		const auto scanType = video::CScanner::EST_EXCLUSIVE;
		video::CReduce* reducer = m_utils->getDefaultReducer();
		video::CScanner* scanner = m_utils->getDefaultScanner();

		CArithmeticOps::DefaultPushConstants push_constants;
		CArithmeticOps::DispatchInfo dispatch_info;
		scanner->buildParameters(elementCount, push_constants, dispatch_info); // common for reducer and scanner

		IGPUBuffer::SCreationParams params = { push_constants.scanParams.getScratchSize(), bitflag(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT };

		auto reduce_pipeline = reducer->getDefaultPipeline(CArithmeticOps::EDT_UINT, CArithmeticOps::EO_ADD, params.size); // TODO: Update to test all operations
		auto scan_pipeline = scanner->getDefaultPipeline(scanType, CArithmeticOps::EDT_UINT, CArithmeticOps::EO_ADD, params.size); // TODO: Update to test all operations

		auto reduceDSLayout = reducer->getDefaultDescriptorSetLayout();
		auto scanDSLayout = scanner->getDefaultDescriptorSetLayout();
		IGPUDescriptorSetLayout const* dsLayouts[2] = { reduceDSLayout, scanDSLayout };
		auto dsPool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, dsLayouts);
		auto reduceDS = dsPool->createDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(reduceDSLayout));
		auto scanDS = dsPool->createDescriptorSet(core::smart_refctd_ptr<IGPUDescriptorSetLayout>(scanDSLayout));

		SBufferRange<IGPUBuffer> scratch_gpu_range = {0u, params.size, m_device->createBuffer(std::move(params)) };
		{
			auto memReqs = scratch_gpu_range.buffer->getMemoryReqs();
			memReqs.memoryTypeBits &= m_physicalDevice->getDeviceLocalMemoryTypeBits();
			auto scratchMem = m_device->allocate(memReqs, scratch_gpu_range.buffer.get());
		}
		reducer->updateDescriptorSet(m_device.get(), reduceDS.get(), in_gpu_range, scratch_gpu_range);
		scanner->updateDescriptorSet(m_device.get(), scanDS.get(), in_gpu_range, scratch_gpu_range);
		
		// Prepare Buffer Barriers
		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t reduceBarrier = {
			.barrier = {
				.dep = {
					.srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
					.srcAccessMask = ACCESS_FLAGS::HOST_WRITE_BIT,
					.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					.dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
				}
			},
			.range = in_gpu_range
		};
		const IGPUCommandBuffer::SPipelineBarrierDependencyInfo reduceInfo[1] = { {.bufBarriers = {&reduceBarrier, 1u}} };

		IGPUCommandBuffer::SPipelineBarrierDependencyInfo::buffer_barrier_t scanBarrier = {
			.barrier = {
				.dep = {
					.srcStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					.srcAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS,
					.dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
					.dstAccessMask = ACCESS_FLAGS::SHADER_READ_BITS
				}
			},
			.range = scratch_gpu_range // the scratch is the one that contains the intermediary Reduce values that we want for the scan
		};
		const IGPUCommandBuffer::SPipelineBarrierDependencyInfo scanInfo[1] = { {.bufBarriers = {&scanBarrier, 1u}} };

		{
			smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
			if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf,1 }))
			{
				logFail("Failed to create Command Buffers!\n");
				return false;
			}
		}

		video::IGPUPipelineLayout const* pipeline_layouts[2] = { reduce_pipeline->getLayout(), scan_pipeline->getLayout() };

		cmdbuf->begin(IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT); // (REVIEW): not sure about this
		cmdbuf->fillBuffer(scratch_gpu_range, 0u); // Host side only?

		cmdbuf->bindComputePipeline(reduce_pipeline);
		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, reduce_pipeline->getLayout(), 0u, 1u, &reduceDS.get());
		reducer->dispatchHelper(cmdbuf.get(), reduce_pipeline->getLayout(), push_constants, dispatch_info, reduceInfo);
		
		// Reset the workgroup enumerator buffer
		SBufferRange<IGPUBuffer> scratch_workgroupenum_range = scratch_gpu_range;
		scratch_workgroupenum_range.offset = sizeof(uint32_t);
		scratch_workgroupenum_range.size = push_constants.scanParams.getWorkgroupEnumeratorSize();
		cmdbuf->fillBuffer(scratch_workgroupenum_range, 0u);

		cmdbuf->bindComputePipeline(scan_pipeline);
		cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, scan_pipeline->getLayout(), 0u, 1u, &scanDS.get());
		scanner->dispatchHelper(cmdbuf.get(), scan_pipeline->getLayout(), push_constants, dispatch_info, scanInfo);

		// REVIEW: Maybe collapse descriptor sets since they're the same? But this way we are prepared for potential future pipeline discrepancies between Reduce and Scan ops

		cmdbuf->end();

		{
			semInfo[0].value = 2;
			semInfo[0].stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT;
			const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
				.cmdbuf = cmdbuf.get()
			} };
			
			const IQueue::SSubmitInfo infos[1] = { {
				.commandBuffers = commandBuffers,
				.signalSemaphores = semInfo
			} };

			m_api->startCapture();
			if (computeQueue->submit(infos) != IQueue::RESULT::SUCCESS) {
				m_logger->log("Submission failure", system::ILogger::ELL_ERROR);
			}
			m_api->endCapture();
		}

		// TODO: Update to support all operations
		// cpu counterpart
		auto cpu_begin = inputData + begin;
		m_logger->log("CPU scan begin", system::ILogger::ELL_PERFORMANCE);

		auto start = std::chrono::high_resolution_clock::now();
		switch (scanType)
		{
		case video::CScanner::EST_INCLUSIVE:
			std::inclusive_scan(cpu_begin, inputData + end, cpu_begin);
			break;
		case video::CScanner::EST_EXCLUSIVE:
			std::exclusive_scan(cpu_begin, inputData + end, cpu_begin, 0u);
			break;
		default:
			assert(false);
			exit(0xdeadbeefu);
			break;
		}
		auto stop = std::chrono::high_resolution_clock::now();

		m_logger->log("CPU scan end. Time taken: %d us", system::ILogger::ELL_PERFORMANCE, std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
		// wait for the gpu impl to complete
		const ISemaphore::SWaitInfo cmdbufDonePending[] = {{
			.semaphore = semaphore.get(),
			.value = 2
		}};
		if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS) {
			m_logger->log("Blocking for operation semaphore failed", ILogger::ELL_ERROR);
			return false;
		}
		
		{
			IGPUBuffer::SCreationParams params = {};
			params.size = in_gpu_range.size;
			params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;
			// (REVIEW): Check if this new download_buffer is needed or if we can directly read from the gpu_input buffer
			auto downloaded_buffer = m_device->createBuffer(std::move(params));
			auto memReqs = downloaded_buffer->getMemoryReqs();
			memReqs.memoryTypeBits &= m_physicalDevice->getDownStreamingMemoryTypeBits();
			auto queriesMem = m_device->allocate(memReqs, downloaded_buffer.get());
			{
				// (REVIEW): Maybe we can just reset the cmdbuf we already have?
				core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
				{
					auto cmdPool = m_device->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::NONE);
					cmdPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &cmdbuf, 1}, core::smart_refctd_ptr(m_logger));
				}
				cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);  // TODO: Reset Frame's CommandPool
				IGPUCommandBuffer::SBufferCopy region;
				region.srcOffset = in_gpu_range.offset;
				region.dstOffset = 0u;
				region.size = in_gpu_range.size;
				cmdbuf->copyBuffer(in_gpu_range.buffer.get(), downloaded_buffer.get(), 1u, &region);
				cmdbuf->end();

				{
					const IQueue::SSubmitInfo::SCommandBufferInfo commandBuffers[1] = { {
						.cmdbuf = cmdbuf.get()
					} };

					semInfo[0].value = 3;
					const IQueue::SSubmitInfo infos[1] = { {
						.commandBuffers = commandBuffers,
						.signalSemaphores = semInfo
					} };

					if (computeQueue->submit(infos) != IQueue::RESULT::SUCCESS) {
						m_logger->log("Download submission failure", system::ILogger::ELL_ERROR);
					}

					const ISemaphore::SWaitInfo cmdbufDonePending[] = { {
						.semaphore = semaphore.get(),
						.value = 3
					} };
					if (m_device->blockForSemaphores(cmdbufDonePending) != ISemaphore::WAIT_RESULT::SUCCESS) {
						m_logger->log("Blocking for download semaphore failed", ILogger::ELL_ERROR);
						return false;
					}
				}
			}

			auto mem = const_cast<video::IDeviceMemoryAllocation*>(downloaded_buffer->getBoundMemory().memory);
			{
				mem->map({ .offset = 0u, .length = params.size }, video::IDeviceMemoryAllocation::EMCAF_READ);
			}
			auto gpu_begin = reinterpret_cast<uint32_t*>(mem->getMappedPointer());
			for (auto i = 0u; i < elementCount; i++)
			{
				if (gpu_begin[i] != cpu_begin[i])
					_NBL_DEBUG_BREAK_IF(true);
			}
			m_logger->log("Result Comparison Test Passed", system::ILogger::ELL_PERFORMANCE);
			operationSuccess = true;
		}

		delete[] inputData;

		return true;
	}
	
	//virtual video::SPhysicalDeviceFeatures getRequiredDeviceFeatures() const override
	//{
	//	video::SPhysicalDeviceFeatures retval = {};

	//	retval.bufferDeviceAddress = true;
	//	retval.subgroupBroadcastDynamicId = true;
	//	retval.shaderSubgroupExtendedTypes = true;
	//	// TODO: actually need to implement this and set it on the pipelines
	//	retval.computeFullSubgroups = true;
	//	retval.subgroupSizeControl = true;

	//	return retval;
	//}

	virtual bool onAppTerminated() override
	{
		m_logger->log("==========Result==========", ILogger::ELL_INFO);
		m_logger->log("Operation Success: %s", ILogger::ELL_INFO, operationSuccess ?"true":"false");
		delete[] inputData;
		return true;
	}

	// the unit test is carried out on init
	void workLoopBody() override {}

	bool keepRunning() override { return false; }

private:
	void logTestOutcome(bool passed, uint32_t workgroupSize)
	{
		if (passed)
			m_logger->log("Passed test #%u", ILogger::ELL_INFO, workgroupSize);
		else
		{
			m_logger->log("Failed test #%u", ILogger::ELL_ERROR, workgroupSize);
		}
	}

	IQueue* computeQueue;
	uint32_t* inputData = nullptr;
	smart_refctd_ptr<IGPUDescriptorSet> descriptorSet;
	smart_refctd_ptr<IGPUPipelineLayout> pipelineLayout;
	smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
	smart_refctd_ptr<ICPUBuffer> resultsBuffer;

	bool operationSuccess = false;
};

NBL_MAIN_FUNC(ComputeScanApp)
