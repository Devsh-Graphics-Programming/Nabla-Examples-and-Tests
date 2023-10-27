#define _NBL_STATIC_LIB_

#include <nabla.h>

#include "nbl/ext/RadixSort/RadixSort.h"
#include "../common/CommonAPI.h"
#include <cstdlib>
#include <chrono>
#include <random>

using namespace nbl;
using namespace core;
using namespace video;
using namespace asset;
using namespace system;

using RadixSortClass = ext::RadixSort::RadixSort;

#define WG_SIZE 256

struct SortElement {
    uint32_t key, data;

    bool operator!=(const SortElement &other) {
      return (key != other.key) || (data != other.data);
    }
};

struct SortElementKeyAccessor {
    _NBL_STATIC_INLINE_CONSTEXPR size_t key_bit_count = 32ull;

    template<auto bit_offset, auto radix_mask>
    inline decltype(radix_mask) operator()(const SortElement &item) const {
      return static_cast<decltype(radix_mask)>(item.key >> static_cast<uint32_t>(bit_offset)) & radix_mask;
    }
};

/*template <typename T>
static T* DebugGPUBufferDownload(smart_refctd_ptr<IGPUBuffer> buffer_to_download, size_t buffer_size, IVideoDriver* driver)
{
	constexpr uint64_t timeout_ns = 15000000000u;
	const uint32_t alignment = uint32_t(sizeof(T));
	auto downloadStagingArea = driver->getDefaultDownStreamingBuffer();
	auto downBuffer = downloadStagingArea->getBuffer();

	bool success = false;

	uint32_t array_size_32 = uint32_t(buffer_size);
	uint32_t address = std::remove_pointer<decltype(downloadStagingArea)>::type::invalid_address;
	auto unallocatedSize = downloadStagingArea->multi_alloc(1u, &address, &array_size_32, &alignment);
	if (unallocatedSize)
	{
		os::Printer::log("Could not download the buffer from the GPU!", ELL_ERROR);
		exit(420);
	}

	driver->copyBuffer(buffer_to_download.get(), downBuffer, 0, address, array_size_32);

	auto downloadFence = driver->placeFence(true);
	auto result = downloadFence->waitCPU(timeout_ns, true);

	T* dataFromBuffer = nullptr;
	if (result != video::E_DRIVER_FENCE_RETVAL::EDFR_TIMEOUT_EXPIRED && result != video::E_DRIVER_FENCE_RETVAL::EDFR_FAIL)
	{
		if (downloadStagingArea->needsManualFlushOrInvalidate())
			driver->invalidateMappedMemoryRanges({ {downloadStagingArea->getBuffer()->getBoundMemory(),address,array_size_32} });

		dataFromBuffer = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(downloadStagingArea->getBufferPointer()) + address);
	}
	else
	{
		os::Printer::log("Could not download the buffer from the GPU, fence not signalled!", ELL_ERROR);
	}

	downloadStagingArea->multi_free(1u, &address, &array_size_32, nullptr);

	return dataFromBuffer;
}

template <typename T>
static void DebugCompareGPUvsCPU(smart_refctd_ptr<IGPUBuffer> gpu_buffer, T* cpu_buffer, size_t buffer_size, IVideoDriver* driver)
{
	T* downloaded_buffer = DebugGPUBufferDownload<T>(gpu_buffer, buffer_size, driver);

	size_t buffer_count = buffer_size / sizeof(T);

	if (downloaded_buffer)
	{
		for (int i = 0; i < buffer_count; ++i)
		{
			if (downloaded_buffer[i] != cpu_buffer[i])
				__debugbreak();
		}

		std::cout << "PASS" << std::endl;
	}
}

static void RadixSort(IVideoDriver* driver, const SBufferRange<IGPUBuffer>& in_gpu_range,
	core::smart_refctd_ptr<IGPUDescriptorSet>* ds_sort, const uint32_t ds_sort_count,
	IGPUComputePipeline* histogram_pipeline, IGPUComputePipeline* scatter_pipeline,
	IGPUDescriptorSet* ds_scan, IGPUComputePipeline* upsweep_pipeline, IGPUComputePipeline* downsweep_pipeline)
{
	const uint32_t total_scan_pass_count = RadixSortClass::buildParameters(in_gpu_range.size / sizeof(SortElement), WG_SIZE,
		nullptr, nullptr, nullptr, nullptr);

	RadixSortClass::Parameters_t sort_push_constants[RadixSortClass::PASS_COUNT];
	RadixSortClass::DispatchInfo_t sort_dispatch_info;

	const uint32_t upsweep_pass_count = (total_scan_pass_count / 2) + 1;
	core::vector<ScanClass::Parameters_t> scan_push_constants(upsweep_pass_count);
	core::vector<ScanClass::DispatchInfo_t> scan_dispatch_info(upsweep_pass_count);

	RadixSortClass::buildParameters(in_gpu_range.size / sizeof(SortElement), WG_SIZE, sort_push_constants, &sort_dispatch_info,
		scan_push_constants.data(), scan_dispatch_info.data());

	SBufferRange<IGPUBuffer> scratch_gpu_range = { 0 };
	scratch_gpu_range.size = in_gpu_range.size;
	scratch_gpu_range.buffer = driver->createDeviceLocalGPUBufferOnDedMem(in_gpu_range.size);

	const uint32_t histogram_count = sort_dispatch_info.wg_count[0] * RadixSortClass::BUCKETS_COUNT;
	SBufferRange<IGPUBuffer> histogram_gpu_range = { 0 };
	histogram_gpu_range.size = histogram_count * sizeof(uint32_t);
	histogram_gpu_range.buffer = driver->createDeviceLocalGPUBufferOnDedMem(histogram_gpu_range.size);

	RadixSortClass::updateDescriptorSet(ds_scan, &histogram_gpu_range, 1u, driver);
	RadixSortClass::updateDescriptorSetsPingPong(ds_sort, in_gpu_range, scratch_gpu_range, driver);

	core::smart_refctd_ptr<video::IQueryObject> time_query(driver->createElapsedTimeQuery());

	std::cout << "GPU sort begin" << std::endl;

	driver->beginQuery(time_query.get());
	RadixSortClass::sort(histogram_pipeline, upsweep_pipeline, downsweep_pipeline, scatter_pipeline, ds_scan, ds_sort, scan_push_constants.data(),
		sort_push_constants, scan_dispatch_info.data(), &sort_dispatch_info, total_scan_pass_count, upsweep_pass_count, driver);
	driver->endQuery(time_query.get());

	uint32_t time_taken;
	time_query->getQueryResult(&time_taken);

	std::cout << "GPU sort end\nTime taken: " << (double)time_taken / 1000000.0 << " ms" << std::endl;
}

int main()
{
	nbl::SIrrlichtCreationParameters params;
	params.Bits = 24;
	params.ZBufferBits = 24;
	params.DriverType = video::EDT_OPENGL;
	params.WindowSize = dimension2d<uint32_t>(512, 512);
	params.Fullscreen = false;
	params.Vsync = true;
	params.Doublebuffer = true;
	params.Stencilbuffer = false;
	params.StreamingDownloadBufferSize = 0x10000000u; // 256MB download required
	auto device = createDeviceEx(params);

	if (!device)
		return 1;

	IVideoDriver* driver = device->getVideoDriver();

	io::IFileSystem* filesystem = device->getFileSystem();
	asset::IAssetManager* am = device->getAssetManager();

	// Create (an almost) 256MB input buffer
	const size_t in_count = (1 << 25) - 23;
	const size_t in_size = in_count * sizeof(SortElement);

	std::cout << "Input element count: " << in_count << std::endl;

	std::random_device random_device;
	std::mt19937 generator(random_device());
	std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);

	SortElement* in = new SortElement[in_count];
	for (size_t i = 0u; i < in_count; ++i)
	{
		in[i].key = distribution(generator);
		in[i].data = i;
	}

	auto in_gpu = driver->createFilledDeviceLocalBufferOnDedMem(in_size, in);

	// Take (an almost) 64MB portion from it to sort
	size_t begin = (1 << 23) + 112;
	size_t end = (1 << 24) - 77;

	assert((begin & (driver->getRequiredSSBOAlignment() - 1ull)) == 0ull);

	SBufferRange<IGPUBuffer> in_gpu_range = { 0 };
	in_gpu_range.offset = begin * sizeof(SortElement);
	in_gpu_range.size = (end - begin) * sizeof(SortElement);
	in_gpu_range.buffer = in_gpu;

	auto sorter = core::make_smart_refctd_ptr<RadixSortClass>(driver, WG_SIZE);

	const uint32_t ds_sort_count = 2u;
	core::smart_refctd_ptr<video::IGPUDescriptorSet> ds_sort[ds_sort_count];
	for (uint32_t i = 0; i < ds_sort_count; ++i)
		ds_sort[i] = driver->createDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(sorter->getDefaultSortDescriptorSetLayout()));
	auto ds_scan = driver->createDescriptorSet(core::smart_refctd_ptr<const video::IGPUDescriptorSetLayout>(sorter->getDefaultScanDescriptorSetLayout()));

	auto histogram_pipeline = sorter->getDefaultHistogramPipeline();
	auto upsweep_pipeline = sorter->getDefaultUpsweepPipeline();
	auto downsweep_pipeline = sorter->getDefaultDownsweepPipeline();
	auto scatter_pipeline = sorter->getDefaultScatterPipeline();

	driver->beginScene(true);
	RadixSort(driver, in_gpu_range, ds_sort, ds_sort_count, histogram_pipeline, scatter_pipeline, ds_scan.get(), upsweep_pipeline, downsweep_pipeline);
	driver->endScene();

	{
		std::cout << "CPU sort begin" << std::endl;

		SortElement* in_data = new SortElement[in_count + (end - begin)];
		memcpy(in_data, in, sizeof(SortElement) * in_count);

		auto start = std::chrono::high_resolution_clock::now();
		SortElement* sorted_data = core::radix_sort(in_data + begin, in_data + in_count, end - begin, SortElementKeyAccessor());
		auto stop = std::chrono::high_resolution_clock::now();

		memcpy(in_data + begin, sorted_data, (end - begin) * sizeof(SortElement));

		std::cout << "CPU sort end\nTime taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;

		std::cout << "Testing: ";
		DebugCompareGPUvsCPU<SortElement>(in_gpu, in_data, in_size, driver);

		delete[] in_data;
	}

	delete[] in;

	return 0;
}*/

void test_result(SortElement *in, uint32_t count) {
  uint32_t prev = 0;
  for (int i = 0; i < count; i++) {
    if ((in + i)->key < prev) {
      std::cout << "Failed at index " << i << std::endl;
      return;
    }
    prev = (in + i)->key;
  }
}

uint32_t* debug_download(core::smart_refctd_ptr<ILogicalDevice>& logicalDevice,
                    nbl::video::IPhysicalDevice* gpuPhysicalDevice,
                    nbl::video::IGPUQueue* computeQueue,
                    SBufferRange<IGPUBuffer>& histogram_gpu_range) {
  // DOWNLOAD DEVICE BUFFER TO HOST
  IGPUBuffer::SCreationParams params = {};
  params.size = histogram_gpu_range.size;
  params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;

  auto downloaded_buffer = logicalDevice->createBuffer(params);
  auto memReqs = downloaded_buffer->getMemoryReqs();
  memReqs.memoryTypeBits &= gpuPhysicalDevice->getDownStreamingMemoryTypeBits();
  auto queriesMem = logicalDevice->allocate(memReqs, downloaded_buffer.get());
  {
    core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
    {
      auto cmdPool = logicalDevice->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::ECF_NONE);
      logicalDevice->createCommandBuffers(cmdPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
    }
    cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
    asset::SBufferCopy region;
    region.srcOffset = histogram_gpu_range.offset;
    region.dstOffset = 0u;
    region.size = histogram_gpu_range.size;
    cmdbuf->copyBuffer(histogram_gpu_range.buffer.get(), downloaded_buffer.get(), 1u, &region);
    cmdbuf->end();
    core::smart_refctd_ptr<IGPUFence> download_fence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
    IGPUQueue::SSubmitInfo submit = {};
    submit.commandBufferCount = 1u;
    submit.commandBuffers = &cmdbuf.get();
    computeQueue->submit(1u, &submit, download_fence.get());
    logicalDevice->blockForFences(1u, &download_fence.get());
  }

  auto mem = const_cast<video::IDeviceMemoryAllocation*>(downloaded_buffer->getBoundMemory());
  {
    video::IDeviceMemoryAllocation::MappedMemoryRange range;
    {
      range.memory = mem;
      range.offset = 0u;
      range.length = histogram_gpu_range.size;
    }
    logicalDevice->mapMemory(range, video::IDeviceMemoryAllocation::EMCAF_READ);
  }
  uint32_t* gpu_begin = reinterpret_cast<uint32_t*>(mem->getMappedPointer());
  return gpu_begin;
}

class RadixSortApp : public NonGraphicalApplicationBase {
public:
    smart_refctd_ptr <ISystem> system;

    NON_GRAPHICAL_APP_CONSTRUCTOR(RadixSortApp)

    void onAppInitialized_impl() override {
      CommonAPI::InitOutput initOutput;
      initOutput.system = core::smart_refctd_ptr(system);
      //CommonAPI::InitWithNoExt(initOutput, video::EAT_VULKAN, "Radix Sort Test");

      /*const CommonAPI::SFeatureRequest<nbl::video::IAPIConnection::E_FEATURE>& requiredInstanceFeatures;
          const CommonAPI::SFeatureRequest<nbl::video::IAPIConnection::E_FEATURE>& optionalInstanceFeatures;
          const CommonAPI::SFeatureRequest<nbl::video::ILogicalDevice::E_FEATURE>& requiredDeviceFeatures;
          const CommonAPI::SFeatureRequest<nbl::video::ILogicalDevice::E_FEATURE>& optionalDeviceFeatures;*/

      CommonAPI::InitWithNoExt(initOutput, video::EAT_VULKAN, "Radix Sort Test");
      system = std::move(initOutput.system);
      auto gl = std::move(initOutput.apiConnection);
      auto logger = std::move(initOutput.logger);
      auto gpuPhysicalDevice = std::move(initOutput.physicalDevice);
      auto logicalDevice = std::move(initOutput.logicalDevice);
      auto queues = std::move(initOutput.queues);
      auto renderpass = std::move(initOutput.renderpass);
      auto commandPools = std::move(initOutput.commandPools);
      auto assetManager = std::move(initOutput.assetManager);
      auto cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
      auto utilities = std::move(initOutput.utilities);

      // Create (an almost) 256MB input buffer
      //const size_t in_count = (1 << 8) - 23;
      //const size_t in_count = (1 << 25) - 23;
      const size_t in_count = (1 << 22);
      const size_t in_size = in_count * sizeof(SortElement);

      logger->log("Input element count: %d", system::ILogger::ELL_PERFORMANCE, in_count);
      SortElement *in = new SortElement[in_count];
      {
        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::uniform_int_distribution<uint32_t> distribution(0u, ~0u);
        for (size_t i = 0u; i < in_count; ++i) {
          //in[i].key = distribution(generator);
          in[i].key = i % 16;
          in[i].data = i;
        }
      }

      // Take (an almost) 64MB portion from it to sort
      /*constexpr size_t begin = (1 << 6) + 11;
      constexpr size_t end = (1 << 7) - 7;*/

      /*constexpr size_t begin = (1 << 23) + 112;
      constexpr size_t end = (1 << 24) - 77;*/

      constexpr size_t begin = 0;
      constexpr size_t end = in_count;

      assert(((begin * sizeof(SortElement)) & (gpuPhysicalDevice->getLimits().SSBOAlignment - 1u)) == 0u);
      assert(((end * sizeof(SortElement)) & (gpuPhysicalDevice->getLimits().SSBOAlignment - 1u)) == 0u);
      constexpr auto elementCount = end - begin;

      RadixSortClass::Parameters_t a_sort_push_constants[RadixSortClass::PASS_COUNT];
      RadixSortClass::DispatchInfo_t sort_dispatch_info;
      const uint32_t histogram_buckets_count = RadixSortClass::buildParameters(elementCount, WG_SIZE, a_sort_push_constants, &sort_dispatch_info);

      CScanner *scanner = utilities->getDefaultScanner();
      core::smart_refctd_ptr<CScanner> smartscanner = core::smart_refctd_ptr<CScanner>(scanner);
      CScanner::DefaultPushConstants scan_push_constants;
      CScanner::DispatchInfo scan_dispatch_info;

      scanner->buildParameters(histogram_buckets_count, scan_push_constants, scan_dispatch_info);
      auto scan_pipeline
          = core::smart_refctd_ptr<video::IGPUComputePipeline>( // if params are constants for radix sort then move this call inside the constructor
              scanner->getDefaultPipeline(video::CScanner::EST_EXCLUSIVE, CScanner::EDT_UINT, CScanner::EO_ADD));

      core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> scanDSLayout
          = core::smart_refctd_ptr<video::IGPUDescriptorSetLayout>(scanner->getDefaultDescriptorSetLayout());
      auto radixSorter = new RadixSortClass(logicalDevice.get(), WG_SIZE, elementCount, scanDSLayout, scan_pipeline);
      video::IGPUComputePipeline *histogramPipeline = radixSorter->getDefaultHistogramPipeline();
      video::IGPUComputePipeline *scanPipeline = radixSorter->getDefaultScanPipeline();
      video::IGPUComputePipeline *scatterPipeline = radixSorter->getDefaultScatterPipeline();

      // we need to descriptor sets for ping ponging between each radix pass
      // first DS uses the input buffer as input and second DS uses the scratch buffer as input
      // this avoids copying scratch buffer to input buffer at the end of each radix pass
      // TODO (Penta): Check if this should be done within the RadixSort class
      const uint32_t sortDSCount = 2;
      core::smart_refctd_ptr<video::IGPUDescriptorSet> a_pingPongSortDS[sortDSCount];
      for (auto &i: a_pingPongSortDS) {
        auto sortDSLayout = radixSorter->getDefaultSortDescriptorSetLayout();
        auto sortDSPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &sortDSLayout, &sortDSLayout + 1);
        auto sortDS = logicalDevice->createDescriptorSet(sortDSPool.get(), core::smart_refctd_ptr<IGPUDescriptorSetLayout>(sortDSLayout)); // TODO (Penta): Check if these go out of scope after method invocation...
        i = sortDS;
      }

      auto scanDSPool = logicalDevice->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, &scanDSLayout.get(), &scanDSLayout.get() + 1u);
      auto scanDS = logicalDevice->createDescriptorSet(scanDSPool.get(), scanDSLayout);

      // SORT INPUT BUFFERS
      SBufferRange <IGPUBuffer> input_gpu_range;
      {
        input_gpu_range.offset = begin * sizeof(SortElement);
        input_gpu_range.size = elementCount * sizeof(SortElement);

        IGPUBuffer::SCreationParams bufferParams = {};
        bufferParams.size = in_size;
        bufferParams.usage = core::bitflag<IGPUBuffer::E_USAGE_FLAGS>(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT |
                             IGPUBuffer::EUF_TRANSFER_SRC_BIT;
        input_gpu_range.buffer = utilities->createFilledDeviceLocalBufferOnDedMem(queues[decltype(initOutput)::EQT_TRANSFER_UP],
                                                                                  std::move(bufferParams), in);
      }

      SBufferRange <IGPUBuffer> scratch_input_gpu_range;
      {
        scratch_input_gpu_range.offset = 0u;
        scratch_input_gpu_range.size = input_gpu_range.size;

        IGPUBuffer::SCreationParams params = {};
        params.size = scratch_input_gpu_range.size;
        params.usage = core::bitflag<IGPUBuffer::E_USAGE_FLAGS>(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
        scratch_input_gpu_range.buffer = logicalDevice->createBuffer(params);
        auto memReqs = scratch_input_gpu_range.buffer->getMemoryReqs();
        memReqs.memoryTypeBits &= gpuPhysicalDevice->getDeviceLocalMemoryTypeBits();
        auto scratchMem = logicalDevice->allocate(memReqs, scratch_input_gpu_range.buffer.get());
      }

      // Update sort and scatter descriptor sets
      radixSorter->updateDescriptorSetsPingPong(a_pingPongSortDS, input_gpu_range, scratch_input_gpu_range, logicalDevice.get());

      // SORT INPUT BUFFERS - END

      // HISTOGRAM BUFFER
      SBufferRange<IGPUBuffer> histogram_gpu_range = { 0 };
      {
        histogram_gpu_range.size = histogram_buckets_count * sizeof(uint32_t);
        IGPUBuffer::SCreationParams histogram_buffer_params = {};
        histogram_buffer_params.size = histogram_gpu_range.size;
        histogram_buffer_params.usage = core::bitflag<IGPUBuffer::E_USAGE_FLAGS>(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT |
                                   IGPUBuffer::EUF_TRANSFER_SRC_BIT;
        histogram_gpu_range.buffer = logicalDevice->createBuffer(histogram_buffer_params);
        auto memReqs = histogram_gpu_range.buffer->getMemoryReqs();
        memReqs.memoryTypeBits &= gpuPhysicalDevice->getDeviceLocalMemoryTypeBits();
        auto histMem = logicalDevice->allocate(memReqs, histogram_gpu_range.buffer.get());
      }

      // SCAN SCRATCH BUFFER
      SBufferRange <IGPUBuffer> scratch_scan_gpu_range;
      {
        scratch_scan_gpu_range.offset = 0u;
        scratch_scan_gpu_range.size = histogram_gpu_range.size;

        IGPUBuffer::SCreationParams params = {};
        params.size = scratch_scan_gpu_range.size;
        params.usage = core::bitflag<IGPUBuffer::E_USAGE_FLAGS>(IGPUBuffer::EUF_STORAGE_BUFFER_BIT) | IGPUBuffer::EUF_TRANSFER_DST_BIT;
        scratch_scan_gpu_range.buffer = logicalDevice->createBuffer(params);
        auto memReqs = scratch_scan_gpu_range.buffer->getMemoryReqs();
        memReqs.memoryTypeBits &= gpuPhysicalDevice->getDeviceLocalMemoryTypeBits();
        auto scratchMem = logicalDevice->allocate(memReqs, scratch_scan_gpu_range.buffer.get());
      }

      // Update scan descriptor sets
      scanner->updateDescriptorSet(logicalDevice.get(), scanDS.get(), histogram_gpu_range, scratch_scan_gpu_range);

      // SCAN BUFFERS - END

      auto computeQueue = queues[CommonAPI::InitOutput::EQT_COMPUTE];
      {
        core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
        auto cmdPool = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];
        logicalDevice->createCommandBuffers(cmdPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);

        cmdbuf->begin(video::IGPUCommandBuffer::EU_SIMULTANEOUS_USE_BIT);

		RadixSortClass::sort(logicalDevice.get(), cmdbuf.get(), scanner,
			histogramPipeline, scanPipeline, scatterPipeline,
			a_pingPongSortDS, &scanDS,
			a_sort_push_constants, &sort_dispatch_info,
			&scan_push_constants, &scan_dispatch_info,
			input_gpu_range,
			scratch_input_gpu_range,
			histogram_gpu_range,
			scratch_scan_gpu_range,
			asset::E_PIPELINE_STAGE_FLAGS::EPSF_TOP_OF_PIPE_BIT, asset::E_PIPELINE_STAGE_FLAGS::EPSF_BOTTOM_OF_PIPE_BIT);
        cmdbuf->end();

        core::smart_refctd_ptr<IGPUFence> lastFence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
        IGPUQueue::SSubmitInfo submit = {};
        submit.commandBufferCount = 1u;
        submit.commandBuffers = &cmdbuf.get();
        computeQueue->startCapture();
        computeQueue->submit(1u, &submit, lastFence.get());
        computeQueue->endCapture();
        logicalDevice->waitForFences(1, &lastFence.get(), true, 1e10);

        uint32_t* debug = debug_download(logicalDevice, gpuPhysicalDevice, computeQueue, histogram_gpu_range);

        // DOWNLOAD DEVICE BUFFER TO HOST
        SBufferRange <IGPUBuffer>& downloaded_gpu_range = input_gpu_range;
        IGPUBuffer::SCreationParams params = {};
        params.size = downloaded_gpu_range.size;
        params.usage = IGPUBuffer::EUF_TRANSFER_DST_BIT;

        auto downloaded_buffer = logicalDevice->createBuffer(params);
        auto memReqs = downloaded_buffer->getMemoryReqs();
        memReqs.memoryTypeBits &= gpuPhysicalDevice->getDownStreamingMemoryTypeBits();
        auto queriesMem = logicalDevice->allocate(memReqs, downloaded_buffer.get());
        {
          core::smart_refctd_ptr<IGPUCommandBuffer> cmdbuf;
          {
            auto cmdPool = logicalDevice->createCommandPool(computeQueue->getFamilyIndex(), IGPUCommandPool::ECF_NONE);
            logicalDevice->createCommandBuffers(cmdPool.get(), IGPUCommandBuffer::EL_PRIMARY, 1u, &cmdbuf);
          }
          cmdbuf->begin(IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);
          asset::SBufferCopy region;
          region.srcOffset = downloaded_gpu_range.offset;
          region.dstOffset = 0u;
          region.size = downloaded_gpu_range.size;
          cmdbuf->copyBuffer(downloaded_gpu_range.buffer.get(), downloaded_buffer.get(), 1u, &region);
          cmdbuf->end();
          lastFence = logicalDevice->createFence(IGPUFence::ECF_UNSIGNALED);
          IGPUQueue::SSubmitInfo submit = {};
          submit.commandBufferCount = 1u;
          submit.commandBuffers = &cmdbuf.get();
          computeQueue->submit(1u, &submit, lastFence.get());
          logicalDevice->blockForFences(1u, &lastFence.get());
        }

        auto mem = const_cast<video::IDeviceMemoryAllocation*>(downloaded_buffer->getBoundMemory());
        {
          video::IDeviceMemoryAllocation::MappedMemoryRange range;
          {
            range.memory = mem;
            range.offset = 0u;
            range.length = downloaded_gpu_range.size;
          }
          logicalDevice->mapMemory(range, video::IDeviceMemoryAllocation::EMCAF_READ);
        }
        auto gpu_begin = reinterpret_cast<SortElement*>(mem->getMappedPointer());
        test_result(gpu_begin, elementCount);
//        logger->log("Result Comparison Test Passed", system::ILogger::ELL_PERFORMANCE);
      }

      logger->log("SUCCESS");

//      {
//        std::cout << "CPU sort begin" << std::endl;
//
//        SortElement *in_data = new SortElement[in_count + (end - begin)];
//        memcpy(in_data, in, sizeof(SortElement) * in_count);
//
//        auto start = std::chrono::high_resolution_clock::now();
//        SortElement *sorted_data = core::radix_sort(in_data + begin, in_data + in_count, end - begin, SortElementKeyAccessor());
//        auto stop = std::chrono::high_resolution_clock::now();
//
//        memcpy(in_data + begin, sorted_data, (end - begin) * sizeof(SortElement));
//
//        std::cout << "CPU sort end\nTime taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms"
//                  << std::endl;
//
//        std::cout << "Testing: ";
//        test_result(in_data + in_count, end - begin);
//        std::cout << "END" << std::endl;
////        DebugCompareGPUvsCPU<SortElement>(in_gpu, in_data, in_size, driver);
//
//        delete[] in_data;
//      }

      delete[] in;
      std::system("pause");
    }

    virtual void workLoopBody() override {

    }

    virtual bool keepRunning() override {
      return false;
    }

    void setSystem(core::smart_refctd_ptr<nbl::system::ISystem> &&s) override {
      system = std::move(s);
    }
};

NBL_COMMON_API_MAIN(RadixSortApp)
