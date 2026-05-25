#include "nbl/ext/CUDAInterop/CUDAInteropNative.h"

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

#define WARP_SIZE 32


/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/

bool check_nv_err(auto err, auto& m_cuHandler, auto& logger, auto file, auto line, std::string const& log)
{
    if (auto re = err; NVRTC_SUCCESS != re) 
    {
        const char* str = m_cuHandler->getNVRTCFunctionTable().pnvrtcGetErrorString(re); 
        logger->log("%s:%d %s\n%s\n", system::ILogger::ELL_ERROR, file, line, str, log.c_str());
        return false;
    }
    return true;
}

#define ASSERT_NV_SUCCESS(expr, log) { auto re = check_nv_err((expr), m_cuHandler, m_logger, __FILE__, __LINE__, log); assert(re); }
#define ASSERT_CUDA_SUCCESS(expr, handler) NBL_CUDA_INTEROP_ASSERT_SUCCESS((expr), (handler))


using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::examples;
using namespace nbl::application_templates;

class CUDA2VKApp : public virtual MonoDeviceApplication, BuiltinResourcesApplication
{
    using device_base_t = MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;


public:
    // Yay thanks to multiple inheritance we cannot forward ctors anymore
    CUDA2VKApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD), m_randGenerator(m_randomDevice()) {}

    smart_refctd_ptr<CCUDAHandler> m_cuHandler;
    smart_refctd_ptr<CCUDADevice> m_cuDevice;
    smart_refctd_ptr<IUtilities> m_utils;
    std::random_device m_randomDevice;
    std::mt19937 m_randGenerator;

    IQueue* queue;


    // a device filter helps you create a set of physical devices that satisfy your requirements in terms of features, limits etc.
    virtual void filterDevices(core::set<video::IPhysicalDevice*>& physicalDevices) const
    {
        device_base_t::filterDevices(physicalDevices);
        auto& cuDevices = m_cuHandler->getAvailableDevices();
        std::erase_if(physicalDevices, [&cuDevices](auto pdev) {
            return cuDevices.end() == std::find_if(cuDevices.begin(), cuDevices.end(), [pdev](auto& cuDev) { return !memcmp(pdev->getProperties().deviceUUID, cuDev.uuid.data(), 16);  });
        });
    }

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;

        m_cuHandler = CCUDAHandler::create(m_system.get(), smart_refctd_ptr<ILogger>(m_logger));
        if (!m_cuHandler) 
            return logFail("Could not create a CUDA handler!");

        if (!device_base_t::onAppInitialized(std::move(system)))
            return false;

        m_utils = IUtilities::create(core::smart_refctd_ptr(m_device), core::smart_refctd_ptr<system::ILogger>(m_logger));
        if (!m_utils)
            return logFail("Could not create IUtilities!");

        m_cuDevice = m_cuHandler->createDevice(smart_refctd_ptr_dynamic_cast<CVulkanConnection>(m_api), m_physicalDevice);
        if (!m_cuDevice) 
            return logFail("Could not create a CUDA Device!");


        queue = getComputeQueue();

        testVectorAddKernel();
        testWmmaGemB1();
        testDestruction();

        return true;
    }


    smart_refctd_ptr<IGPUBuffer> createExternalBuffer(IDeviceMemoryAllocation* mem)
    {
        IGPUBuffer::SCreationParams params = {};
        params.size = mem->getAllocationSize();
        params.usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT;
        params.externalHandleTypes = mem->getCreationParams().externalHandleTypes;
        auto buf = m_device->createBuffer(std::move(params));
        ILogicalDevice::SBindBufferMemoryInfo bindInfo = { .buffer = buf.get(), .binding = {.memory = mem } };
        m_device->bindBufferMemory(1, &bindInfo);
        return buf;
    }

    smart_refctd_ptr<IGPUBuffer> createStaging(size_t sz)
    {
        auto buf = m_device->createBuffer({ {.size = sz, .usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT} });
        auto req = buf->getMemoryReqs();
        req.memoryTypeBits &= m_device->getPhysicalDevice()->getDownStreamingMemoryTypeBits()
                            & m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits()
                            & m_device->getPhysicalDevice()->getMemoryTypeBitsFromMemoryTypeFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT);
        auto allocation = m_device->allocate(req, { buf.get() });
    
        void* mapping = allocation.memory->map(IDeviceMemoryAllocation::MemoryRange(0, req.size), IDeviceMemoryAllocation::EMCAF_READ);
        if (!mapping)
            logFail("Failed to map an staging buffer");
        memset(mapping, 0, req.size);
        return buf;
    };

    smart_refctd_ptr<ICPUBuffer> compilePtx(const char* filepath)
    {
        IAssetLoader::SAssetLoadParams lp = {};
        lp.logger = m_logger.get();
        lp.workingDirectory = ""; // virtual root
        // this time we load a shader directly from a file
        auto assetBundle = m_assetMgr->getAsset(filepath, lp);
        const auto assets = assetBundle.getContents();
        if (assets.empty())
            logFail("Could not load kernel!");

        smart_refctd_ptr<ICPUBuffer> source = IAsset::castDown<ICPUBuffer>(assets[0]);
        std::string log;
        auto [ptx, res] = m_cuHandler->compileDirectlyToPTX(std::string((const char*)source->getPointer(), source->getSize()),
            filepath, m_cuDevice->geDefaultCompileOptions(), &log, 0, 0, 0);
        ASSERT_NV_SUCCESS(res, log);

        return ptx;
    }

    std::tuple<smart_refctd_ptr<IGPUBuffer>, smart_refctd_ptr<CCUDAImportedMemory>> createSharedBuffer(uint32_t size)
    {
        IGPUBuffer::SCreationParams vkBufferParams;
        vkBufferParams.size = m_cuDevice->roundToGranularity(CU_MEM_LOCATION_TYPE_DEVICE, size);
        vkBufferParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT;
        vkBufferParams.externalHandleTypes = CCUDADevice::ExternalMemoryHandleType;
        const auto outputBuf = m_device->createBuffer(std::move(vkBufferParams));
        auto outputMemReq = outputBuf->getMemoryReqs();

        auto allocation = m_device->allocate(outputMemReq, { outputBuf.get(), IDeviceMemoryAllocation::EMAF_NONE, CCUDADevice::ExternalMemoryHandleType });
        const auto cudaOutputMemory = m_cuDevice->importExternalMemory(core::smart_refctd_ptr(allocation.memory));
        if (!cudaOutputMemory)
          logFail("Fail to import Vulkan Memory into CUDA!");

        return std::tuple(std::move(outputBuf), std::move(cudaOutputMemory));
    }

    void testVectorAddKernel()
    {
        // This function demonstrates bidirectional resource sharing between CUDA and Vulkan:
        // 
        // Shared Resources:
        // - 3 buffers: 2 input buffers + 1 output buffer for vector addition results
        // - 1 semaphore for synchronization
        //
        // Memory Allocation Patterns:
        // - Input buffers: Allocated by CUDA (CCUDADevice::createExportableMemory) → imported to Vulkan
        // - Output buffer: Allocated by Vulkan → imported to CUDA (CCUDADevice::importExternalMemory)
        //
        // Synchronization:
        // - Semaphore: Created by Vulkan → imported to CUDA
        // - Demonstrates bidirectional signaling: CUDA signals → Vulkan waits, and vice versa
        //
        // Data Flow:
        // - CUDA kernel writes to shared buffer → Vulkan reads the results

        static constexpr uint32_t GridDim[3] = { 4096,1,1 };
        static constexpr uint32_t BlockDim[3] = { 1024,1,1 };
        static constexpr size_t NumElements = GridDim[0] * BlockDim[0];
        static constexpr size_t BufferSize = sizeof(float) * NumElements;

        const auto ptx = compilePtx("app_resources/vectorAdd_kernel.cu");
        auto& cu = m_cuHandler->getCUDAFunctionTable();

        CUmodule   module;
        ASSERT_CUDA_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr), m_cuHandler);
        auto moduleCleanup = nbl::core::makeRAIIExiter([&]() {
            cu.pcuModuleUnload(module);
        });

        CUfunction kernel;
        ASSERT_CUDA_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"), m_cuHandler);

        CUstream   stream;
        ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), m_cuHandler);
        auto streamCleanup = nbl::core::makeRAIIExiter([&] {
            cu.pcuStreamDestroy_v2(stream);
        });

        // CPU memory which we fill with random numbers between [-1,1] that will be copied to corresponding cudaMemory
        std::array<smart_refctd_ptr<ICPUBuffer>, 2> cpuBufs;

        for (auto& buf : cpuBufs)
        {
            ICPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            buf = ICPUBuffer::create(std::move(params));
        }

        std::uniform_real_distribution<float32_t> dist(-RAND_MAX, RAND_MAX);
        for (auto buf_i = 0; buf_i < cpuBufs.size(); buf_i++)
        {
            for (auto elem_i = 0; elem_i < NumElements; elem_i++)
            {
                auto* data = reinterpret_cast<float*>(cpuBufs[buf_i]->getPointer());
                data[elem_i] = dist(m_randGenerator);
            }
        }

        constexpr auto InputCount = 2;
        // Create CUDA-allocated input buffers that will be exported to Vulkan
        // This demonstrates the CUDA → Vulkan memory sharing pattern 
        std::array<smart_refctd_ptr<CCUDAExportableMemory>, InputCount> cudaInputMemories = {};
        std::array<smart_refctd_ptr<IDeviceMemoryAllocation>, InputCount> vulkanMemories = {};
        std::array<smart_refctd_ptr<IGPUBuffer>, InputCount> vulkanInputBuffers = {};
        std::array<smart_refctd_ptr<IGPUBuffer>, InputCount> inputStagingBuffers = {};

        auto initInputBuffers = [&]
        {
            for (auto input_i = 0; input_i < InputCount; input_i++)
            {
              cudaInputMemories[input_i] = m_cuDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .locationType = CU_MEM_LOCATION_TYPE_DEVICE });
              vulkanMemories[input_i] = cudaInputMemories[input_i]->exportAsMemory(m_device.get(), nullptr);
              vulkanInputBuffers[input_i] = createExternalBuffer(vulkanMemories[input_i].get());
              inputStagingBuffers[input_i] = createStaging(BufferSize);
            }
        };
        initInputBuffers();

        // Create Vulkan-allocated output buffer and import to CUDA
        // This demonstrates the Vulkan → CUDA memory sharing pattern
        auto [outputBuf, cudaOutputMemory] = createSharedBuffer(BufferSize);
        
        // Create timeline semaphore for cross-API synchronization
        // Timeline values: 0=initial, 1=release vulkan output buffer ownership, 2=cuda kernel done, 3=copy done 
        static constexpr uint64_t SyncPointInitial = 0;
        static constexpr uint64_t SyncPointReleased = 1;
        static constexpr uint64_t SyncPointKernelDone = 2;
        static constexpr uint64_t SyncPointCopyDone = 3;
        ISemaphore::SCreationParams semParams;
        semParams.initialValue = SyncPointInitial;
        semParams.externalHandleTypes = CCUDADevice::ExternalSemaphoreHandleType;
        const auto semaphore = m_device->createSemaphore(std::move(semParams));
        const auto cudaSemaphore = m_cuDevice->importExternalSemaphore(core::smart_refctd_ptr(semaphore));
        if (!cudaSemaphore)
          logFail("Fail to import Vulkan Semaphore into CUDA!");
        
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, 2> cmd;
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        bool re = commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, cmd.size(), cmd.data(), smart_refctd_ptr(m_logger));
        
        const auto outputStagingBuffer = createStaging(BufferSize);

        // === Phase 1: Vulkan releases ownership to external queue (CUDA) ===
        // Signal semaphore to value=1 after ownership transfer
        {
            const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                .barrier = {
                    .dep = {
                        .srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
                        .srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .range = {
                  .offset = 0, 
                  .size = outputBuf->getSize(), 
                  .buffer = outputBuf, 
                },
            };
    
            // start recording
            bool re = true;
            re &= cmd[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            re &= cmd[0]->pipelineBarrier(EDF_NONE, {
              .bufBarriers = std::span{&bufBarrier,&bufBarrier + 1}
            });
            re &= cmd[0]->end();
    
            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), 
              .value = SyncPointReleased,
              .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            };
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[0].get() };
            const IQueue::SSubmitInfo submitInfo = {
              .commandBuffers = {&cmdInfo, &cmdInfo + 1}, 
              .signalSemaphores = {&signalInfo, &signalInfo + 1}
            };
            const auto submitRe = queue->submit({ &submitInfo, &submitInfo + 1 });
            re &= IQueue::RESULT::SUCCESS == submitRe;
            if (!re) logFail("Something went wrong readying resources for CUDA");
        }
        
        // === Phase 2: CUDA executes kernel ===
        // 1. Copy input data from CPU to CUDA device memory
        // 2. Wait for semaphore value=1 (ownership released)
        // 3. Launch vectorAdd kernel
        // 4. Signal semaphore to value=2 (kernel complete)
        {
            // Step 1
            CUdeviceptr outputBufPtr;
            cudaOutputMemory->getMappedBuffer(&outputBufPtr);
            CUdeviceptr ptrs[] = {
              cudaInputMemories[0]->getDeviceptr(),
              cudaInputMemories[1]->getDeviceptr(),
              outputBufPtr
            };
            auto numElements = &NumElements;
            void* parameters[] = { &ptrs[0], &ptrs[1], &ptrs[2], &numElements };
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[0], cpuBufs[0]->getPointer(), BufferSize, stream), m_cuHandler);
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[1], cpuBufs[1]->getPointer(), BufferSize, stream), m_cuHandler);
    
            // Step 2
            CUexternalSemaphore semaphore = cudaSemaphore->getInternalObject();
            const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = SyncPointReleased } } };
            ASSERT_CUDA_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream), m_cuHandler); // Wait for release op from vulkan

            // Step 3
            ASSERT_CUDA_SUCCESS(cu.pcuLaunchKernel(kernel, GridDim[0], GridDim[1], GridDim[2], BlockDim[0], BlockDim[1], BlockDim[2], 0, stream, parameters, nullptr), m_cuHandler);

            // Step 4
            const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = SyncPointKernelDone } } };
            ASSERT_CUDA_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream), m_cuHandler); // Signal the imported semaphore
        }

        // === Phase 3: Vulkan acquires ownership and copies results ===
        // Wait for semaphore value=2, then copy output to staging buffer
        // Signal semaphore to value=3 after copy completes
        {
            const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                .barrier = {
                    .dep = {
                        .dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
                        .dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .range = { 
                  .offset = 0,
                  .size = outputBuf->getSize(),
                  .buffer = outputBuf, 
                },
            };
            bool re = true;
            re &= cmd[1]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            re &= cmd[1]->pipelineBarrier(EDF_NONE, {.bufBarriers = std::span{&bufBarrier,&bufBarrier + 1}});
            const auto region = IGPUCommandBuffer::SBufferCopy{ 
              .srcOffset = 0,
              .dstOffset = 0,
              .size = BufferSize 
            };
            re &= cmd[1]->copyBuffer(outputBuf.get(), outputStagingBuffer.get(), 1, &region);
            for (auto input_i = 0; input_i < InputCount; input_i++)
              re &= cmd[1]->copyBuffer(vulkanInputBuffers[input_i].get(), inputStagingBuffers[input_i].get(), 1, &region);
            cmd[1]->end();
            
            const IQueue::SSubmitInfo::SSemaphoreInfo waitInfo= {
              .semaphore = semaphore.get(), 
              .value = SyncPointKernelDone,
              .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
            };
            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), 
              .value = SyncPointCopyDone,
              .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            };
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[1].get() };
            const IQueue::SSubmitInfo submitInfo = { 
                .waitSemaphores = { &waitInfo, &waitInfo + 1 },
                .commandBuffers = { &cmdInfo, &cmdInfo + 1 },  
                .signalSemaphores = { &signalInfo, &signalInfo + 1 } 
            };
            const auto submitRe = queue->submit({ &submitInfo, &submitInfo + 1 });
            re &= IQueue::RESULT::SUCCESS == submitRe;
            if (!re)
                logFail("Something went wrong copying results from CUDA");

        }

        // === Phase 4: Validate the output buffer content ===
        {
            // Make sure we are also done with the readback 
            const auto wait = std::array{
              ISemaphore::SWaitInfo{
                .semaphore = semaphore.get(), 
                .value = SyncPointCopyDone,
              }
            };
            m_device->blockForSemaphores(wait, true);

            auto* stagingMem = outputStagingBuffer->getBoundMemory().memory;
            if (!stagingMem->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            {
                ILogicalDevice::MappedMemoryRange range(stagingMem, 0, stagingMem->getAllocationSize());
                m_device->invalidateMappedMemoryRanges(1, &range);
            }

            const auto* inputs1 = reinterpret_cast<float*>(cpuBufs[0]->getPointer());
            const auto* inputs2 = reinterpret_cast<float*>(cpuBufs[1]->getPointer());

            const auto* outputs = reinterpret_cast<float*>(outputStagingBuffer->getBoundMemory().memory->getMappedPointer());

            for (auto elem_i = 0; elem_i < NumElements; elem_i++)
            {
              const auto input1 = inputs1[elem_i];
              const auto input2 = inputs2[elem_i];
              const auto output = outputs[elem_i];
              const auto expected = input1 + input2;
              const auto diff = abs(output - expected);
              if (diff > 0.01)
                m_logger->log("TestVectorAdd: Element at index %d is incorrect!", ILogger::ELL_ERROR, elem_i);
            }

            m_logger->log("TestVectorAdd Complete", ILogger::ELL_INFO);

        }

    }

    void testWmmaGemB1()
    {
        // This function demonstrates a key advantage of CUDA-Vulkan interoperability:
        // accessing CUDA-exclusive hardware features that Vulkan cannot natively support.
        //
        // WMMA (Warp Matrix Multiply-Accumulate) with b1 (1-bit) primitives leverages
        // specialized Tensor Core instructions for ultra-efficient binary matrix operations.
        // Since Vulkan lacks native support for 1-bit matrix operations, this test showcases
        // how applications can:
        // 1. Allocate and manage matrices using Vulkan's memory system
        // 2. Share those buffers with CUDA via external memory handles
        // 3. Execute CUDA-exclusive Tensor Core operations (b1 WMMA GEMM)
        // 4. Retrieve results back to Vulkan for further GPU processing or readback
        //
        // Test methodology:
        // - Matrix A (M×K): 1-bit reverse diagonal matrix (1s on anti-diagonal, 0s elsewhere)
        // - Matrix B (K×N): 1-bit random matrix
        // - Matrix C (M×N): Result stored as int32s (popcount of bitwise AND per row/col pair)
        //
        // Verification strategy:
        // Multiplying a reverse diagonal matrix by any matrix B produces a result where each
        // column of B is reversed. This makes verification trivial: C[i,j] should equal B[K-1-i, j]
        // Example with K=4:
        //   [0 0 0 1]   [b00 b01]   [b30 b31]
        //   [0 0 1 0] × [b10 b11] = [b20 b21]
        //   [0 1 0 0]   [b20 b21]   [b10 b11]
        //   [1 0 0 0]   [b30 b31]   [b00 b01]

        // b1 WMMA dimensions: M=8, N=8, K=128
        constexpr auto WmmaSize = uint32_t3{ 8, 8, 128 };
        constexpr auto TileCount = uint32_t3{ 128, 128, 8 };  // Adjust for b1 dimensions
        constexpr auto ElementCount = WmmaSize * TileCount; // M=1024, N=1024, K=1024
        constexpr auto BlockDim = uint32_t2{ 32, 1 };       // 1 warp per block
        constexpr auto GridDim = uint32_t2(
            (ElementCount.x + WmmaSize.x - 1) / WmmaSize.x,  // M tiles
            (ElementCount.y + WmmaSize.y - 1) / WmmaSize.y   // N tiles
        );
        static constexpr auto BitsPerUint32 = 32;

        const auto ptx = compilePtx("app_resources/wmmaGemm_b1_kernel.cu");
        auto& cu = m_cuHandler->getCUDAFunctionTable();

        CUmodule   module;
        ASSERT_CUDA_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr), m_cuHandler);
        auto moduleCleanup = nbl::core::makeRAIIExiter([&]() {
            cu.pcuModuleUnload(module);
        });

        CUfunction kernel;
        ASSERT_CUDA_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "b1_wmma_gemm_kernel"), m_cuHandler);

        CUstream   stream;
        ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), m_cuHandler);
        auto streamCleanup = nbl::core::makeRAIIExiter([&] {
            cu.pcuStreamDestroy_v2(stream);
        });

        // Calculate buffer sizes (bits packed into uint32_t)
        const size_t matA_size = (ElementCount.x * ElementCount.z) / BitsPerUint32 * sizeof(uint32_t); // M x K bits
        const size_t matB_size = (ElementCount.z * ElementCount.y) / BitsPerUint32 * sizeof(uint32_t); // K x N bits
        const size_t matC_size = ElementCount.x * ElementCount.y * sizeof(int32_t);         // M x N ints

        auto [vkBufferMatC, cuMemMatC] = createSharedBuffer(matC_size);
        if (!vkBufferMatC || !cuMemMatC)
        {
            logFail("Failed to create shared buffer for matrix C");
            return;
        }

        ICPUBuffer::SCreationParams cpuBufferParamsA;
        cpuBufferParamsA.size = matA_size;
        const auto cpuBufferA = ICPUBuffer::create(std::move(cpuBufferParamsA));
        const auto cpuBufferAData = reinterpret_cast<uint32_t*>(cpuBufferA->getPointer());
        const auto cpuMatA = std::span(cpuBufferAData, matA_size / sizeof(uint32_t));

        ICPUBuffer::SCreationParams cpuBufferParamsB;
        cpuBufferParamsB.size = matB_size;
        const auto cpuBufferB = ICPUBuffer::create(std::move(cpuBufferParamsB));
        const auto cpuBufferBData = reinterpret_cast<uint32_t*>(cpuBufferB->getPointer());
        const auto cpuMatB = std::span(cpuBufferBData, matB_size / sizeof(uint32_t));
        
        // Initialize with simple patterns for verification
        auto initBinaryMatrices = [&]()
        {
            // Fill cpuMatA with reverse diagonal pattern
            std::fill(cpuMatA.begin(), cpuMatA.end(), 0);
            for (int i = 0; i < ElementCount.x; i++)
            {
              auto j = ElementCount.z - 1 - i;
              auto bitIdx = i * ElementCount.z + j;
              auto wordIdx = bitIdx / BitsPerUint32;
              auto bitOffset = bitIdx % BitsPerUint32;
              cpuMatA[wordIdx] |= (1u << bitOffset);
            }
            cpuBufferA->setContentHash(cpuBufferA->computeContentHash());

            std::uniform_int_distribution<uint32_t> dist;
            // Fill cpuMatB with random bits
            for (auto& val : cpuMatB) val = dist(m_randGenerator);
            cpuBufferB->setContentHash(cpuBufferB->computeContentHash());
            
        };
        initBinaryMatrices();

        std::array inputBuffers = {cpuBufferA.get(), cpuBufferB.get()};

        CAssetConverter::SInputs inputs = {};
        std::get<CAssetConverter::SInputs::asset_span_t<ICPUBuffer>>(inputs.assets) = inputBuffers;
        std::array<CAssetConverter::patch_t<asset::ICPUBuffer>, std::size(inputBuffers)> inputBufferPatches;
        for (auto& inputPatch : inputBufferPatches)
        {
          inputPatch.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT;
          inputPatch.externalHandleTypes = CCUDADevice::ExternalMemoryHandleType;
        }
        std::get<CAssetConverter::SInputs::patch_span_t<ICPUBuffer>>(inputs.patches) = inputBufferPatches;
        smart_refctd_ptr<CAssetConverter> converter = CAssetConverter::create({ .device = m_device.get(), .optimizer = {} });
        auto reservation = converter->reserve(inputs);
        if (!reservation)
        {
            logFail("reserve failed!");
            return;
        }

        // Create transfer queue resources
        auto transferQueue = getComputeQueue();
        auto transferCmdPool = m_device->createCommandPool(
          transferQueue->getFamilyIndex(),
          IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT | IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT
        );

        // SIntendedSubmitInfo needs at least one scratch cmdbuf in RECORDING state
        smart_refctd_ptr<IGPUCommandBuffer> transferCmdBuf;
        transferCmdPool->createCommandBuffers(
          IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &transferCmdBuf, smart_refctd_ptr(m_logger)
        );
        transferCmdBuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

        auto transferScratchSemaphore = m_device->createSemaphore({ .initialValue = 0 });

        IQueue::SSubmitInfo::SCommandBufferInfo transferCmdBufInfo = {
          transferCmdBuf.get()
        };
        SIntendedSubmitInfo transferSubmitInfo;
        transferSubmitInfo.queue = transferQueue;
        transferSubmitInfo.scratchCommandBuffers = { &transferCmdBufInfo, 1 };
        transferSubmitInfo.scratchSemaphore = {
            .semaphore = transferScratchSemaphore.get(),
            .value = 0,
            .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
        };

        nbl::video::CAssetConverter::SConvertParams convertParams = {};
        convertParams.utilities = m_utils.get();
        convertParams.transfer = &transferSubmitInfo;
  
        auto future = reservation.convert(convertParams);
        if (future.copy() != IQueue::RESULT::SUCCESS)
        {
            logFail("CAssetConverter convert failed!");
            return;
        }

        auto gpuBuffers = reservation.getGPUObjects<ICPUBuffer>();
        auto gpuBufferA = gpuBuffers[0].value;
        const auto boundedMemA = gpuBufferA->getBoundMemory();
        auto cuMemMatA = m_cuDevice->importExternalMemory(core::smart_refctd_ptr<IDeviceMemoryAllocation>(boundedMemA.memory));

        auto gpuBufferB = gpuBuffers[1].value;
        const auto boundedMemB = gpuBufferB->getBoundMemory();
        auto cuMemMatB = m_cuDevice->importExternalMemory(
          core::smart_refctd_ptr<IDeviceMemoryAllocation>(boundedMemB.memory));
        
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, 2> cmd;
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, cmd.size(), cmd.data(), smart_refctd_ptr(m_logger));

        const auto outputStagingBuffer = createStaging(vkBufferMatC->getSize());

        static constexpr uint64_t SyncPointInitial = 0;
        static constexpr uint64_t SyncPointReleased = 1;
        static constexpr uint64_t SyncPointKernelDone = 2;
        static constexpr uint64_t SyncPointCopyDone = 3;
        ISemaphore::SCreationParams semParams;
        semParams.externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32;
        semParams.initialValue = SyncPointInitial;
        auto semaphore = m_device->createSemaphore(std::move(semParams));
        const auto cudaSemaphore = m_cuDevice->importExternalSemaphore(core::smart_refctd_ptr(semaphore));
        if (!cudaSemaphore)
          logFail("Fail to import Vulkan Semaphore into CUDA!");

        // Release ownership to CUDA
        {
            const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                .barrier = {
                    .dep = {
                        .srcStageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
                        .srcAccessMask = ACCESS_FLAGS::MEMORY_WRITE_BITS,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .range = { .offset = 0, .size = vkBufferMatC->getSize(), .buffer = vkBufferMatC },
            };

            cmd[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            cmd[0]->pipelineBarrier(EDF_NONE, {.bufBarriers = std::span{&bufBarrier, &bufBarrier + 1}});
            cmd[0]->end();

            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), .value = SyncPointReleased, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            };
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[0].get() };
            const IQueue::SSubmitInfo submitInfo = {
              .commandBuffers = {&cmdInfo, &cmdInfo + 1}, .signalSemaphores = {&signalInfo, &signalInfo + 1}
            };
            queue->submit({ &submitInfo, &submitInfo + 1 });
        }

        // Launch CUDA kernel
        {
            CUdeviceptr matrixAPtr, matrixBPtr, matrixCPtr;
            cuMemMatA->getMappedBuffer(&matrixAPtr, gpuBufferA->getSize(), gpuBufferA->getBoundMemory().offset);
            cuMemMatB->getMappedBuffer(&matrixBPtr, gpuBufferB->getSize(), gpuBufferB->getBoundMemory().offset);
            cuMemMatC->getMappedBuffer(&matrixCPtr);

            void* parameters[] = { &matrixAPtr, &matrixBPtr, &matrixCPtr, 
                                   (void*)&ElementCount.x, (void*)&ElementCount.y, (void*)&ElementCount.z };

            CUexternalSemaphore semaphore_cu = cudaSemaphore->getInternalObject();
            const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = SyncPointReleased } } };
            ASSERT_CUDA_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore_cu, &waitParams, 1, stream), m_cuHandler);
            
            ASSERT_CUDA_SUCCESS(cu.pcuLaunchKernel(kernel, GridDim.x, GridDim.y, 1, BlockDim.x, BlockDim.y, 1, 0, stream, parameters, nullptr), m_cuHandler);
            
            const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = SyncPointKernelDone } } };
            ASSERT_CUDA_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore_cu, &signalParams, 1, stream), m_cuHandler);
        }

        // Acquire ownership and copy results back
        {
            const IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                .barrier = {
                    .dep = {
                        .dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
                        .dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .range = { .offset = 0, .size = vkBufferMatC->getSize(), .buffer = vkBufferMatC },
            };
            
            cmd[1]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            cmd[1]->pipelineBarrier(EDF_NONE, {.bufBarriers = std::span{&bufBarrier, &bufBarrier + 1}});
            const auto region = IGPUCommandBuffer::SBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = matC_size };
            cmd[1]->copyBuffer(vkBufferMatC.get(), outputStagingBuffer.get(), 1, &region);
            cmd[1]->end();
            
            const IQueue::SSubmitInfo::SSemaphoreInfo waitInfo = {
              .semaphore = semaphore.get(), .value = SyncPointKernelDone, .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
            };
            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), .value = SyncPointCopyDone, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
            };
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[1].get() };
            const IQueue::SSubmitInfo submitInfo = { 
                .waitSemaphores = { &waitInfo, &waitInfo + 1 },
                .commandBuffers = { &cmdInfo, &cmdInfo + 1 },  
                .signalSemaphores = { &signalInfo, &signalInfo + 1 } 
            };
            queue->submit({ &submitInfo, &submitInfo + 1 });
        }

        // Wait and verify results
        {
            const auto wait = std::array{ ISemaphore::SWaitInfo{.semaphore = semaphore.get(), .value = SyncPointCopyDone} };
            m_device->blockForSemaphores(wait, true);

            auto* stagingMem = outputStagingBuffer->getBoundMemory().memory;
        if (!stagingMem->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
        {
            ILogicalDevice::MappedMemoryRange range(stagingMem, 0, stagingMem->getAllocationSize());
            m_device->invalidateMappedMemoryRanges(1, &range);
        }

            const auto* results = reinterpret_cast<int32_t*>(stagingMem->getMappedPointer());
            
            // Verify results
            int errorCount = 0;
            for (int i = 0; i < ElementCount.x * ElementCount.y; i++) {
                const auto expected = [&]
                {
                // Since we are multiplying reverse diagonal matrix to matrixB. The result should be matrix b but each column reversed.
                // The calculation below is to get the index of cpuMatB if the column is reversed to get the expected bit.
                const auto row = i / ElementCount.y;
                const auto col = i % ElementCount.y;
                    const auto expectedCol = col;
                    const auto expectedRow = ElementCount.z - row - 1;
                    const auto expectedIdx = expectedCol * ElementCount.z + expectedRow;
                    const auto expectedWordIdx = expectedIdx / BitsPerUint32;
                    const auto expectedBitOffset = expectedIdx % BitsPerUint32;
                    return (cpuMatB[expectedWordIdx] >> expectedBitOffset) & uint32_t(1);
                }();
                const auto result = results[i];
                if (result != expected) {
                    m_logger->log("WMMA b1 test error at [%d]: GPU=%d, CPU=%d", 
                                 system::ILogger::ELL_ERROR, i, results[i], expected);
                    errorCount++;
                    constexpr int MaxErrorsToReport = 10;
                    if (errorCount == MaxErrorsToReport) break;
                }
            }
            
            if (errorCount == 0)
                m_logger->log("b1 WMMA test PASSED!", system::ILogger::ELL_INFO);
            else
                m_logger->log("b1 WMMA test FAILED with %d errors!", system::ILogger::ELL_ERROR, errorCount);
        }
    }

    void testDestruction()
    {
        
        // Tests proper resource lifetime management across CUDA-Vulkan interop by creating exportable CUDA memory,
        // copying data to it, then destroying the CUDA memory object while keeping the exported Vulkan memory alive.
        // Verifies that the exported memory remains valid and accessible after the original CUDA object is destroyed,
        // confirming correct reference counting and external memory handle semantics.

        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        constexpr auto ElementCount = 1024;
        constexpr auto BufferSize = ElementCount * sizeof(int);

        // Construct testData
        core::vector<uint32_t> testData(ElementCount);
        std::iota(testData.begin(), testData.end(), 0);

        auto& cu = m_cuHandler->getCUDAFunctionTable();

        // This vulkan memory will outlive the CUDA memory object below
        smart_refctd_ptr<IDeviceMemoryAllocation> escaped;
        {
            // Create exportable CUDA memory - this object will be destroyed at the end of this scope
            const auto cudaMemory = m_cuDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .locationType = CU_MEM_LOCATION_TYPE_DEVICE });
            if (!cudaMemory) logFail("Fail to create exportable memory!");

            // Export CUDA memory as Vulkan device memory - this reference will persist
            escaped = cudaMemory->exportAsMemory(m_device.get());
            if (!escaped) logFail("Fail to export CUDA memory!");

            // Copy testData into cudaMemory
            CUstream stream;
            ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), m_cuHandler);
            auto streamCleanup = nbl::core::makeRAIIExiter([&] {
                cu.pcuStreamDestroy_v2(stream);
            });
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(cudaMemory->getDeviceptr(), testData.data(), BufferSize, stream), m_cuHandler);
            ASSERT_CUDA_SUCCESS(cu.pcuStreamSynchronize(stream), m_cuHandler);

        }
        // CRITICAL: cudaMemory object destroyed here, but escaped memory should remain valid
        
        {
            // Re-import the exported memory - this tests if the memory survived CUDA object destruction
            auto tmpBuf = createExternalBuffer(escaped.get());
            auto staging = createStaging(BufferSize);
        
            // Setup synchronization for readback
            ISemaphore::SCreationParams semParams;
            semParams.initialValue = 0;
            const auto semaphore = m_device->createSemaphore(std::move(semParams));
            IQueue::SSubmitInfo::SSemaphoreInfo semInfo;
            semInfo.semaphore = semaphore.get();
            semInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
            semInfo.value = 1;
        
            // Copy data back from the persistent buffer to staging for verification
            smart_refctd_ptr<IGPUCommandBuffer> cmd;
            commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmd);
            cmd->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            IGPUCommandBuffer::SBufferCopy region = { .size = BufferSize };
            assert(cmd->copyBuffer(tmpBuf.get(), staging.get(), 1, &region));
            cmd->end();
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd.get() };
            const IQueue::SSubmitInfo submitInfo = {
              .commandBuffers = {&cmdInfo, &cmdInfo + 1}, 
              .signalSemaphores = {&semInfo, 1}
            };
            auto qre = queue->submit({ &submitInfo, &submitInfo + 1 });
            assert(IQueue::RESULT::SUCCESS == qre);
        
            m_device->waitIdle();
        
            auto& ptr = *(std::array<uint32_t, BufferSize>*)staging->getBoundMemory().memory->getMappedPointer();
            for (uint32_t i = 0; i < ElementCount; ++i)
            {
                if (ptr[i] != i) logFail("Test Destruction: Element %d is incorrect", i);
            }
            m_logger->log("Test Destruction complete", ILogger::ELL_INFO);
        }
    
    }

    }

    // Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
    bool keepRunning() override { return false; }

    // Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
    void workLoopBody() override {}
};

NBL_MAIN_FUNC(CUDA2VKApp)
