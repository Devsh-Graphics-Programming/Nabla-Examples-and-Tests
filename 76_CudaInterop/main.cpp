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
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    smart_refctd_ptr<CCUDAHandler> m_cuHandler;
    smart_refctd_ptr<CCUDADevice> m_cuDevice;
    smart_refctd_ptr<IUtilities> m_utils;

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

        testWmmaGemB1();
        // testVectorAddKernel();
        // testDestruction();
        // testLargeAllocations();

        return true;
    }

    smart_refctd_ptr<IGPUBuffer> createExternalBuffer(IDeviceMemoryAllocation* mem)
    {
        IGPUBuffer::SCreationParams params = {};
        params.size = mem->getAllocationSize();
        params.usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT;
        params.externalHandleTypes = mem->getCreationParams().externalHandleType;
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
        auto allocation = m_device->allocate(req, buf.get());
    
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
        vkBufferParams.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE;
        const auto outputBuf = m_device->createBuffer(std::move(vkBufferParams));
        auto outputMemReq = outputBuf->getMemoryReqs();

        auto allocation = m_device->allocate(outputMemReq, outputBuf.get(), IDeviceMemoryAllocation::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE);
        const auto cudaOutputMemory = m_cuDevice->importExternalMemory(core::smart_refctd_ptr(allocation.memory));
        if (!cudaOutputMemory)
          logFail("Fail to import Vulkan Memory into CUDA!");

        return std::tuple(std::move(outputBuf), std::move(cudaOutputMemory));
    }

    void testVectorAddKernel()
    {
        static constexpr uint32_t GridDim[3] = { 4096,1,1 };
        static constexpr uint32_t BlockDim[3] = { 1,1,1 };
        static constexpr size_t NumElements = GridDim[0] * BlockDim[0];
        static constexpr size_t BufferSize = sizeof(float) * NumElements;

        const auto ptx = compilePtx("app_resources/vectorAdd_kernel.cu");
        auto& cu = m_cuHandler->getCUDAFunctionTable();

        CUmodule   module;
        CUfunction kernel;
        CUstream   stream;

        ASSERT_CUDA_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), m_cuHandler);

        // CPU memory which we fill with random numbers between [-1,1] that will be copied to corresponding cudaMemory
        std::array<smart_refctd_ptr<ICPUBuffer>, 2> cpuBufs;

        for (auto& buf : cpuBufs)
        {
            ICPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            buf = ICPUBuffer::create(std::move(params));
        }

        for (auto buf_i = 0; buf_i < cpuBufs.size(); buf_i++)
            for (auto elem_i = 0; elem_i < NumElements; elem_i++)
                reinterpret_cast<float*>(cpuBufs[buf_i]->getPointer())[elem_i] = rand() / float(RAND_MAX);

        constexpr auto InputCount = 2;
        // // CUDA resources that we input to the kernel 'vectorAdd_kernel.cu'
        // // Kernel writes to cudaInputMemories[2] which we later use to export and read on nabla side
        std::array<smart_refctd_ptr<CCUDAExportableMemory>, InputCount> cudaInputMemories = {};
        std::array<smart_refctd_ptr<IDeviceMemoryAllocation>, InputCount> vulkanMemories = {};
        std::array<smart_refctd_ptr<IGPUBuffer>, InputCount> vulkanInputBuffers = {};
        std::array<smart_refctd_ptr<IGPUBuffer>, InputCount> inputStagingBuffers = {};

        for (auto input_i = 0; input_i < InputCount; input_i++)
        {
          // create and allocate CUmem with CUDA and slap it inside a simple IReferenceCounted wrapper
          cudaInputMemories[input_i] = m_cuDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .locationType = CU_MEM_LOCATION_TYPE_DEVICE });
          vulkanMemories[input_i] = cudaInputMemories[input_i]->exportAsMemory(m_device.get(), nullptr);
          vulkanInputBuffers[input_i] = createExternalBuffer(vulkanMemories[input_i].get());
          inputStagingBuffers[input_i] = createStaging(BufferSize);
        }

        auto [outputBuf, cudaOutputMemory] = createSharedBuffer(BufferSize);
        
        ISemaphore::SCreationParams semParams;
        semParams.externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32;
        auto semaphore = m_device->createSemaphore(0, std::move(semParams));
        const auto cudaSemaphore = m_cuDevice->importExternalSemaphore(core::smart_refctd_ptr(semaphore));
        if (!cudaSemaphore)
          logFail("Fail to import Vulkan Semaphore into CUDA!");
        
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, 2> cmd;
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        bool re = commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, cmd.size(), cmd.data(), smart_refctd_ptr(m_logger));
        
        const auto outputStagingBuffer = createStaging(BufferSize);

        // First we record a release ownership transfer to let vulkan know that resources are going to be used in an external API
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
              .value = 1,
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
        
        // Launch kernel
        {
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
    
            CUexternalSemaphore semaphore = cudaSemaphore->getInternalObject();
            const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = 1 } } };
            ASSERT_CUDA_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream), m_cuHandler); // Wait for release op from vulkan
            ASSERT_CUDA_SUCCESS(cu.pcuLaunchKernel(kernel, GridDim[0], GridDim[1], GridDim[2], BlockDim[0], BlockDim[1], BlockDim[2], 0, stream, parameters, nullptr), m_cuHandler);
            const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = 2 } } };
            ASSERT_CUDA_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream), m_cuHandler); // Signal the imported semaphore
        }
        
        // After the cuda kernel has signalled our exported vk semaphore, we will download the results through the buffer imported from CUDA
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
              .value = 2,
              .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
            };
            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), 
              .value = 3,
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

        struct CallbackContext
        {
            core::smart_refctd_ptr<ISemaphore> semaphore;
            std::array<core::smart_refctd_ptr<ICPUBuffer>, InputCount> cpuBuffers;
            std::array<core::smart_refctd_ptr<IGPUBuffer>, InputCount> inputStagingBuffers;
            core::smart_refctd_ptr<IGPUBuffer> outputStagingBuffer;
            core::smart_refctd_ptr<video::ILogicalDevice> device;
            core::smart_refctd_ptr<system::ILogger> logger;
        };

        CallbackContext ctx;
        ctx.semaphore = semaphore;
        ctx.cpuBuffers = cpuBufs;
        ctx.inputStagingBuffers = inputStagingBuffers;
        ctx.outputStagingBuffer = outputStagingBuffer;
        ctx.device = m_device;
        ctx.logger = m_logger;

        auto cudaCallback = [](void* userData)
        {
            const auto* ctx = reinterpret_cast<CallbackContext*>(userData);

            // Make sure we are also done with the readback 
            const auto wait = std::array{
              ISemaphore::SWaitInfo{
                .semaphore = ctx->semaphore.get(), 
                .value = 3,
              }
            };
            ctx->device->blockForSemaphores(wait, true);

            auto* stagingMem = ctx->outputStagingBuffer->getBoundMemory().memory;
            if (!stagingMem->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            {
                ILogicalDevice::MappedMemoryRange range(stagingMem, 0, stagingMem->getAllocationSize());
                ctx->device->invalidateMappedMemoryRanges(1, &range);
            }

            const auto* inputs1 = reinterpret_cast<float*>(ctx->cpuBuffers[0]->getPointer());
            const auto* inputs2 = reinterpret_cast<float*>(ctx->cpuBuffers[1]->getPointer());

            const auto* outputs = reinterpret_cast<float*>(ctx->outputStagingBuffer->getBoundMemory().memory->getMappedPointer());
            const auto* inputsInStaging1 = reinterpret_cast<float*>(ctx->inputStagingBuffers[0]->getBoundMemory().memory->getMappedPointer());
            const auto* inputsInStaging2 = reinterpret_cast<float*>(ctx->inputStagingBuffers[1]->getBoundMemory().memory->getMappedPointer());

            for (auto elem_i = 0; elem_i < NumElements; elem_i++)
            {
              const auto input1 = inputs1[elem_i];
              const auto input2 = inputs2[elem_i];
              const auto inputInStaging1 = inputsInStaging1[elem_i];
              const auto inputInStaging2 = inputsInStaging2[elem_i];
              if (inputInStaging1 != input1)
                ctx->logger->log("Input1 in Staging %d is incorrect!", ILogger::ELL_ERROR, elem_i);
              if (inputInStaging2 != input2)
                ctx->logger->log("Input2 in Staging %d is incorrect!", ILogger::ELL_ERROR, elem_i);

              const auto output = outputs[elem_i];
              const auto expected = input1 + input2;
              const auto diff = abs(output - expected);
              if (diff > 0.01)
                ctx->logger->log("TestSharedResources: Element at index %d is incorrect!", ILogger::ELL_ERROR, elem_i);
            }

            ctx->logger->log("TestSharedResources Complete", ILogger::ELL_INFO);
        };

        ASSERT_CUDA_SUCCESS(cu.pcuLaunchHostFunc(stream, cudaCallback, &ctx), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamSynchronize(stream), m_cuHandler);

        ASSERT_CUDA_SUCCESS(cu.pcuModuleUnload(module), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamDestroy_v2(stream), m_cuHandler);
    }

    void testWmmaGemB1()
    {
        // b1 WMMA dimensions: M=8, N=8, K=128
        constexpr auto WmmaSize = uint32_t3{ 8, 8, 128 };
        constexpr auto TileCount = uint32_t3{ 128, 128, 8 };  // Adjust for b1 dimensions
        constexpr auto ElementCount = WmmaSize * TileCount; // M=1024, N=1024, K=1024
        constexpr auto BlockDim = uint32_t2{ 32, 1 };       // 1 warp per block
        constexpr auto GridDim = uint32_t2(
            (ElementCount.x + WmmaSize.x - 1) / WmmaSize.x,  // M tiles
            (ElementCount.y + WmmaSize.y - 1) / WmmaSize.y   // N tiles
        );

        const auto ptx = compilePtx("app_resources/wmmaGemm_b1_kernel.cu");
        auto& cu = m_cuHandler->getCUDAFunctionTable();

        CUmodule   module;
        CUfunction kernel;
        CUstream   stream;

        ASSERT_CUDA_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "b1_wmma_gemm_kernel"), m_cuHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), m_cuHandler);

        // Calculate buffer sizes (bits packed into uint32_t)
        const size_t matA_size = (ElementCount.x * ElementCount.z) / 32 * sizeof(uint32_t); // M x K bits
        const size_t matB_size = (ElementCount.z * ElementCount.y) / 32 * sizeof(uint32_t); // K x N bits
        const size_t matC_size = ElementCount.x * ElementCount.y * sizeof(int32_t);         // M x N ints

        auto [vkBufferMatA, cuMemMatA] = createSharedBuffer(matA_size);
        auto [vkBufferMatB, cuMemMatB] = createSharedBuffer(matB_size);
        auto [vkBufferMatC, cuMemMatC] = createSharedBuffer(matC_size);

        // CPU matrices for initialization and verification
        core::vector<uint32_t> cpuMatA(ElementCount.x * ElementCount.z / 32);
        core::vector<uint32_t> cpuMatB(ElementCount.z * ElementCount.y / 32);
        core::vector<int32_t> cpuMatC_expected(ElementCount.x * ElementCount.y);

        // Initialize with simple patterns for verification
        auto initBinaryMatrices = [&]()
        {
            // Fill cpuMatA with reverse diagonal pattern
            std::fill(cpuMatA.begin(), cpuMatA.end(), 0);

            for (int i = 0; i < ElementCount.x; i++)
            {
              auto j = ElementCount.z - 1 - i;
              auto bitIdx = i * ElementCount.z + j;
              auto wordIdx = bitIdx / 32;
              auto bitOffset = bitIdx % 32;
              cpuMatA[wordIdx] |= (1u << bitOffset);
            }

            // Fill cpuMatB with random bits
            for (auto& val : cpuMatB) val = rand();
            
            // Compute expected result: For bmma with bmmaBitOpAND
            // C[i][j] = popcount(A[i,:] AND B[:,j])
            for (int i = 0; i < ElementCount.x; i++) {
                for (int j = 0; j < ElementCount.y; j++) {
                    const int k = ElementCount.z - 1 - i;
                    const int b_bit_idx = j * ElementCount.z + k; // col-major
                    const int32_t bit = (cpuMatB[b_bit_idx / 32] >> (b_bit_idx % 32)) & 1;
                    cpuMatC_expected[i * ElementCount.y + j] = bit;
                }
            }
        };
        initBinaryMatrices();
  
        ISemaphore::SCreationParams semParams;
        semParams.externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32;
        auto semaphore = m_device->createSemaphore(0, std::move(semParams));
        const auto cudaSemaphore = m_cuDevice->importExternalSemaphore(core::smart_refctd_ptr(semaphore));
        if (!cudaSemaphore)
          logFail("Fail to import Vulkan Semaphore into CUDA!");
        
        std::array<smart_refctd_ptr<IGPUCommandBuffer>, 2> cmd;
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, cmd.size(), cmd.data(), smart_refctd_ptr(m_logger));

        const auto outputStagingBuffer = createStaging(vkBufferMatC->getSize());

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
              .semaphore = semaphore.get(), .value = 1, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
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
            cuMemMatA->getMappedBuffer(&matrixAPtr);
            cuMemMatB->getMappedBuffer(&matrixBPtr);
            cuMemMatC->getMappedBuffer(&matrixCPtr);

            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(matrixAPtr, cpuMatA.data(), matA_size, stream), m_cuHandler);
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(matrixBPtr, cpuMatB.data(), matB_size, stream), m_cuHandler);
            core::vector<int32_t> cpuMatC(ElementCount.x * ElementCount.y, 15);
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(matrixCPtr, cpuMatC.data(), matC_size, stream), m_cuHandler);

            void* parameters[] = { &matrixAPtr, &matrixBPtr, &matrixCPtr, 
                                   (void*)&ElementCount.x, (void*)&ElementCount.y, (void*)&ElementCount.z };

            CUexternalSemaphore semaphore_cu = cudaSemaphore->getInternalObject();
            const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = 1 } } };
            ASSERT_CUDA_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore_cu, &waitParams, 1, stream), m_cuHandler);
            
            ASSERT_CUDA_SUCCESS(cu.pcuLaunchKernel(kernel, GridDim.x, GridDim.y, 1, 
                                                   BlockDim.x, BlockDim.y, 1, 
                                                   0, stream, parameters, nullptr), m_cuHandler);
            
            const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = 2 } } };
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
              .semaphore = semaphore.get(), .value = 2, .stageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
            };
            const IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = {
              .semaphore = semaphore.get(), .value = 3, .stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS,
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
        const auto wait = std::array{ ISemaphore::SWaitInfo{.semaphore = semaphore.get(), .value = 3} };
        m_device->blockForSemaphores(wait, true);

        auto* stagingMem = outputStagingBuffer->getBoundMemory().memory;
        if (!stagingMem->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
        {
            ILogicalDevice::MappedMemoryRange range(stagingMem, 0, stagingMem->getAllocationSize());
            m_device->invalidateMappedMemoryRanges(1, &range);
        }

        const auto* results = reinterpret_cast<int32_t*>(stagingMem->getMappedPointer());
        
        // Verify results
        bool success = true;
        int errors = 0;
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
                const auto expectedWordIdx = expectedIdx / 32;
                const auto expectedBitOffset = expectedIdx % 32;
                return (cpuMatB[expectedWordIdx] >> expectedBitOffset) & uint32_t(1);
            }();

            // const auto expected = [&]
            // {
            //     const auto row = i / ElementCount.y;            // row-major
            //     const auto col = i % ElementCount.y;
            //     const auto k   = ElementCount.z - 1 - row;      // reverse-diagonal A
            //     const auto bIdx = col * ElementCount.z + k;     // col-major B
            //     return (cpuMatB[bIdx / 32] >> (bIdx % 32)) & uint32_t(1);
            // }();

            // const auto expected = cpuMatC_expected[i];

            const auto result = results[i];
            if (result != expected) {
                m_logger->log("WMMA b1 test error at [%d]: GPU=%d, CPU=%d", 
                             system::ILogger::ELL_ERROR, i, results[i], expected);
                errors++;
                success = false;
            }
        }
        
        if (success)
            m_logger->log("b1 WMMA test PASSED!", system::ILogger::ELL_INFO);
        else
            m_logger->log("b1 WMMA test FAILED with %d errors!", system::ILogger::ELL_ERROR, errors);
    }

    void testDestruction()
    {
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        constexpr auto ElementCount = 1024;
        constexpr auto BufferSize = ElementCount * sizeof(int);
        auto& cu = m_cuHandler->getCUDAFunctionTable();
        smart_refctd_ptr<IDeviceMemoryAllocation> escaped;
        {
            const auto cudaMemory = m_cuDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .locationType = CU_MEM_LOCATION_TYPE_DEVICE });
            if (!cudaMemory) logFail("Fail to create exportable memory!");

            escaped = cudaMemory->exportAsMemory(m_device.get());
            if (!escaped) logFail("Fail to export CUDA memory!");
        
            auto tmpBuf = createExternalBuffer(escaped.get());
            auto staging = createStaging(BufferSize);
        
            auto ptr = (uint32_t*)staging->getBoundMemory().memory->getMappedPointer();
            for (uint32_t i = 0; i < ElementCount; ++i)
                ptr[i] = i;
        
            const auto semaphore = m_device->createSemaphore(0);
            IQueue::SSubmitInfo::SSemaphoreInfo semInfo;
            semInfo.semaphore = semaphore.get();
            semInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
            semInfo.value = 1;
        
            smart_refctd_ptr<IGPUCommandBuffer> cmdBuffer;
            commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmdBuffer);
            cmdBuffer->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            IGPUCommandBuffer::SBufferCopy region = { .size = BufferSize };
            assert(cmdBuffer->copyBuffer(staging.get(), tmpBuf.get(), 1, &region));
            cmdBuffer->end();
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmdBuffer.get() };
            const IQueue::SSubmitInfo submitInfo = {
              .commandBuffers = {&cmdInfo, &cmdInfo + 1}, 
              .signalSemaphores = {&semInfo, 1}
            };
            auto qre = queue->submit({ &submitInfo, &submitInfo + 1 });
            assert(IQueue::RESULT::SUCCESS == qre);
            m_device->waitIdle();
        }        
        
        {
            auto tmpBuf = createExternalBuffer(escaped.get());
            auto staging = createStaging(BufferSize);
        
            const auto semaphore = m_device->createSemaphore(0);
            IQueue::SSubmitInfo::SSemaphoreInfo semInfo;
            semInfo.semaphore = semaphore.get();
            semInfo.stageMask = PIPELINE_STAGE_FLAGS::ALL_COMMANDS_BITS;
            semInfo.value = 1;
        
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

    void testLargeAllocations()
    {
        // TODO(kevin): Calculate BufferSize that is big enough to fill the machine VRAM
        constexpr auto BufferSize = 1024;
        IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = {
            .size = BufferSize,
            .memoryTypeBits = m_physicalDevice->getDeviceLocalMemoryTypeBits(),
            .alignmentLog2 = 10,
        };
    
        for (size_t i = 0; i < (1 << 8); ++i)
        {
            auto memory = m_device->allocate(reqs, 0, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE).memory;
            assert(memory);
            auto tmpBuf = createExternalBuffer(memory.get());
        }
    }

    // Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
    bool keepRunning() override { return false; }

    // Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
    void workLoopBody() override {}
};

NBL_MAIN_FUNC(CUDA2VKApp)
