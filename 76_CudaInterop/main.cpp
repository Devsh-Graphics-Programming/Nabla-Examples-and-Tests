#include "nbl/video/CCUDAHandler.h"
// #include "nbl/video/CCUDAExportableMemory.h"
// #include "nbl/video/CCUDAImportedSemaphore.h"

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/

bool check_nv_err(auto err, auto& cudaHandler, auto& logger, auto file, auto line, std::string const& log)
{
    if (auto re = err; NVRTC_SUCCESS != re) 
    {
        const char* str = cudaHandler->getNVRTCFunctionTable().pnvrtcGetErrorString(re); 
        logger->log("%s:%d %s\n%s\n", system::ILogger::ELL_ERROR, file, line, str, log.c_str());
        return false;
    }
    return true;
}

#define ASSERT_NV_SUCCESS(expr, log) { auto re = check_nv_err((expr), cudaHandler, m_logger, __FILE__, __LINE__, log); assert(re); }


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

    smart_refctd_ptr<CCUDAHandler> cudaHandler;
    smart_refctd_ptr<CCUDADevice> cudaDevice;

    IQueue* queue;


    // a device filter helps you create a set of physical devices that satisfy your requirements in terms of features, limits etc.
    virtual void filterDevices(core::set<video::IPhysicalDevice*>& physicalDevices) const
    {
        device_base_t::filterDevices(physicalDevices);
        auto& cuDevices = cudaHandler->getAvailableDevices();
        std::erase_if(physicalDevices, [&cuDevices](auto pdev) {
            return cuDevices.end() == std::find_if(cuDevices.begin(), cuDevices.end(), [pdev](auto& cuDev) { return !memcmp(pdev->getProperties().deviceUUID, &cuDev.uuid, 16);  });
        });
    }

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!asset_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;

        cudaHandler = CCUDAHandler::create(m_system.get(), smart_refctd_ptr<ILogger>(m_logger));
        if (!cudaHandler) 
            return logFail("Could not create a CUDA handler!");

        if (!device_base_t::onAppInitialized(std::move(system)))
            return false;

        cudaDevice = cudaHandler->createDevice(smart_refctd_ptr_dynamic_cast<CVulkanConnection>(m_api), m_physicalDevice);
        if (!cudaDevice) 
            return logFail("Could not create a CUDA Device!");


        queue = getComputeQueue();

        testVectorAddKernel();
        testDestruction();
        testLargeAllocations();

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

    void testVectorAddKernel()
    {
        static constexpr uint32_t GridDim[3] = { 4096,1,1 };
        static constexpr uint32_t BlockDim[3] = { 1024,1,1 };
        static constexpr size_t NumElements = GridDim[0] * BlockDim[0];
        static constexpr size_t BufferSize = sizeof(float) * NumElements;

        smart_refctd_ptr<ICPUBuffer> ptx;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            // this time we load a shader directly from a file
            auto assetBundle = m_assetMgr->getAsset("app_resources/vectorAdd_kernel.cu", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                logFail("Could not load kernel!");

            smart_refctd_ptr<ICPUBuffer> source = IAsset::castDown<ICPUBuffer>(assets[0]);
            std::string log;
            auto [ptx_, res] = cudaHandler->compileDirectlyToPTX(std::string((const char*)source->getPointer(), source->getSize()), 
                "app_resources/vectorAdd_kernel.cu", cudaDevice->geDefaultCompileOptions(), 0, 0, 0, &log);
            ASSERT_NV_SUCCESS(res, log);

            ptx = std::move(ptx_);
        }

        auto& cu = cudaHandler->getCUDAFunctionTable();

        CUmodule   module;
        CUfunction kernel;
        CUstream   stream;

        ASSERT_CUDA_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr), cudaHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"), cudaHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), cudaHandler);

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
          cudaInputMemories[input_i] = cudaDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE });
          vulkanMemories[input_i] = cudaInputMemories[input_i]->exportAsMemory(m_device.get(), nullptr);
          vulkanInputBuffers[input_i] = createExternalBuffer(vulkanMemories[input_i].get());
          inputStagingBuffers[input_i] = createStaging(BufferSize);
        }

        IGPUBuffer::SCreationParams outputBufferParams;
        outputBufferParams.size = cudaDevice->roundToGranularity(CU_MEM_LOCATION_TYPE_DEVICE, BufferSize);
        outputBufferParams.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT;
        outputBufferParams.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE;
        const auto outputBuf = m_device->createBuffer(std::move(outputBufferParams));
        auto outputMemReq = outputBuf->getMemoryReqs();

        auto allocation = m_device->allocate(outputMemReq, outputBuf.get(), IDeviceMemoryAllocation::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE);
        const auto cudaOutputMemory = cudaDevice->importExternalMemory(core::smart_refctd_ptr(allocation.memory));
        if (!cudaOutputMemory)
          logFail("Fail to import Vulkan Memory into CUDA!");
        
        ISemaphore::SCreationParams semParams;
        semParams.externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32;
        auto semaphore = m_device->createSemaphore(0, std::move(semParams));
        const auto cudaSemaphore = cudaDevice->importExternalSemaphore(core::smart_refctd_ptr(semaphore));
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
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[0], cpuBufs[0]->getPointer(), BufferSize, stream), cudaHandler);
            ASSERT_CUDA_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[1], cpuBufs[1]->getPointer(), BufferSize, stream), cudaHandler);
    
            auto semaphore = cudaSemaphore->getInternalObject();
            const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = 1 } } };
            ASSERT_CUDA_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream), cudaHandler); // Wait for release op from vulkan
            ASSERT_CUDA_SUCCESS(cu.pcuLaunchKernel(kernel, GridDim[0], GridDim[1], GridDim[2], BlockDim[0], BlockDim[1], BlockDim[2], 0, stream, parameters, nullptr), cudaHandler);
            const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = 2 } } };
            ASSERT_CUDA_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream), cudaHandler); // Signal the imported semaphore
        }
        ASSERT_CUDA_SUCCESS(cu.pcuStreamSynchronize(stream), cudaHandler);
        
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
              if (diff < 0.01)
                ctx->logger->log("TestSharedResources: Element at index %d is incorrect!", ILogger::ELL_ERROR, elem_i);
            }

            ctx->logger->log("TestSharedResources Complete", ILogger::ELL_INFO);
        };

        ASSERT_CUDA_SUCCESS(cu.pcuLaunchHostFunc(stream, cudaCallback, &ctx), cudaHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamSynchronize(stream), cudaHandler);

        ASSERT_CUDA_SUCCESS(cu.pcuModuleUnload(module), cudaHandler);
        ASSERT_CUDA_SUCCESS(cu.pcuStreamDestroy_v2(stream), cudaHandler);
    }

    void testDestruction()
    {
        auto commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        constexpr auto ElementCount = 1024;
        constexpr auto BufferSize = ElementCount * sizeof(int);
        auto& cu = cudaHandler->getCUDAFunctionTable();
        smart_refctd_ptr<IDeviceMemoryAllocation> escaped;
        {
            const auto cudaMemory = cudaDevice->createExportableMemory({ .size = BufferSize, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE });
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
    
        // {
        //     constexpr size_t M = 32;
        //     auto staging = createStaging(size * M);
        //
        //     auto ptr = (uint32_t*)staging->getBoundMemory().memory->getMappedPointer();
        //     for (uint32_t i = 0; i < (M * size) / 4; ++i)
        //         ptr[i] = rand();
        //
        //     std::vector<smart_refctd_ptr<IGPUCommandBuffer>> cmd(1 << 10);
        //     commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1 << 10, cmd.data());
        //
        //     for (size_t i = 0; i < 1 << 10; ++i)
        //     {
        //         IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = {
        //             .size = size * M,
        //             .memoryTypeBits = m_physicalDevice->getDeviceLocalMemoryTypeBits(),
        //             .alignmentLog2 = 10,
        //         };
        //     RE:
        //         auto memory = m_device->allocate(reqs, 0, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE).memory;
        //
        //         if (!memory)
        //         {
        //             m_device->waitIdle();
        //             for (size_t j = 0; j < i; ++j)
        //                 cmd[j] = 0;
        //             goto END;
        //         }
        //         assert(memory);
        //         auto tmpBuf = createExternalBuffer(memory.get());
        //
        //         cmd[i]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        //         IGPUCommandBuffer::SBufferCopy region = { .size = size * M };
        //         assert(cmd[i]->copyBuffer(staging.get(), tmpBuf.get(), 1, &region));
        //         cmd[i]->end();
        //         IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[i].get() };
        //         IQueue::SSubmitInfo submitInfo = { .commandBuffers = {&cmdInfo, &cmdInfo + 1} };
        //         assert(IQueue::RESULT::SUCCESS == queue->submit({ &submitInfo,&submitInfo + 1 }));
        //     }
        // END:
        //     m_device->waitIdle();
        // }
    
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