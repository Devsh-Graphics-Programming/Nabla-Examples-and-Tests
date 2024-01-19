// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#include "nbl/video/CCUDAHandler.h"
#include "nbl/video/CCUDASharedMemory.h"
#include "nbl/video/CCUDASharedSemaphore.h"

#include "../common/MonoDeviceApplication.hpp"
#include "../common/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;

/*
The start of the main function starts like in most other example. We ask the
user for the desired renderer and start it up.
*/

bool check_cuda_err(cudaError_enum err, auto& cu, auto& logger, auto file, auto line)
{
    if (auto re = err; CUDA_SUCCESS != re) 
    {
        const char* name = 0, * str = 0;
        cu.pcuGetErrorName(re, &name);
        cu.pcuGetErrorString(re, &str);
        logger->log("%s:%d %s:\n\t%s\n", system::ILogger::ELL_ERROR, file, line, name, str);
        return false;
    }
    return true;
}

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

#define ASSERT_SUCCESS(expr) { auto re = check_cuda_err((expr), cu, m_logger, __FILE__, __LINE__); assert(re); }
#define ASSERT_SUCCESS_NV(expr, log) { auto re = check_nv_err((expr), cudaHandler, m_logger, __FILE__, __LINE__, log); assert(re); }

#ifndef _NBL_COMPILE_WITH_CUDA_
static_assert(false);
#endif

class CUDA2VKApp : public examples::MonoDeviceApplication, public examples::MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = examples::MonoDeviceApplication;
    using asset_base_t = examples::MonoAssetManagerAndBuiltinResourceApplication;

    static constexpr uint32_t gridDim[3] = { 4096,1,1 };
    static constexpr uint32_t blockDim[3] = { 1024,1,1 };
    static constexpr size_t numElements = gridDim[0] * blockDim[0];
    static constexpr size_t size = sizeof(float) * numElements;
public:
    // Yay thanks to multiple inheritance we cannot forward ctors anymore
    CUDA2VKApp(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    smart_refctd_ptr<CCUDAHandler> cudaHandler;
    smart_refctd_ptr<CCUDADevice> cudaDevice;

    IQueue* queue;

    // CPU memory which we fill with random numbers between [-1,1] that will be copied to corresponding cudaMemory
    std::array<smart_refctd_ptr<ICPUBuffer>, 2> cpuBufs;
    // CUDA resources that we input to the kernel 'vectorAdd_kernel.cu'
    // Kernel writes to cudaMemories[2] which we later use to export and read on nabla side
    std::array<smart_refctd_ptr<CCUDASharedMemory>, 3> cudaMemories = {};
    // A semaphore created in CUDA which will alias a Nabla semaphore to help sync between the CUDA kernel and Nabla device to host transfer
    smart_refctd_ptr<CCUDASharedSemaphore> cudaSemaphore;

    // our Buffer that is bound to cudaMemories[2]
    smart_refctd_ptr<IGPUBuffer> importedBuf;
    // our Image that is also bound to cudaMemories[2]
    smart_refctd_ptr<IGPUImage> importedImg;

    // host visible buffers that we use to copy from the resources above after CUDA kernel is done writing
    smart_refctd_ptr<IGPUBuffer> stagingBufs[2];

    // Nabla semaphore for sync
    smart_refctd_ptr<ISemaphore> semaphore;

    smart_refctd_ptr<IGPUCommandPool> commandPool;
    smart_refctd_ptr<IGPUCommandBuffer> cmd[2];

    // a device filter helps you create a set of physical devices that satisfy your requirements in terms of features, limits etc.
    core::set<video::IPhysicalDevice*> filterDevices(const core::SRange<video::IPhysicalDevice* const>& physicalDevices) const override
    {
        auto devices = device_base_t::filterDevices(physicalDevices);
        auto& cuDevices = cudaHandler->getAvailableDevices();
        std::erase_if(devices, [&cuDevices](auto pdev) {
            return cuDevices.end() == std::find_if(cuDevices.begin(), cuDevices.end(), [pdev](auto& cuDev) { return !memcmp(pdev->getProperties().deviceUUID, &cuDev.uuid, 16);  });
        });
        return devices;
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

        
        queue = device_base_t::getComputeQueue();
        
        createResources();

        smart_refctd_ptr<ICPUBuffer> ptx;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            // this time we load a shader directly from a file
            auto assetBundle = m_assetMgr->getAsset("app_resources/vectorAdd_kernel.cu", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load kernel!");

            smart_refctd_ptr<ICPUBuffer> source = IAsset::castDown<ICPUBuffer>(assets[0]);
            std::string log;
            auto [ptx_, res] = cudaHandler->compileDirectlyToPTX(std::string((const char*)source->getPointer(), source->getSize()), 
                "app_resources/vectorAdd_kernel.cu", cudaDevice->geDefaultCompileOptions(), 0, 0, 0, &log);
            ASSERT_SUCCESS_NV(res, log);

            ptx = std::move(ptx_);
        }
        CUmodule   module;
        CUfunction kernel;
        CUstream   stream;

        auto& cu = cudaHandler->getCUDAFunctionTable();

        ASSERT_SUCCESS(cu.pcuModuleLoadDataEx(&module, ptx->getPointer(), 0u, nullptr, nullptr));
        ASSERT_SUCCESS(cu.pcuModuleGetFunction(&kernel, module, "vectorAdd"));
        ASSERT_SUCCESS(cu.pcuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

        launchKernel(kernel, stream);

        ASSERT_SUCCESS(cu.pcuStreamSynchronize(stream));
        ASSERT_SUCCESS(cu.pcuModuleUnload(module));
        ASSERT_SUCCESS(cu.pcuStreamDestroy_v2(stream));

        m_device->waitIdle();
        
        testInterop();

        return true;
    }

    void createResources()
    {
        auto& cu = cudaHandler->getCUDAFunctionTable();

        for (auto& buf : cpuBufs)
            buf = make_smart_refctd_ptr<ICPUBuffer>(size);

        for (auto j = 0; j < 2; j++)
            for (auto i = 0; i < numElements; i++)
                reinterpret_cast<float*>(cpuBufs[j]->getPointer())[i] = rand() / float(RAND_MAX);


        // create and allocate CUmem with CUDA and slap it inside a simple IReferenceCounted wrapper
        ASSERT_SUCCESS(cudaDevice->createSharedMemory(&cudaMemories[0], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
        ASSERT_SUCCESS(cudaDevice->createSharedMemory(&cudaMemories[1], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
        ASSERT_SUCCESS(cudaDevice->createSharedMemory(&cudaMemories[2], { .size = size, .alignment = sizeof(float), .location = CU_MEM_LOCATION_TYPE_DEVICE }));
        
        semaphore = m_device->createSemaphore(0, { .externalHandleTypes = ISemaphore::EHT_OPAQUE_WIN32 });
        ASSERT_SUCCESS(cudaDevice->importGPUSemaphore(&cudaSemaphore, semaphore.get()));
        {
            // export the CUmem we have just created into a refctd IDeviceMemoryAllocation
            auto devmemory = cudaMemories[2]->exportAsMemory(m_device.get());
            if (!devmemory)
                logFail("Failed to export CUDA memory!");


            // create an importing external buffer on Nabla side
            IGPUBuffer::SCreationParams params = {};
            params.size = devmemory->getAllocationSize();
            params.usage = asset::IBuffer::EUF_STORAGE_BUFFER_BIT | asset::IBuffer::EUF_TRANSFER_SRC_BIT;
            params.externalHandleTypes = CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE;
            importedBuf = m_device->createBuffer(std::move(params));
            if (!importedBuf) 
                logFail("Failed to create an external buffer");

            // bind that imported IDeviceMemoryAllocation to the external buffer we've just created
            ILogicalDevice::SBindBufferMemoryInfo bindInfo = { .buffer = importedBuf.get(), .binding = {.memory = devmemory.get() } };
            bool re = m_device->bindBufferMemory(1, &bindInfo);
            if (!re) logFail("Failed to bind CUDA memory to buffer");
        }

        {
            // same thing as above
            // we create an external image and bind the imported external memory to it
            // now we have 2 different resources that are bound to the same memory
            IImage::SCreationParams params = {};
            params.type = IGPUImage::ET_2D;
            params.samples = IGPUImage::ESCF_1_BIT;
            params.format = EF_R32_SFLOAT;
            params.extent = { gridDim[0], blockDim[0], 1 };
            params.mipLevels = 1;
            params.arrayLayers = 1;
            params.usage = IGPUImage::EUF_TRANSFER_SRC_BIT;
            importedImg = cudaMemories[2]->createAndBindImage(m_device.get(), std::move(params));
            if (!importedImg) logFail("Failed to create an external image");
        }
        
        commandPool = m_device->createCommandPool(queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        bool re = commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 2, cmd, smart_refctd_ptr(m_logger));

        stagingBufs[0] = createStaging();
        stagingBufs[1] = createStaging();
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

    smart_refctd_ptr<IGPUBuffer> createStaging(size_t sz = size)
    {
        auto buf = m_device->createBuffer({ {.size = sz, .usage = asset::IBuffer::EUF_TRANSFER_SRC_BIT | asset::IBuffer::EUF_TRANSFER_DST_BIT} });
        auto req = buf->getMemoryReqs();
        req.memoryTypeBits &= m_device->getPhysicalDevice()->getDownStreamingMemoryTypeBits();
        auto allocation = m_device->allocate(req, buf.get());

        void* mapping = allocation.memory->map(IDeviceMemoryAllocation::MemoryRange(0, req.size), IDeviceMemoryAllocation::EMCAF_READ);
        if (!mapping)
            logFail("Failed to map an staging buffer");
        memset(mapping, 0, req.size);
        return buf;
    };

    void launchKernel(CUfunction kernel, CUstream stream)
    {

        // First we record a release ownership transfer to let vulkan know that resources are going to be used in an external API
        {
            IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                    .barrier = {
                        .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
                        .otherQueueFamilyIndex = IQueue::FamilyExternal,
                    },
                    .range = {.buffer = importedBuf, },
            };

            IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarrier = {
                .barrier = {
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::RELEASE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .image = importedImg.get(),
                .subresourceRange = {
                    .aspectMask = IImage::EAF_COLOR_BIT,
                    .levelCount = 1u,
                    .layerCount = 1u,
                }
            };
            // start recording
            bool re = true;
            re &= cmd[0]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            re &= cmd[0]->pipelineBarrier(EDF_NONE, { .bufBarriers = std::span{&bufBarrier,&bufBarrier + 1}, .imgBarriers = {&imgBarrier,&imgBarrier + 1} });
            re &= cmd[0]->end();

            IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = { .semaphore = semaphore.get(), .value = 1 };
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[0].get()};
            IQueue::SSubmitInfo submitInfo = { .commandBuffers = {&cmdInfo, &cmdInfo + 1}, .signalSemaphores = {&signalInfo,&signalInfo + 1} };
            auto submitRe = queue->submit({ &submitInfo,&submitInfo + 1 });
            re &= IQueue::RESULT::SUCCESS == submitRe;
            if (!re)
                logFail("Something went wrong readying resources for CUDA");
        }
        
        auto& cu = cudaHandler->getCUDAFunctionTable();
        // Launch kernel
        {
            CUdeviceptr ptrs[] = {
                cudaMemories[0]->getDeviceptr(),
                cudaMemories[1]->getDeviceptr(),
                cudaMemories[2]->getDeviceptr(),
            };
            auto numEles = numElements;
            void* parameters[] = { &ptrs[0], &ptrs[1], &ptrs[2], &numEles };
            ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[0], cpuBufs[0]->getPointer(), size, stream));
            ASSERT_SUCCESS(cu.pcuMemcpyHtoDAsync_v2(ptrs[1], cpuBufs[1]->getPointer(), size, stream));

            auto semaphore = cudaSemaphore->getInternalObject();
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams = { .params = {.fence = {.value = 1 } } };
            ASSERT_SUCCESS(cu.pcuWaitExternalSemaphoresAsync(&semaphore, &waitParams, 1, stream)); // Wait for release op from vulkan
            ASSERT_SUCCESS(cu.pcuLaunchKernel(kernel, gridDim[0], gridDim[1], gridDim[2], blockDim[0], blockDim[1], blockDim[2], 0, stream, parameters, nullptr));
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signalParams = { .params = {.fence = {.value = 2 } } };
            ASSERT_SUCCESS(cu.pcuSignalExternalSemaphoresAsync(&semaphore, &signalParams, 1, stream)); // Signal the imported semaphore
        }
        
        // After the cuda kernel has signalled our exported vk semaphore, we will download the results through the buffer imported from CUDA
        {
            IGPUCommandBuffer::SBufferMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> bufBarrier = {
                .barrier = {
                    .dep = {
                        .dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
                        .dstAccessMask = ACCESS_FLAGS::TRANSFER_READ_BIT,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .range = { .buffer = importedBuf, },
            };
            bool re = true;
            re &= cmd[1]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);

            re &= cmd[1]->pipelineBarrier(EDF_NONE, {.bufBarriers = std::span{&bufBarrier,&bufBarrier + 1}});

            IGPUCommandBuffer::SBufferCopy region = { .size = size };
            re &= cmd[1]->copyBuffer(importedBuf.get(), stagingBufs[0].get(), 1, &region);

            IGPUCommandBuffer::SImageMemoryBarrier<IGPUCommandBuffer::SOwnershipTransferBarrier> imgBarrier = {
                .barrier = { 
                    .dep = { 
                        .dstStageMask = PIPELINE_STAGE_FLAGS::COPY_BIT,
                        .dstAccessMask = ACCESS_FLAGS::MEMORY_READ_BITS,
                    },
                    .ownershipOp = IGPUCommandBuffer::SOwnershipTransferBarrier::OWNERSHIP_OP::ACQUIRE,
                    .otherQueueFamilyIndex = IQueue::FamilyExternal,
                },
                .image = importedImg.get(),
                .subresourceRange = {
                    .aspectMask = IImage::EAF_COLOR_BIT,
                    .levelCount = 1u,
                    .layerCount = 1u,
                },
                .oldLayout = IImage::LAYOUT::PREINITIALIZED,
                .newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL,
            };

            re &= cmd[1]->pipelineBarrier(EDF_NONE, {.imgBarriers = {&imgBarrier,&imgBarrier + 1}});

            IImage::SBufferCopy imgRegion = {
                .imageSubresource = {
                    .aspectMask = imgBarrier.subresourceRange.aspectMask,
                    .layerCount = imgBarrier.subresourceRange.layerCount,
                },
                .imageExtent = importedImg->getCreationParameters().extent,
            };

            re &= cmd[1]->copyImageToBuffer(importedImg.get(), imgBarrier.newLayout, stagingBufs[1].get(), 1, &imgRegion);
            re &= cmd[1]->end();
            
            IQueue::SSubmitInfo::SSemaphoreInfo waitInfo= { .semaphore = semaphore.get(), .value = 2 };
            IQueue::SSubmitInfo::SSemaphoreInfo signalInfo = { .semaphore = semaphore.get(), .value = 3 };
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[1].get() };
            IQueue::SSubmitInfo submitInfo = { 
                .waitSemaphores = {&waitInfo,&waitInfo + 1},
                .commandBuffers = {&cmdInfo, &cmdInfo + 1},  
                .signalSemaphores = {&signalInfo,&signalInfo + 1} 
            };
            auto submitRe = queue->submit({ &submitInfo,&submitInfo + 1 });
            re &= IQueue::RESULT::SUCCESS == submitRe;
            if (!re)
                logFail("Something went wrong copying results from CUDA");
        }
        
        ASSERT_SUCCESS(cu.pcuLaunchHostFunc(stream, [](void* userData) { decltype(this)(userData)->kernelCallback(); }, this));
    }

    void kernelCallback()
    {
        // Make sure we are also done with the readback
        auto wait = std::array{ISemaphore::SWaitInfo{.semaphore = semaphore.get(), .value = 3}};
        m_device->waitForSemaphores(wait, true, -1);

        float* A = reinterpret_cast<float*>(cpuBufs[0]->getPointer());
        float* B = reinterpret_cast<float*>(cpuBufs[1]->getPointer());

        float* CBuf = reinterpret_cast<float*>(stagingBufs[0]->getBoundMemory().memory->getMappedPointer());
        float* CImg = reinterpret_cast<float*>(stagingBufs[1]->getBoundMemory().memory->getMappedPointer());

        if(memcmp(CBuf, CImg, size))
            logFail("Buffer and Image memories do not match!");

        for (auto i = 0; i < numElements; i++)
        {
            bool re = (abs(CBuf[i] - A[i] - B[i]) < 0.01f) && (abs(CImg[i] - A[i] - B[i]) < 0.01f);
            if(!re)
                logFail("Element at index %d is incorrect!", i);
        }
        
        std::cout << "Success\n";
    }


    void testInterop()
    {
        {
            IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = {
                .size = size,
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

        smart_refctd_ptr<IDeviceMemoryAllocation> escaped;
        {
            IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = {
                .size = size,
                .memoryTypeBits = m_physicalDevice->getDeviceLocalMemoryTypeBits(),
                .alignmentLog2 = 10,
            };

            auto memory = m_device->allocate(reqs, 0, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE).memory;

            auto tmpBuf = createExternalBuffer(memory.get());
            auto staging = createStaging();

            auto ptr = (uint32_t*)staging->getBoundMemory().memory->getMappedPointer();
            for (uint32_t i = 0; i < size / 4; ++i)
                ptr[i] = i;

            smart_refctd_ptr<IGPUCommandBuffer> cmd;
            commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmd);
            cmd->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            IGPUCommandBuffer::SBufferCopy region = { .size = size };
            assert(cmd->copyBuffer(staging.get(), tmpBuf.get(), 1, &region));
            cmd->end();
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd.get() };
            IQueue::SSubmitInfo submitInfo = { .commandBuffers = {&cmdInfo, &cmdInfo + 1} };
            queue->submit({ &submitInfo,&submitInfo + 1 });
            m_device->waitIdle();
            escaped = m_device->allocate(reqs, 0, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE, memory->getCreationParams().externalHandle).memory;
        }

        //{
        //    constexpr size_t M = 32;
        //    auto staging = createStaging(size * M);

        //    auto ptr = (uint32_t*)staging->getBoundMemory().memory->getMappedPointer();
        //    for (uint32_t i = 0; i < (M * size) / 4; ++i)
        //        ptr[i] = rand();

        //    std::vector<smart_refctd_ptr<IGPUCommandBuffer>> cmd(1 << 10);
        //    commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1 << 10, cmd.data());

        //    for (size_t i = 0; i < 1 << 10; ++i)
        //    {
        //        IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = {
        //            .size = size * M,
        //            .memoryTypeBits = m_physicalDevice->getDeviceLocalMemoryTypeBits(),
        //            .alignmentLog2 = 10,
        //        };
        //    RE:
        //        auto memory = m_device->allocate(reqs, 0, IDeviceMemoryAllocation::E_MEMORY_ALLOCATE_FLAGS::EMAF_NONE, CCUDADevice::EXTERNAL_MEMORY_HANDLE_TYPE).memory;

        //        if (!memory)
        //        {
        //            m_device->waitIdle();
        //            for (size_t j = 0; j < i; ++j)
        //                cmd[j] = 0;
        //            goto END;
        //        }
        //        assert(memory);
        //        auto tmpBuf = createExternalBuffer(memory.get());

        //        cmd[i]->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
        //        IGPUCommandBuffer::SBufferCopy region = { .size = size * M };
        //        assert(cmd[i]->copyBuffer(staging.get(), tmpBuf.get(), 1, &region));
        //        cmd[i]->end();
        //        IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd[i].get() };
        //        IQueue::SSubmitInfo submitInfo = { .commandBuffers = {&cmdInfo, &cmdInfo + 1} };
        //        assert(IQueue::RESULT::SUCCESS == queue->submit({ &submitInfo,&submitInfo + 1 }));
        //    }
        //END:
        //    m_device->waitIdle();
        //}

        {
            auto tmpBuf = createExternalBuffer(escaped.get());
            auto staging = createStaging();

            smart_refctd_ptr<IGPUCommandBuffer> cmd;
            commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1, &cmd);
            cmd->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            IGPUCommandBuffer::SBufferCopy region = { .size = size };
            assert(cmd->copyBuffer(tmpBuf.get(), staging.get(), 1, &region));
            cmd->end();
            IQueue::SSubmitInfo::SCommandBufferInfo cmdInfo = { cmd.get() };
            IQueue::SSubmitInfo submitInfo = { .commandBuffers = {&cmdInfo, &cmdInfo + 1} };
            auto qre = queue->submit({ &submitInfo,&submitInfo + 1 });
            assert(IQueue::RESULT::SUCCESS == qre);
            m_device->waitIdle();

            auto& ptr = *(std::array<uint32_t, size>*)staging->getBoundMemory().memory->getMappedPointer();
            for (uint32_t i = 0; i < size / 4; ++i)
                assert(ptr[i] == i);
        }

    }


    // Whether to keep invoking the above. In this example because its headless GPU compute, we do all the work in the app initialization.
    bool keepRunning() override { return false; }

    // Platforms like WASM expect the main entry point to periodically return control, hence if you want a crossplatform app, you have to let the framework deal with your "game loop"
    void workLoopBody() override {}
};

NBL_MAIN_FUNC(CUDA2VKApp)