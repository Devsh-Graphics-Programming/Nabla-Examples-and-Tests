// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <assert.h>

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "app_resources/common.hlsl"
#include "app_resources/emulated_float64_t_test/common.hlsl"


using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::application_templates;


//using namespace glm;

void cpu_tests();

struct S
{
    float32_t3 f;
};

struct T
{
    float32_t       a;
    float32_t3      b;
    S               c;
    float32_t2x3    d;
    float32_t2x3    e;
    int             f[3];
    float32_t2      g[2];
    float32_t4      h;
};

class CompatibilityTest final : public MonoDeviceApplication, public MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = MonoDeviceApplication;
    using asset_base_t = MonoAssetManagerAndBuiltinResourceApplication;
public:
    CompatibilityTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;
    
        m_queue = m_device->getQueue(0, 0);
        m_commandPool = m_device->createCommandPool(m_queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        m_commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &m_cmdbuf,1 }, smart_refctd_ptr(m_logger));

        // TODO: remove
        {
            emulated::emulated_float64_t a = emulated::emulated_float64_t::create(123.321);
            emulated::emulated_float64_t b = emulated::emulated_float64_t::create(12233.69);

            auto add = a + b;
            auto sub = a - b;
            auto mul = a * b;
            auto div = a / b;
            std::cout << "a: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(a) << std::endl;
            std::cout << "b: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(b) << std::endl << std::endl;;
            std::cout << "add: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(add) << std::endl << std::endl;
            std::cout << "sub: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(sub) << std::endl << std::endl;
            std::cout << "mul: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(mul) << std::endl << std::endl;
            std::cout << "div: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(div) << std::endl << std::endl << std::endl;
        }

        {
            emulated::emulated_float64_t a = emulated::emulated_float64_t::create(-123.321);
            emulated::emulated_float64_t b = emulated::emulated_float64_t::create(12233.69);
            emulated::emulated_float64_t c = emulated::emulated_float64_t::create(123ull);

            auto add = a + b;
            auto sub = a - b;
            auto mul = a * b;
            auto div = a / b;
            std::cout << "a: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(a) << std::endl;
            std::cout << "b: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(b) << std::endl << std::endl;;
            std::cout << "c: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(c) << std::endl << std::endl;;
            std::cout << "add: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(add) << std::endl << std::endl;
            std::cout << "sub: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(sub) << std::endl << std::endl;
            std::cout << "mul: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(mul) << std::endl << std::endl;
            std::cout << "div: ";
            std::cout << std::fixed << std::setprecision(20) << reinterpret_cast<double&>(div) << std::endl << std::endl;
        }

        smart_refctd_ptr<IGPUShader> shader;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            auto assetBundle = m_assetMgr->getAsset("app_resources/test.comp.hlsl", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load shader!");

            // lets go straight from ICPUSpecializedShader to IGPUSpecializedShader
            auto source = IAsset::castDown<ICPUShader>(assets[0]);
            // The down-cast should not fail!
            assert(source);
            assert(source->getStage() == IShader::ESS_COMPUTE);

            // this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
            shader = m_device->createShader(source.get());
            if (!shader)
                return logFail("Creation of a GPU Shader to from CPU Shader source failed!");
        }

		const uint32_t bindingCount = 4u;
		IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {};
		bindings[0].type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
		bindings[1].type = IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
		bindings[2].type = IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
		bindings[3].type = IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
        
        for(int i = 0; i < bindingCount; ++i)
        {
            bindings[i].stageFlags = IShader::ESS_COMPUTE;
            bindings[i].count = 1;
            bindings[i].binding = i;
        }
		m_descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
        {
		    SPushConstantRange pcRange = {};
		    pcRange.stageFlags = IShader::ESS_COMPUTE;
		    pcRange.offset = 0u;
		    pcRange.size = 2 * sizeof(uint32_t);
            auto layout = m_device->createPipelineLayout({ &pcRange,1 }, smart_refctd_ptr(m_descriptorSetLayout));
            IGPUComputePipeline::SCreationParams params = {};
            params.layout = layout.get();
            params.shader.shader = shader.get();
            if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                return logFail("Failed to create compute pipeline!\n");
        }


        for (int i = 0; i < 2; ++i)
        {
            m_images[i] = m_device->createImage(IGPUImage::SCreationParams {
                {
                    .type = IGPUImage::E_TYPE::ET_2D,
                    .samples = IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
                    .format = E_FORMAT::EF_R32G32B32A32_SFLOAT,
                    .extent = { 1920,1080,1 },
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .usage = IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT 
                        | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT
                        | IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT,
                }, {}, IGPUImage::TILING::LINEAR,
            });

            auto reqs = m_images[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            m_device->allocate(reqs, m_images[i].get());

            m_imageViews[i] = m_device->createImageView(IGPUImageView::SCreationParams {
                .image = m_images[i],
                    .viewType = IGPUImageView::E_TYPE::ET_2D,
                    .format = E_FORMAT::EF_R32G32B32A32_SFLOAT,
                    // .subresourceRange = { IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, 0, 1, 0, 1 },
            });

            m_buffers[i] = m_device->createBuffer(IGPUBuffer::SCreationParams {
                {.size = reqs.size, .usage = 
                    IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | 
                    IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT,
                }
            });

            reqs = m_buffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            m_device->allocate(reqs, m_buffers[i].get());

            m_readbackBuffers[i] = m_device->createBuffer(IGPUBuffer::SCreationParams {
                {.size = reqs.size, .usage = IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT }
            });

            reqs = m_readbackBuffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            m_device->allocate(reqs, m_readbackBuffers[i].get());
        }

        smart_refctd_ptr<IDescriptorPool> descriptorPool = nullptr;
        {
            IDescriptorPool::SCreateInfo createInfo = {};
            createInfo.maxSets = 1;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = 2;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 2;
            descriptorPool = m_device->createDescriptorPool(std::move(createInfo));
        }

        m_descriptorSet = descriptorPool->createDescriptorSet(smart_refctd_ptr(m_descriptorSetLayout));


        IGPUDescriptorSet::SDescriptorInfo descriptorInfos[bindingCount] = {};
        IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[bindingCount] = {};
        
        for(int i = 0; i < bindingCount; ++i)
        {
            writeDescriptorSets[i].info = &descriptorInfos[i];
            writeDescriptorSets[i].dstSet = m_descriptorSet.get();
            writeDescriptorSets[i].binding = i;
            writeDescriptorSets[i].count = bindings[i].count;

            if(i<2)
            {
                descriptorInfos[i].desc = m_imageViews[i];
                descriptorInfos[i].info.image.imageLayout = IImage::LAYOUT::GENERAL;
            }
            else
            {
                descriptorInfos[i].desc = m_buffers[i-2];
                descriptorInfos[i].info.buffer.size = ~0ull;
            }
        }

        m_device->updateDescriptorSets(bindingCount, writeDescriptorSets, 0u, nullptr);
       
        // In contrast to fences, we just need one semaphore to rule all dispatches
        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override
    {
        cpu_tests();

        constexpr auto StartedValue = 0;

        smart_refctd_ptr<ISemaphore> progress = m_device->createSemaphore(StartedValue);
        

        m_cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::RELEASE_RESOURCES_BIT);
        m_cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);


        IGPUCommandBuffer::SPipelineBarrierDependencyInfo::image_barrier_t layoutTransBarriers[2] = { {
            .barrier = {
                .dep = {
                    .srcStageMask = PIPELINE_STAGE_FLAGS::HOST_BIT,
                    .srcAccessMask = ACCESS_FLAGS::HOST_WRITE_BIT,
                    .dstStageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT,
                    .dstAccessMask = ACCESS_FLAGS::SHADER_WRITE_BITS
                }
            },
            .image = m_images[0].get(),
            .subresourceRange = {
                .aspectMask = IImage::EAF_COLOR_BIT,
                .baseMipLevel = 0u,
                .levelCount = 1u,
                .baseArrayLayer = 0u,
                .layerCount = 1u,
            },
            .oldLayout = IImage::LAYOUT::UNDEFINED,
            .newLayout = IImage::LAYOUT::GENERAL
        } };
        layoutTransBarriers[1] = layoutTransBarriers[0];
        layoutTransBarriers[1].image = m_images[1].get();

        const IGPUCommandBuffer::SPipelineBarrierDependencyInfo depInfo = { .imgBarriers = layoutTransBarriers };
        m_cmdbuf->pipelineBarrier(EDF_NONE, depInfo);
        

        const uint32_t pushConstants[2] = { 1920, 1080 };
        const IGPUDescriptorSet* set = m_descriptorSet.get();
        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->bindDescriptorSets(EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, &set);
        m_cmdbuf->dispatch(240, 135, 1u);
        for (int i = 0; i < 2; ++i)
        {
            layoutTransBarriers[i].barrier.dep = layoutTransBarriers[i].barrier.dep.nextBarrier(PIPELINE_STAGE_FLAGS::COPY_BIT,ACCESS_FLAGS::TRANSFER_READ_BIT);
            layoutTransBarriers[i].oldLayout = layoutTransBarriers[i].newLayout;
            layoutTransBarriers[i].newLayout = IImage::LAYOUT::TRANSFER_SRC_OPTIMAL;
        }
        m_cmdbuf->pipelineBarrier(EDF_NONE,depInfo);

        //{
        //    constexpr auto FinishedValue1 = 42;
        //    IQueue::SSubmitInfo submitInfos[1] = {};
        //    const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
        //    submitInfos[0].commandBuffers = cmdbufs;
        //    const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue1,.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
        //    submitInfos[0].signalSemaphores = signals;
        //    m_queue->startCapture();
        //    m_queue->submit(submitInfos);  //Command buffer is NOT IN THE EXECUTABLE STATE
        //    m_queue->endCapture();
        //    const ISemaphore::SWaitInfo waitInfos[] = { {
        //            .semaphore = progress.get(),
        //            .value = FinishedValue1
        //        } };
        //    m_device->blockForSemaphores(waitInfos);
       
        //}
        IImage::SBufferCopy copy = {
            .imageSubresource = { 
                .aspectMask = IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageExtent = {1920, 1080, 1},
        };
        
        bool succ = m_cmdbuf->copyImageToBuffer(m_images[0].get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, m_readbackBuffers[0].get(), 1, &copy);
        succ &= m_cmdbuf->copyImageToBuffer(m_images[1].get(), IImage::LAYOUT::TRANSFER_SRC_OPTIMAL, m_readbackBuffers[1].get(), 1, &copy);
        assert(succ);
        m_cmdbuf->end();

        {
            constexpr auto FinishedValue = 69;
            IQueue::SSubmitInfo submitInfos[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
            submitInfos[0].commandBuffers = cmdbufs;
            const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
            submitInfos[0].signalSemaphores = signals;
            m_queue->startCapture();
            m_queue->submit(submitInfos);
            m_queue->endCapture();
            const ISemaphore::SWaitInfo waitInfos[] = { {
                    .semaphore = progress.get(),
                    .value = FinishedValue
                } };
            m_device->blockForSemaphores(waitInfos);
        }

        using res = std::array<std::array<std::array<float, 4>, 1080>, 1920>;
        res* ptrs[4] = {};
        
        static_assert(sizeof(res) == sizeof(float) * 4 * 1920 * 1080);

        for (int i = 0; i < 4; ++i)
        {
            auto mem = (i < 2 ? m_buffers[i] : m_readbackBuffers[i-2])->getBoundMemory();
            assert(mem.memory->isMappable());
            auto* ptr = mem.memory->map({ .offset = 0, .length = mem.memory->getAllocationSize() });
            ptrs[i] = (res*)ptr;
        }
        res& buf = *ptrs[1];
        res& img = *ptrs[3];

        std::cout << buf[0][0][0] << " " 
                  << buf[0][0][1] << " "
                  << buf[0][0][2] << " "
                  << buf[0][0][3] << " "
                  << "\n";
                  
        const std::ios::fmtflags f(std::cout.flags());
        std::cout << std::hex
            << std::bit_cast<uint32_t>(buf[0][0][0]) << " " 
            << std::bit_cast<uint32_t>(buf[0][0][1]) << " "
            << std::bit_cast<uint32_t>(buf[0][0][2]) << " "
            << std::bit_cast<uint32_t>(buf[0][0][3]) << " "
            << "\n";
        std::cout.flags(f);

        if(buf[0][0][0] != -1.f)
        {
            std::cout << "Shader tests failed\n";
        }

        validateEmulatedFloat64();

        m_keepRunning = false;
    }

    bool keepRunning() override
    {
        return m_keepRunning;
    }


private:
    smart_refctd_ptr<IGPUComputePipeline> m_pipeline = nullptr;
    smart_refctd_ptr<IGPUDescriptorSetLayout> m_descriptorSetLayout;
    smart_refctd_ptr<IGPUDescriptorSet> m_descriptorSet;

    smart_refctd_ptr<IGPUImage> m_images[2];
    smart_refctd_ptr<IGPUBuffer> m_buffers[2];
    smart_refctd_ptr<IGPUBuffer> m_readbackBuffers[2];
    smart_refctd_ptr<IGPUImageView> m_imageViews[2];
    smart_refctd_ptr<IGPUCommandBuffer> m_cmdbuf = nullptr;
    IQueue* m_queue;
    smart_refctd_ptr<IGPUCommandPool> m_commandPool;
    uint64_t m_iteration = 0;
    constexpr static inline uint64_t MaxIterations = 200;

    bool m_keepRunning = true;
    
    bool validateEmulatedFloat64()
    {
        constexpr auto FinishedValue = 45;
        smart_refctd_ptr<ISemaphore> progress;

        nbl::video::IDeviceMemoryAllocator::SAllocation allocation = {};

        {
            smart_refctd_ptr<IGPUShader> shader;
            {
                IAssetLoader::SAssetLoadParams lp = {};
                lp.logger = m_logger.get();
                lp.workingDirectory = ""; // virtual root
                // this time we load a shader directly from a file
                auto assetBundle = m_assetMgr->getAsset("app_resources/emulated_float64_t_test/test.comp.hlsl", lp);
                const auto assets = assetBundle.getContents();
                if (assets.empty())
                {
                    logFail("Could not load shader!");
                    assert(0);
                }

                // It would be super weird if loading a shader from a file produced more than 1 asset
                assert(assets.size() == 1);
                smart_refctd_ptr<ICPUShader> source = IAsset::castDown<ICPUShader>(assets[0]);

                auto* compilerSet = m_assetMgr->getCompilerSet();

                nbl::asset::IShaderCompiler::SCompilerOptions options = {};
                options.stage = source->getStage();
                options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
                options.spirvOptimizer = nullptr;
                options.debugInfoFlags |= IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
                options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
                options.preprocessorOptions.logger = m_logger.get();
                options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(source->getContentType())->getDefaultIncludeFinder();

                auto spirv = compilerSet->compileToSPIRV(source.get(), options);

                ILogicalDevice::SShaderCreationParameters params{};
                params.cpushader = spirv.get();
                shader = m_device->createShader(params);
            }

            if (!shader)
                return logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

            nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
                {
                    .binding = 0,
                    .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                    .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                    .stageFlags = IGPUShader::ESS_COMPUTE,
                    .count = 1,
                    .samplers = nullptr
                }
            };
            smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings);
            if (!dsLayout)
                return logFail("Failed to create a Descriptor Layout!\n");

            smart_refctd_ptr<nbl::video::IGPUPipelineLayout> pplnLayout = m_device->createPipelineLayout({}, smart_refctd_ptr(dsLayout));
            if (!pplnLayout)
                return logFail("Failed to create a Pipeline Layout!\n");

            smart_refctd_ptr<nbl::video::IGPUComputePipeline> pipeline;
            {
                IGPUComputePipeline::SCreationParams params = {};
                params.layout = pplnLayout.get();
                params.shader.entryPoint = "main";
                params.shader.shader = shader.get();
                if (!m_device->createComputePipelines(nullptr, { &params,1 }, &pipeline))
                    return logFail("Failed to create pipelines (compile & link shaders)!\n");
            }

            smart_refctd_ptr<nbl::video::IGPUDescriptorSet> ds;

            // Allocate the memory
            {
                constexpr size_t BufferSize = sizeof(TestValues);

                nbl::video::IGPUBuffer::SCreationParams params = {};
                params.size = BufferSize;
                params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                smart_refctd_ptr<IGPUBuffer> outputBuff = m_device->createBuffer(std::move(params));
                if (!outputBuff)
                    return logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

                outputBuff->setObjectDebugName("emulated_float64_t output buffer");

                nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
                reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

                allocation = m_device->allocate(reqs, outputBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
                if (!allocation.isValid())
                    return logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

                assert(outputBuff->getBoundMemory().memory == allocation.memory.get());
                smart_refctd_ptr<nbl::video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

                ds = pool->createDescriptorSet(std::move(dsLayout));
                {
                    IGPUDescriptorSet::SDescriptorInfo info[1];
                    info[0].desc = smart_refctd_ptr(outputBuff);
                    info[0].info.buffer = { .offset = 0,.size = BufferSize };
                    IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                        {.dstSet = ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
                    };
                    m_device->updateDescriptorSets(writes, {});
                }
            }

            if (!allocation.memory->map({ 0ull,allocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
                return logFail("Failed to map the Device Memory!\n");

            uint32_t queueFamily = getComputeQueue()->getFamilyIndex();
            smart_refctd_ptr<nbl::video::IGPUCommandBuffer> cmdbuf;
            {

                smart_refctd_ptr<nbl::video::IGPUCommandPool> cmdpool = m_device->createCommandPool(queueFamily, IGPUCommandPool::CREATE_FLAGS::TRANSIENT_BIT);
                if (!cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &cmdbuf))
                    return logFail("Failed to create Command Buffers!\n");
            }

            cmdbuf->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            cmdbuf->beginDebugMarker("emulated_float64_t compute dispatch", vectorSIMDf(0, 1, 0, 1));
            cmdbuf->bindComputePipeline(pipeline.get());
            cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, pplnLayout.get(), 0, 1, &ds.get());
            cmdbuf->dispatch(WORKGROUP_SIZE, 1, 1);
            cmdbuf->endDebugMarker();
            cmdbuf->end();

            // Create the Semaphore
            constexpr auto StartedValue = 0;
            static_assert(StartedValue < FinishedValue);
            progress = m_device->createSemaphore(StartedValue);
            {
                IQueue* queue = m_device->getQueue(queueFamily, 0);

                IQueue::SSubmitInfo submitInfos[1] = {};
                const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = cmdbuf.get()} };
                submitInfos[0].commandBuffers = cmdbufs;
                const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = progress.get(),.value = FinishedValue,.stageMask = PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
                submitInfos[0].signalSemaphores = signals;

                queue->startCapture();
                queue->submit(submitInfos);
                queue->endCapture();
            }
        }

        // As the name implies this function will not progress until the fence signals or repeated waiting returns an error.
        const ISemaphore::SWaitInfo waitInfos[] = { {
            .semaphore = progress.get(),
            .value = FinishedValue
        } };
        m_device->blockForSemaphores(waitInfos);

        // if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
        const ILogicalDevice::MappedMemoryRange memoryRange(allocation.memory.get(), 0ull, allocation.memory->getAllocationSize());
        if (!allocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        double multiplicationExpectedValue = 200.0;
        assert(memoryRange.valid() && memoryRange.length >= sizeof(TestValues));
        TestValues expectedTestValues = {
            .intCreateVal = static_cast<emulated::emulated_float64_t::storage_t>(24),
            .uintCreateVal = static_cast<emulated::emulated_float64_t::storage_t>(24u),
            .uint64CreateVal = static_cast<emulated::emulated_float64_t::storage_t>(24ull),
            .floatCreateVal = static_cast<emulated::emulated_float64_t::storage_t>(1.2f),
            .doubleCreateVal = static_cast<emulated::emulated_float64_t::storage_t>(1.2),
            .additionVal = emulated::emulated_float64_t::create(30.0),
            .substractionVal = emulated::emulated_float64_t::create(10.0),
            .multiplicationVal = reinterpret_cast<uint64_t&>(multiplicationExpectedValue),
            .divisionVal = emulated::emulated_float64_t::create(2.0),
            .lessOrEqualVal = false,
            .greaterOrEqualVal = true,
            .equalVal = false,
            .notEqualVal = true,
            .lessVal = false,
            .greaterVal = true,
            .convertionToBoolVal = true,
            .convertionToIntVal = 20,
            .convertionToUint32Val = 20u,
            .convertionToUint64Val = 20ull,
            .convertionToFloatVal = 20.0f,
            .convertionToDoubleVal = 20.0,
            //.convertionToHalfVal = 20;
        };

        auto compareValues = [this](TestValues& lhs, TestValues& rhs) -> bool
            {
                bool success = true;

                if (lhs.intCreateVal != rhs.intCreateVal)
                {
                    m_logger->log("intCreateVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.intCreateVal, rhs.intCreateVal);
                    success = false;
                }
                if (lhs.uintCreateVal != rhs.uintCreateVal)
                {
                    m_logger->log("uintCreateVal not equal, expected value: %u     test value: %u", ILogger::ELL_DEBUG, lhs.uintCreateVal, rhs.uintCreateVal);
                    success = false;
                }
                if (lhs.uint64CreateVal != rhs.uint64CreateVal)
                {
                    m_logger->log("uint64CreateVal not equal, expected value: %llu     test value: %llu", ILogger::ELL_DEBUG, lhs.uint64CreateVal, rhs.uint64CreateVal);
                    success = false;
                }
                if (lhs.floatCreateVal != rhs.floatCreateVal)
                {
                    m_logger->log("floatCreateVal not equal, expected value: %f     test value: %f", ILogger::ELL_DEBUG, lhs.floatCreateVal, rhs.floatCreateVal);
                    success = false;
                }
                if (lhs.doubleCreateVal != rhs.doubleCreateVal)
                {
                    m_logger->log("doubleCreateVal not equal, expected value: %lf     test value: %lf", ILogger::ELL_DEBUG, lhs.doubleCreateVal, rhs.doubleCreateVal);
                    success = false;
                }
                if (lhs.additionVal != rhs.additionVal)
                {
                    m_logger->log("additionVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.additionVal, rhs.additionVal);
                    success = false;
                }
                if (lhs.substractionVal != rhs.substractionVal)
                {
                    m_logger->log("substractionVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.substractionVal, rhs.substractionVal);
                    success = false;
                }
                if (reinterpret_cast<double&>(lhs.multiplicationVal) != reinterpret_cast<double&>(rhs.multiplicationVal))
                {
                    m_logger->log("multiplicationVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.multiplicationVal, rhs.multiplicationVal);
                    success = false;
                }
                if (lhs.divisionVal != rhs.divisionVal)
                {
                    m_logger->log("divisionVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.divisionVal, rhs.divisionVal);
                    success = false;
                }
                if (lhs.lessOrEqualVal != rhs.lessOrEqualVal)
                {
                    m_logger->log("lessOrEqualVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.lessOrEqualVal, rhs.lessOrEqualVal);
                    success = false;
                }
                if (lhs.greaterOrEqualVal != rhs.greaterOrEqualVal)
                {
                    m_logger->log("greaterOrEqualVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.greaterOrEqualVal, rhs.greaterOrEqualVal);
                    success = false;
                }
                if (lhs.equalVal != rhs.equalVal)
                {
                    m_logger->log("equalVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.equalVal, rhs.equalVal);
                    success = false;
                }
                if (lhs.notEqualVal != rhs.notEqualVal)
                {
                    m_logger->log("notEqualVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.notEqualVal, rhs.notEqualVal);
                    success = false;
                }
                if (lhs.lessVal != rhs.lessVal)
                {
                    m_logger->log("lessVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.lessVal, rhs.lessVal);
                    success = false;
                }
                if (lhs.greaterVal != rhs.greaterVal)
                {
                    m_logger->log("greaterVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.greaterVal, rhs.greaterVal);
                    success = false;
                }
                if (lhs.convertionToBoolVal != rhs.convertionToBoolVal)
                {
                    m_logger->log("convertionToBoolVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.convertionToBoolVal, rhs.convertionToBoolVal);
                    success = false;
                }
                if (lhs.convertionToIntVal != rhs.convertionToIntVal)
                {
                    m_logger->log("convertionToIntVal not equal, expected value: %d     test value: %d", ILogger::ELL_DEBUG, lhs.convertionToIntVal, rhs.convertionToIntVal);
                    success = false;
                }
                if (lhs.convertionToUint32Val != rhs.convertionToUint32Val)
                {
                    m_logger->log("convertionToUint32Val not equal, expected value: %u     test value: %u", ILogger::ELL_DEBUG, lhs.convertionToUint32Val, rhs.convertionToUint32Val);
                    success = false;
                }
                if (lhs.convertionToUint64Val != rhs.convertionToUint64Val)
                {
                    m_logger->log("convertionToUint64Val not equal, expected value: %llu     test value: %llu", ILogger::ELL_DEBUG, lhs.convertionToUint64Val, rhs.convertionToUint64Val);
                    success = false;
                }
                if (lhs.convertionToFloatVal != rhs.convertionToFloatVal)
                {
                    m_logger->log("convertionToFloatVal not equal, expected value: %f     test value: %f", ILogger::ELL_DEBUG, lhs.convertionToFloatVal, rhs.convertionToFloatVal);
                    success = false;
                }
                if (lhs.convertionToDoubleVal != rhs.convertionToDoubleVal)
                {
                    m_logger->log("convertionToDoubleVal not equal, expected value: %lf     test value: %lf", ILogger::ELL_DEBUG, lhs.convertionToDoubleVal, rhs.convertionToDoubleVal);
                    success = false;
                }


                return success;
            };

        emulated::emulated_float64_t a = emulated::emulated_float64_t::create(20.0);
        emulated::emulated_float64_t b = emulated::emulated_float64_t::create(10.0);
        TestValues cpuTestValues = {
            .intCreateVal = emulated::emulated_float64_t::create(24),
            .uintCreateVal = emulated::emulated_float64_t::create(24u),
            .uint64CreateVal = emulated::emulated_float64_t::create(24ull),
            .floatCreateVal = emulated::emulated_float64_t::create(1.2f),
            .doubleCreateVal = emulated::emulated_float64_t::create(1.2),
            .additionVal = (a + b).data,
            .substractionVal = (a - b).data,
            .multiplicationVal = (a * b).data,
            .divisionVal = (a / b).data,
            .lessOrEqualVal = a <= b,
            .greaterOrEqualVal = a >= b,
            .equalVal = a == b,
            .notEqualVal = a != b,
            .lessVal = a < b,
            .greaterVal = a > b,
            .convertionToBoolVal = bool(a),
            .convertionToIntVal = int(a),
            .convertionToUint32Val = uint32_t(a),
            .convertionToUint64Val = uint64_t(a),
            .convertionToFloatVal = float(a),
            .convertionToDoubleVal = double(a),
            //.convertionToHalfVal = 
        };

        m_device->waitIdle();
        TestValues* gpuTestValues = static_cast<TestValues*>(memoryRange.memory->getMappedPointer());
        if (!compareValues(expectedTestValues, *gpuTestValues))
            logFail("Incorrect GPU determinated values!");
        else
            m_logger->log("Correct GPU determinated values!", ILogger::ELL_PERFORMANCE);

        if (!compareValues(expectedTestValues, cpuTestValues))
            logFail("Incorrect CPU determinated values!");
        else
            m_logger->log("Correct CPU determinated values!", ILogger::ELL_PERFORMANCE);
        
        allocation.memory->unmap();
    }
    
    template<typename... Args>
    inline bool logFail(const char* msg, Args&&... args)
    {
        m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
        return false;
    }
};

template<class T>
constexpr bool val(T a)
{
    return std::is_const_v<T>;
}

template<class T, class U> 
bool equal(T l, U r)
{
    static_assert(sizeof(T) == sizeof(U));
    return 0==memcmp(&l, &r, sizeof(T));
}


bool almost_equal(float l, float r)
{
    return fabs(l - r) < std::numeric_limits<float>::epsilon() * 1000;
}

template<class T>
constexpr auto limits_var(T obj)
{
    if constexpr (std::is_function_v<std::remove_pointer_t<T>>)
        return obj();
    else
        return obj;
}

template<class T>
T random(T lo, T hi)
{
     return (hi-lo)/RAND_MAX * rand() + lo;
}

NBL_MAIN_FUNC(CompatibilityTest)

void cpu_tests()
{
    float32_t3 a = float32_t3(1.0f, 2.0f, 3.0f);
    float32_t3 b = float32_t3(2.0f, 3.0f, 4.0f);
    b = a * 3.0f;
    bool3 asdf = bool3(true, false, true);
    pow(a, b);

    // TODO: later this whole test should be templated so we can check all `T` not just `float`, but for this we need `type_traits`
  
    // DO NOT EVER THINK TO CHANGE `using type1 = vector<type,1>` to `using type1 = type` EVER!
    static_assert(!std::is_same_v<float32_t1,float32_t>);
    static_assert(!std::is_same_v<float64_t1,float64_t>);
    static_assert(!std::is_same_v<int32_t1,int32_t>);
    static_assert(!std::is_same_v<uint32_t1,uint32_t>);
    //static_assert(!std::is_same_v<vector<T,1>,T>);

    // checking matrix memory layout
    {
        float32_t4x3 a;
        float32_t3x4 b;
        float32_t3 v;
        float32_t4 u;
        mul(a, b);
        mul(b, a);
        mul(a, v);
        mul(v, b);
        mul(u, a);
        mul(b, u);

        float32_t4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        a - a;
        b + b;
        static_assert(std::is_same_v<float32_t4x4, decltype(mul(a, b))>);
        static_assert(std::is_same_v<float32_t3x3, decltype(mul(b, a))>);
        static_assert(std::is_same_v<float32_t4, decltype(mul(a, v))>);
        static_assert(std::is_same_v<float32_t4, decltype(mul(v, b))>);
        static_assert(std::is_same_v<float32_t3, decltype(mul(u, a))>);
        static_assert(std::is_same_v<float32_t3, decltype(mul(b, u))>);

    }

    // making sure linear operators returns the correct type

    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() = float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() + float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(float32_t4x4() - float32_t4x4())>>);
    static_assert(std::is_same_v<float32_t4x4, std::remove_cvref_t<decltype(mul(float32_t4x4(), float32_t4x4()))>>);

    // checking scalar packing
    static_assert(offsetof(T, a) == 0);
    static_assert(offsetof(T, b) == offsetof(T, a) + sizeof(T::a));
    static_assert(offsetof(T, c) == offsetof(T, b) + sizeof(T::b));
    static_assert(offsetof(T, d) == offsetof(T, c) + sizeof(T::c));
    static_assert(offsetof(T, e) == offsetof(T, d) + sizeof(T::d));
    static_assert(offsetof(T, f) == offsetof(T, e) + sizeof(T::e));
    static_assert(offsetof(T, g) == offsetof(T, f) + sizeof(T::f));
    static_assert(offsetof(T, h) == offsetof(T, g) + sizeof(T::g));
    
    // use some functions
    float32_t3 x;
    float32_t2x3 y;
    float32_t3x3 z;
    //barycentric::reconstructBarycentrics(x, y);
    //barycentric::reconstructBarycentrics(x, z);
  
    // color matrix tests:
    //testColorMatrices();
    
    // promote.hlsl tests:
        // promote scalar to vector
    //float32_t3 v0 = nbl::hlsl::promote<float32_t3, float>(2.0f);
        // promote scalar to matrix
    //float32_t3x3 m0 = nbl::hlsl::promote<float32_t3x3, float>(2.0f);

        // TODO?: promote vector to matrix
    //glm::mat3 m1 = nbl::hlsl::promote<glm::mat3, glm::vec3>(glm::vec3(1.0f, 2.0f, 3.0f));

    // test vector comparison operators
    {
        /*float32_t3 a = float32_t3(1.0f, 2.0f, 3.0f);
        float32_t3 b = float32_t3(0.5f, 0.5f, 0.5f);
        assert(glm::all(a > b));
        assert(glm::all(b < a));

        b = float32_t3(0.5f, 2.0f, 0.5f);
        assert(glm::all(a >= b));
        assert(glm::all(b <= a));*/
    }

    // TODO[Przemek]: tests function output
    float32_t3 ZERO_VEC = float32_t3(0.0f, 0.0f, 0.0f);
    float32_t3 ONE_VEC = float32_t3(1.0f, 1.0f, 1.0f);

    // test functions from EOTF.hlsl
    //assert(areVectorsEqual(colorspace::eotf::identity<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::impl_shared_2_4<float32_t3>(ZERO_VEC, 0.5f), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::sRGB<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::Display_P3<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::DCI_P3_XYZ<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::SMPTE_170M<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::SMPTE_ST2084<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::HDR10_HLG<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::AdobeRGB<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::eotf::Gamma_2_2<float32_t3>(ZERO_VEC), ZERO_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::ACEScc<float32_t3>(ZERO_VEC), ZERO_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::ACEScct<float32_t3>(ZERO_VEC), ZERO_VEC));

    //assert(areVectorsEqual(colorspace::eotf::identity<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::impl_shared_2_4<float32_t3>(ONE_VEC, 0.5f), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::sRGB<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::Display_P3<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::DCI_P3_XYZ<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::SMPTE_170M<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::SMPTE_ST2084<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::HDR10_HLG<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::AdobeRGB<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::eotf::Gamma_2_2<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::ACEScc<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::eotf::ACEScct<float32_t3>(ONE_VEC), ONE_VEC));

    //// test functions from OETF.hlsl
    //assert(areVectorsEqual(colorspace::oetf::identity<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::impl_shared_2_4<float32_t3>(ZERO_VEC, 0.5f), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::sRGB<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::Display_P3<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::DCI_P3_XYZ<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::SMPTE_170M<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::SMPTE_ST2084<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::HDR10_HLG<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::AdobeRGB<float32_t3>(ZERO_VEC), ZERO_VEC));
    //assert(areVectorsEqual(colorspace::oetf::Gamma_2_2<float32_t3>(ZERO_VEC), ZERO_VEC));
    ////assert(areVectorsEqual(colorspace::oetf::ACEScc<float32_t3>(ZERO_VEC), ZERO_VEC));
    ////assert(areVectorsEqual(colorspace::oetf::ACEScct<float32_t3>(ZERO_VEC), ZERO_VEC));

    //assert(areVectorsEqual(colorspace::oetf::identity<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::impl_shared_2_4<float32_t3>(ONE_VEC, 0.5f), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::sRGB<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::Display_P3<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::oetf::DCI_P3_XYZ<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::SMPTE_170M<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::SMPTE_ST2084<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::HDR10_HLG<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::AdobeRGB<float32_t3>(ONE_VEC), ONE_VEC));
    //assert(areVectorsEqual(colorspace::oetf::Gamma_2_2<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::oetf::ACEScc<float32_t3>(ONE_VEC), ONE_VEC));
    ////assert(areVectorsEqual(colorspace::oetf::ACEScct<float32_t3>(ONE_VEC), ONE_VEC));

    // xoroshiro64 tests
    //constexpr uint32_t2 state = uint32_t2(12u, 34u);
    //Xoroshiro64Star xoroshiro64Star = Xoroshiro64Star::construct(state);
    //xoroshiro64Star();
    //Xoroshiro64StarStar xoroshiro64StarStar = Xoroshiro64StarStar::construct(state);
    //xoroshiro64StarStar();

    auto zero = cross(x,x);
    auto lenX2 = dot(x,x);
    //auto z_inv = inverse(z); //busted return type conversion
    auto mid = lerp(x,x,0.5f);
    //auto w = transpose(y); //also busted
    

    // half test
    {

        float16_t MIN = 6.103515e-05F;
        float16_t MAX = 65504.0F;
        float16_t DENORM_MIN = 5.96046448e-08F;
        uint16_t  QUIET_NAN = 0x7FFF;
        uint16_t  SIGNALING_NAN = 0x7DFF;

// TODO: reenable after port to OpenEXR 3.0
// TODO: This whole test is wrong, the uint constants should be reinterpret casted into `float16_t` not static-casted
#if 0 // disabling test, because Imath 2.4.0 doesn't provide constexpr limits, which makes the specialization of `nbl::hlsl::numeric_limits` impossible
        if(!equal((float16_t)nbl::hlsl::impl::numeric_limits<float16_t>::min, nbl::hlsl::numeric_limits<float16_t>::min()))
        {
            std::cout << "numeric_limits<float16_t>::min does not match\n";
        }
        if(!equal((float16_t)nbl::hlsl::impl::numeric_limits<float16_t>::max, nbl::hlsl::numeric_limits<float16_t>::max()))
        {
            std::cout << "numeric_limits<float16_t>::max does not match\n";
        }
        if(!equal((float16_t)nbl::hlsl::impl::numeric_limits<float16_t>::denorm_min, nbl::hlsl::numeric_limits<float16_t>::denorm_min()))
        {
            std::cout << "numeric_limits<float16_t>::denorm_min does not match\n";
        }
        if(!equal(nbl::hlsl::impl::numeric_limits<float16_t>::quiet_NaN, nbl::hlsl::numeric_limits<float16_t>::quiet_NaN()))
        {
            std::cout << "numeric_limits<float16_t>::quiet_NaN does not match\n";
        }
        if(!equal(nbl::hlsl::impl::numeric_limits<float16_t>::signaling_NaN, nbl::hlsl::numeric_limits<float16_t>::signaling_NaN()))
        {
            std::cout << "numeric_limits<float16_t>::signaling_NaN does not match\n";
        }
#endif
    }

    auto test_type_limits = []<class T>() 
    {
        using L = std::numeric_limits<T>;
        using R = nbl::hlsl::impl::numeric_limits<T>;
        
        #define TEST_AND_LOG(var) \
            { \
                auto rhs = limits_var(R::var); \
                auto lhs = limits_var(L::var); \
                if(!equal(lhs, rhs)) \
                { \
                    std::cout << typeid(T).name() << " " << #var << " does not match : " << double(lhs) << " - " << double(rhs) << "\n"; \
                } \
            }

        TEST_AND_LOG(is_specialized);
        TEST_AND_LOG(is_signed);
        TEST_AND_LOG(is_integer);
        TEST_AND_LOG(is_exact);
        TEST_AND_LOG(has_infinity);
        TEST_AND_LOG(has_quiet_NaN);
        TEST_AND_LOG(has_signaling_NaN);
        TEST_AND_LOG(has_denorm);
        TEST_AND_LOG(has_denorm_loss);
        TEST_AND_LOG(round_style);
        TEST_AND_LOG(is_iec559);
        TEST_AND_LOG(is_bounded);
        TEST_AND_LOG(is_modulo);
        TEST_AND_LOG(digits);
        TEST_AND_LOG(digits10);
        TEST_AND_LOG(max_digits10);
        TEST_AND_LOG(radix);
        TEST_AND_LOG(min_exponent);
        TEST_AND_LOG(min_exponent10);
        TEST_AND_LOG(max_exponent);
        TEST_AND_LOG(max_exponent10);
        TEST_AND_LOG(traps);
        TEST_AND_LOG(tinyness_before);
        TEST_AND_LOG(min);
        TEST_AND_LOG(max);
        TEST_AND_LOG(lowest);
        TEST_AND_LOG(epsilon);
        TEST_AND_LOG(round_error);
        TEST_AND_LOG(infinity);
        TEST_AND_LOG(quiet_NaN);
        TEST_AND_LOG(signaling_NaN);
        TEST_AND_LOG(denorm_min);
        #undef TEST_AND_LOG
    };

    test_type_limits.template operator()<float32_t>();
    test_type_limits.template operator()<float64_t>();
    test_type_limits.template operator()<int8_t>();
    test_type_limits.template operator()<int16_t>();
    test_type_limits.template operator()<int32_t>();
    test_type_limits.template operator()<int64_t>();
    test_type_limits.template operator()<uint8_t>();
    test_type_limits.template operator()<uint16_t>();
    test_type_limits.template operator()<uint32_t>();
    test_type_limits.template operator()<uint64_t>();
    test_type_limits.template operator()<bool>();

    // countl_zero test
    mpl::countl_zero<uint32_t, 5>::value;
    std::countl_zero(5u);
    nbl::hlsl::countl_zero(5u);

    // bit.hlsl test
    /*nbl::hlsl::rotl(1u, 1u);
    nbl::hlsl::rotr(1u, 1u);*/


    // cmath


#define PASS_VARS1 x0
#define PASS_VARS2 x0,x1
#define PASS_VARS3 x0,x1,x2
#define PASS_VARS(count) PASS_VARS##count


#define ASSERT_EQ(fn) \
    if (!almost_equal(lhs, rhs)) \
        std::cout << #fn << " does not match " << lhs << " vs " << rhs << '\n';

#define INIT_VARS(T) \
    T x0 = random(T(-10000), T(10000)); \
    T x1 = random(T(1), T(1000)); \
    T x2 = random(T(1), T(1000)); \

#define TEST_CMATH(fn, varcount, T) \
    {   INIT_VARS(T)\
        auto lhs = nbl::hlsl::fn(PASS_VARS(varcount)); \
        auto rhs = std::fn(PASS_VARS(varcount)); \
        ASSERT_EQ(fn); \
    }

#define TEST_CMATHT(fn, out_type, varcount, T) \
    {   INIT_VARS(T) \
        out_type o0, o1; \
        auto lhs = nbl::hlsl::fn(PASS_VARS(varcount),o0); \
        auto rhs = std::fn(PASS_VARS(varcount),&o1); \
        ASSERT_EQ(fn); \
        assert(almost_equal(o0,o1)); \
    }

#ifndef DISABLE_TGMATH_TESTS 
#define TEST_CMATH_FOR_TYPE(type) \
    TEST_CMATH(cos, 1, type) \
    TEST_CMATH(sin, 1, type) \
    TEST_CMATH(tan, 1, type) \
    TEST_CMATH(acos, 1, type) \
    TEST_CMATH(asin, 1, type) \
    TEST_CMATH(atan, 1, type) \
    TEST_CMATH(atan2, 2, type) \
    TEST_CMATH(cosh, 1, type) \
    TEST_CMATH(sinh, 1, type) \
    TEST_CMATH(tanh, 1, type) \
    TEST_CMATH(acosh, 1, type) \
    TEST_CMATH(asinh, 1, type) \
    TEST_CMATH(atanh, 1, type) \
    TEST_CMATH(exp, 1, type) \
    TEST_CMATHT(frexp, int, 1, type) \
    TEST_CMATH(ldexp, 2, type) \
    TEST_CMATH(log,1,type) \
    TEST_CMATH(log10,1,type) \
    TEST_CMATHT(modf, type, 1, type) \
    TEST_CMATH(exp2, 1, type) \
    TEST_CMATH(log2, 1, type) \
    TEST_CMATH(logb, 1, type) \
    TEST_CMATH(expm1, 1, type) \
    TEST_CMATH(log1p, 1, type) \
    TEST_CMATH(ilogb, 1, type) \
    TEST_CMATH(scalbn, 2, type) \
    TEST_CMATH(pow, 2, type) \
    TEST_CMATH(sqrt, 1, type) \
    TEST_CMATH(cbrt, 1, type) \
    TEST_CMATH(hypot, 2, type) \
    TEST_CMATH(copysign, 2, type) \
    TEST_CMATH(erf, 1, type) \
    TEST_CMATH(erfc, 1, type) \
    TEST_CMATH(tgamma, 1, type) \
    TEST_CMATH(lgamma, 1, type) \
    TEST_CMATH(ceil, 1, type) \
    TEST_CMATH(floor, 1, type) \
    TEST_CMATH(fmod, 2, type) \
    TEST_CMATH(trunc, 1, type) \
    TEST_CMATH(round, 1, type) \
    TEST_CMATH(rint, 1, type) \
    TEST_CMATH(nearbyint, 1, type) \
    TEST_CMATHT(remquo, int, 2, type) \
    TEST_CMATH(remainder, 2, type) \
    TEST_CMATH(abs, 1, type) \
    TEST_CMATH(fabs, 1, type) \
    TEST_CMATH(fma, 3, type) \
    TEST_CMATH(fmax, 2, type) \
    TEST_CMATH(fmin, 2, type) \
    TEST_CMATH(fdim, 2, type) \


    TEST_CMATH_FOR_TYPE(float32_t)
    TEST_CMATH_FOR_TYPE(float64_t)
#endif
    std::cout << "cpu tests done\n";
}
