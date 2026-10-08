// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "app_resources/common.hlsl"

#include "CTgmathTester.h"
#include "CIntrinsicsTester.h"

#include "nbl/builtin/hlsl/ieee754.hlsl"
#include "nbl/builtin/hlsl/emulated/float64_t.hlsl"
#include "nbl/builtin/hlsl/approx/abs_rel.hlsl"
#include "nbl/builtin/hlsl/approx/ulp.hlsl"
#include "nbl/builtin/hlsl/approx/vector.hlsl"
#include "nbl/builtin/hlsl/utils/elementwise.hlsl"

#include <iostream>
#include <cstdio>
#include <assert.h>


using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::examples;

//using namespace glm;

void cpu_tests();
void ieee754_tests();
void approx_tests();

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

class CompatibilityTest final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
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

        // before the GPU testers, which end the app early when they fail
        cpu_tests();

        bool pass = true;
        {
            CTgmathTester::PipelineSetupData pplnSetupData;
            pplnSetupData.device = m_device;
            pplnSetupData.api = m_api;
            pplnSetupData.assetMgr = m_assetMgr;
            pplnSetupData.logger = m_logger;
            pplnSetupData.physicalDevice = m_physicalDevice;
            pplnSetupData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            pplnSetupData.shaderKey = nbl::this_example::builtin::build::get_spirv_key<"tgmathTest">(m_device.get());

            CTgmathTester tgmathTester(8);
            tgmathTester.setupPipeline(pplnSetupData);
            pass &= tgmathTester.performTestsAndVerifyResults("TgmathTestLog.txt");
        }
        {
            CIntrinsicsTester::PipelineSetupData pplnSetupData;
            pplnSetupData.device = m_device;
            pplnSetupData.api = m_api;
            pplnSetupData.assetMgr = m_assetMgr;
            pplnSetupData.logger = m_logger;
            pplnSetupData.physicalDevice = m_physicalDevice;
            pplnSetupData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            pplnSetupData.shaderKey = nbl::this_example::builtin::build::get_spirv_key<"intrinsicsTest">(m_device.get());

            CIntrinsicsTester intrinsicsTester(8);
            intrinsicsTester.setupPipeline(pplnSetupData);
            pass &= intrinsicsTester.performTestsAndVerifyResults("IntrinsicsTestLog.txt");
        }
        if (!pass)
            return false;

        m_queue = m_device->getQueue(0, 0);
        m_commandPool = m_device->createCommandPool(m_queue->getFamilyIndex(), IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        m_commandPool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, { &m_cmdbuf,1 }, smart_refctd_ptr(m_logger));

        smart_refctd_ptr<IShader> shader;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = "app_resources"; // virtual root
            auto key = nbl::this_example::builtin::build::get_spirv_key<"test">(m_device.get());
            auto assetBundle = m_assetMgr->getAsset(key.data(), lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load shader!");

            auto source = IAsset::castDown<IShader>(assets[0]);
            // The down-cast should not fail!
            assert(source);

            // this time we skip the use of the asset converter since the ICPUShader->IGPUShader path is quick and simple
            shader = m_device->compileShader({ source.get() });
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
            bindings[i].stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
            bindings[i].count = 1;
            bindings[i].binding = i;
        }
		m_descriptorSetLayout = m_device->createDescriptorSetLayout(bindings);
        {
		    SPushConstantRange pcRange = {};
		    pcRange.stageFlags = IShader::E_SHADER_STAGE::ESS_COMPUTE;
		    pcRange.offset = 0u;
		    pcRange.size = 2 * sizeof(uint32_t);
            auto layout = m_device->createPipelineLayout({ &pcRange,1 }, smart_refctd_ptr(m_descriptorSetLayout));
            IGPUComputePipeline::SCreationParams params = {};
            params.layout = layout.get();
            params.shader.shader = shader.get();
            params.shader.entryPoint = "main";
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
            m_device->allocate(reqs, { m_images[i].get() });

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
            m_device->allocate(reqs, { m_buffers[i].get() });

            m_readbackBuffers[i] = m_device->createBuffer(IGPUBuffer::SCreationParams {
                {.size = reqs.size, .usage = IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT }
            });

            reqs = m_readbackBuffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            m_device->allocate(reqs, { m_readbackBuffers[i].get() });
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
        //    m_api->startCapture();
        //    m_queue->submit(submitInfos);  //Command buffer is NOT IN THE EXECUTABLE STATE
        //    m_api->endCapture();
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
            m_api->startCapture();
            m_queue->submit(submitInfos);
            m_api->endCapture();
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
    auto mid = nbl::hlsl::mix(x,x,float32_t3(0.5f));
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
    // TODO: fix warning about nodiscard
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


    //TEST_CMATH_FOR_TYPE(float32_t)
    //TEST_CMATH_FOR_TYPE(float64_t)
#endif
    ieee754_tests();
    approx_tests();
    std::cout << "cpu tests done\n";
}

// `ieee754` bit pattern helpers: uint in gives uint out, float in gives float out
void ieee754_tests()
{
    auto check = [](const bool condition, const char* type, const char* what)
    {
        if (!condition)
            std::cout << "ieee754 test failed [" << type << "]: " << what << '\n';
    };

    auto testType = []<typename F>(auto& check, const char* type)
    {
        using AsUint = hlsl::unsigned_integer_of_size_t<sizeof(F)>;
        using traits_t = hlsl::ieee754::traits<F>;
        auto bits = [](const F x) { return hlsl::ieee754::impl::bitCastToUintType(x); };
        auto fromBits = [](const AsUint b) { return hlsl::ieee754::impl::castBackToFloatType<AsUint>(b); };
        auto sameBits = [&](const F a, const F b) { return bits(a) == bits(b); };
        auto biased = [](const int exponent) { return AsUint(traits_t::exponentBias + exponent); };

        const F one = F(1.0);
        const F oneUp = fromBits(AsUint(bits(one) + AsUint(1)));
        const F posZero = fromBits(AsUint(0));
        const F negZero = fromBits(AsUint(traits_t::signMask));
        const F inf = fromBits(AsUint(traits_t::inf));
        const F largest = fromBits(AsUint(traits_t::inf - AsUint(1)));
        const F nanA = fromBits(AsUint(traits_t::quietNaN));
        const F nanB = fromBits(AsUint(traits_t::quietNaN | AsUint(1)));
        const F posDenormMin = fromBits(AsUint(1));
        const F negDenormMin = fromBits(AsUint(traits_t::signMask | AsUint(1)));

        // bit casts round trip
        static_assert(std::is_same_v<decltype(hlsl::ieee754::impl::castBackToFloatType<AsUint>(AsUint(0))), F>);
        check(sameBits(fromBits(bits(F(-3.5))), F(-3.5)), type, "castBackToFloatType<AsUint>(bitCastToUintType(x)) == x");

        // replaceBiasedExponent, float version
        static_assert(std::is_same_v<decltype(hlsl::ieee754::replaceBiasedExponent(F(0), AsUint(0))), F>);
        check(sameBits(hlsl::ieee754::replaceBiasedExponent(F(3.0), biased(3)), F(12.0)), type, "replaceBiasedExponent(3, bias+3) == 12");
        check(sameBits(hlsl::ieee754::replaceBiasedExponent(F(-3.0), biased(0)), F(-1.5)), type, "replaceBiasedExponent(-3, bias) == -1.5 (sign and mantissa kept)");
        check(sameBits(hlsl::ieee754::replaceBiasedExponent(one, AsUint(traits_t::specialValueExp)), inf), type, "replaceBiasedExponent(1, all ones) == inf");
        check(sameBits(hlsl::ieee754::replaceBiasedExponent(one, AsUint(0)), posZero), type, "replaceBiasedExponent(1, 0) == +0");

        // replaceBiasedExponent, bit pattern version
        static_assert(std::is_same_v<decltype(hlsl::ieee754::replaceBiasedExponent(AsUint(0), AsUint(0))), AsUint>);
        check(hlsl::ieee754::replaceBiasedExponent(bits(F(3.0)), biased(3)) == bits(F(12.0)), type, "replaceBiasedExponent(bits(3), bias+3) == bits(12)");
        check(hlsl::ieee754::replaceBiasedExponent(bits(F(-3.0)), biased(0)) == bits(F(-1.5)), type, "replaceBiasedExponent(bits(-3), bias) == bits(-1.5)");
        check(hlsl::ieee754::replaceBiasedExponent(bits(F(3.0)), AsUint(~AsUint(0))) == AsUint(bits(F(3.0)) | traits_t::exponentMask), type, "replaceBiasedExponent drops biasedExp bits that don't fit the exponent");
        check(hlsl::ieee754::replaceBiasedExponent(AsUint(~AsUint(0)), AsUint(0)) == AsUint(~traits_t::exponentMask), type, "replaceBiasedExponent(all ones, 0) only clears the exponent");

        // fastMulExp2 goes through replaceBiasedExponent
        static_assert(std::is_same_v<decltype(hlsl::ieee754::fastMulExp2(F(0), 0)), F>);
        static_assert(std::is_same_v<decltype(hlsl::ieee754::fastMulExp2(AsUint(0), 0)), AsUint>);
        check(sameBits(hlsl::ieee754::fastMulExp2(F(3.0), 2), F(12.0)), type, "fastMulExp2(3, 2) == 12");
        check(sameBits(hlsl::ieee754::fastMulExp2(F(12.0), -2), F(3.0)), type, "fastMulExp2(12, -2) == 3");
        check(sameBits(hlsl::ieee754::fastMulExp2(F(-0.75), 3), F(-6.0)), type, "fastMulExp2(-0.75, 3) == -6");
        check(hlsl::ieee754::fastMulExp2(bits(F(3.0)), 2) == bits(F(12.0)), type, "fastMulExp2(bits(3), 2) == bits(12)");

        // nextDown, nextTowardZero
        check(sameBits(hlsl::ieee754::nextDown(one), fromBits(AsUint(bits(one) - AsUint(1)))), type, "nextDown(1) is one ULP below 1");
        check(sameBits(hlsl::ieee754::nextDown(F(-1.0)), fromBits(AsUint(bits(F(-1.0)) + AsUint(1)))), type, "nextDown(-1) is one ULP below -1");
        check(sameBits(hlsl::ieee754::nextTowardZero(one), hlsl::ieee754::nextDown(one)), type, "nextTowardZero(1) == nextDown(1)");

        // ulpDistance
        check(hlsl::ieee754::ulpDistance(one, oneUp) == AsUint(1), type, "ulpDistance(1, next after 1) == 1");
        check(hlsl::ieee754::ulpDistance(oneUp, one) == AsUint(1), type, "ulpDistance is symmetric");
        check(hlsl::ieee754::ulpDistance(posZero, negZero) == AsUint(0), type, "ulpDistance(+0, -0) == 0");
        check(hlsl::ieee754::ulpDistance(negDenormMin, posDenormMin) == AsUint(2), type, "ulpDistance(-denorm_min, +denorm_min) == 2");
        check(hlsl::ieee754::ulpDistance(largest, inf) == AsUint(1), type, "ulpDistance(largest, inf) == 1");
        check(hlsl::ieee754::ulpDistance(nanA, nanB) == AsUint(1), type, "ulpDistance between NaN payloads == 1");
    };
    testType.template operator()<float16_t>(check, "float16_t");
    testType.template operator()<float32_t>(check, "float32_t");
    testType.template operator()<float64_t>(check, "float64_t");

    // emulated_float64_t goes through the `uint64_t` bit pattern versions
    {
        using emulated_t = hlsl::emulated_float64_t<true, true>;
        const uint64_t biasedExp3 = uint64_t(hlsl::ieee754::traits<float64_t>::exponentBias + 3);
        check(hlsl::ieee754::replaceBiasedExponent(emulated_t::create(3.0), biasedExp3).data == std::bit_cast<uint64_t>(12.0), "emulated_float64_t", "replaceBiasedExponent(3, bias+3) == 12");
        check(hlsl::ieee754::fastMulExp2(emulated_t::create(3.0), 2).data == std::bit_cast<uint64_t>(12.0), "emulated_float64_t", "fastMulExp2(3, 2) == 12");
        check(hlsl::ieee754::fastMulExp2(emulated_t::create(-12.0), -2).data == std::bit_cast<uint64_t>(-3.0), "emulated_float64_t", "fastMulExp2(-12, -2) == -3");
    }
}

// `approx::` comparators and `utils::elementwise*` testers
void approx_tests()
{
    auto check = [](const bool condition, const char* type, const char* what)
    {
        if (!condition)
            std::cout << "approx test failed [" << type << "]: " << what << '\n';
    };

    auto scalarTests = []<typename F>(auto& check, const char* type)
    {
        using AsUint = hlsl::unsigned_integer_of_size_t<sizeof(F)>;
        using traits_t = hlsl::ieee754::traits<F>;
        auto fromBits = [](const AsUint bits) { return hlsl::ieee754::impl::castBackToFloatType<AsUint>(bits); };
        auto sameBits = [](const F a, const F b) { return hlsl::ieee754::impl::bitCastToUintType(a) == hlsl::ieee754::impl::bitCastToUintType(b); };

        const F one = F(1.0);
        const F oneUp = fromBits(AsUint(hlsl::ieee754::impl::bitCastToUintType(one) + AsUint(1)));
        const F posZero = fromBits(AsUint(0));
        const F negZero = fromBits(AsUint(traits_t::signMask));
        const F inf = fromBits(AsUint(traits_t::inf));
        const F negInf = fromBits(AsUint(traits_t::inf | traits_t::signMask));
        const F largest = fromBits(AsUint(traits_t::inf - AsUint(1)));
        const F nanA = fromBits(AsUint(traits_t::quietNaN));
        const F nanB = fromBits(AsUint(traits_t::quietNaN | AsUint(1)));

        // hlsl::approx::ulpEqual
        check(hlsl::approx::ulpEqual(one, oneUp, AsUint(1)), type, "ulpEqual(1, next after 1, 1)");
        check(!hlsl::approx::ulpEqual(one, oneUp, AsUint(0)), type, "!ulpEqual(1, next after 1, 0)");
        check(hlsl::approx::ulpEqual(one, hlsl::ieee754::nextDown(one), AsUint(1)), type, "ulpEqual(1, nextDown(1), 1)");
        check(hlsl::approx::ulpEqual(posZero, negZero, AsUint(0)), type, "ulpEqual(+0, -0, 0)");
        check(!hlsl::approx::ulpEqual(nanA, nanB, AsUint(~AsUint(0))), type, "!ulpEqual(NaN, NaN', max)");
        check(!hlsl::approx::ulpEqual(nanA, nanA, AsUint(~AsUint(0))), type, "!ulpEqual(NaN, NaN, max)");
        check(!hlsl::approx::ulpEqual(inf, largest, AsUint(1)), type, "!ulpEqual(inf, largest, 1)");
        check(hlsl::approx::ulpEqual(inf, inf, AsUint(0)), type, "ulpEqual(inf, inf, 0)");

        // hlsl::approx::absRelEqual
        check(hlsl::approx::absRelEqual(F(1e-5), posZero, F(1e-4), F(1e-4)), type, "absRelEqual(1e-5, 0, 1e-4, 1e-4)");
        check(hlsl::approx::absRelEqual(posZero, F(1e-5), F(1e-4), F(0)), type, "absRelEqual(0, 1e-5, 1e-4, 0)");
        check(!hlsl::approx::absRelEqual(one, -one, F(0), F(0.1)), type, "!absRelEqual(1, -1, 0, 0.1)");
        check(hlsl::approx::absRelEqual(F(100.0), F(100.5), F(0), F(0.01)), type, "absRelEqual(100, 100.5, 0, 0.01)");
        check(!hlsl::approx::absRelEqual(F(100.0), F(102.0), F(0), F(0.01)), type, "!absRelEqual(100, 102, 0, 0.01)");
        check(hlsl::approx::absRelEqual(posZero, negZero, F(0), F(0)), type, "absRelEqual(+0, -0, 0, 0)");
        check(hlsl::approx::absRelEqual(inf, inf, F(0), F(0)), type, "absRelEqual(inf, inf, 0, 0)");
        check(!hlsl::approx::absRelEqual(inf, one, F(0), F(1)), type, "!absRelEqual(inf, 1, 0, 1)");
        check(!hlsl::approx::absRelEqual(inf, negInf, F(1), F(1)), type, "!absRelEqual(inf, -inf, 1, 1)");
        // IEEE semantics, NaN is never equal (may not hold for CPU builds with `/fp:fast` or `-ffast-math`)
        check(!hlsl::approx::absRelEqual(nanA, nanA, F(1), F(1)), type, "!absRelEqual(NaN, NaN, 1, 1)");
        check(!hlsl::approx::absRelEqual(nanA, one, F(1), F(1)), type, "!absRelEqual(NaN, 1, 1, 1)");
    };
    scalarTests.template operator()<float16_t>(check, "float16_t");
    scalarTests.template operator()<float32_t>(check, "float32_t");
    scalarTests.template operator()<float64_t>(check, "float64_t");

    // probe 9a from 09_GeometryCreator: a drifted-but-orthogonal dot product vs exact zero
    check(hlsl::approx::absRelEqual<float64_t>(1e-17, 0.0, 1e-4, 1e-4), "float64_t", "absRelEqual(1e-17, 0, 1e-4, 1e-4)");

    auto compositeTests = []<typename F>(auto& check, const char* type)
    {
        using vec_t = hlsl::vector<F, 3>;
        using mat_t = hlsl::matrix<F, 3, 3>;
        using pred_t = hlsl::approx::AbsRelEqualPred<F>;
        const pred_t pred = pred_t::create(F(0.1), F(0));

        const vec_t v = vec_t(F(1), F(2), F(3));
        const vec_t vOneOff = vec_t(F(1), F(2), F(3.5));
        const vec_t vAllOff = vec_t(F(11), F(12), F(13));
        check(hlsl::approx::absRelEqual(v, v, F(0.1), F(0)), type, "vector absRelEqual(v, v)");
        check(!hlsl::approx::absRelEqual(v, vOneOff, F(0.1), F(0)), type, "vector !absRelEqual with one element off");
        check(!hlsl::approx::ulpEqual(v, vOneOff, hlsl::unsigned_integer_of_size_t<sizeof(F)>(1)), type, "vector !ulpEqual with one element off");
        check(!hlsl::utils::elementwiseAll(v, vOneOff, pred), type, "vector !elementwiseAll with one element off");
        check(hlsl::utils::elementwiseAny(v, vOneOff, pred), type, "vector elementwiseAny with one element off");
        check(!hlsl::utils::elementwiseNone(v, vOneOff, pred), type, "vector !elementwiseNone with one element off");
        check(!hlsl::utils::elementwiseAny(v, vAllOff, pred), type, "vector !elementwiseAny with all elements off");
        check(hlsl::utils::elementwiseNone(v, vAllOff, pred), type, "vector elementwiseNone with all elements off");

        // isPerpendicular with |cos(theta)| <= 1e-5
        const F cosThetaEpsilon = F(1e-5);
        const vec_t x = vec_t(F(1), F(0), F(0));
        const vec_t y = vec_t(F(0), F(1), F(0));
        const vec_t yTiltedWithin = vec_t(F(5e-6), F(1), F(0)); // cos(theta) ~= 5e-6
        const vec_t yTiltedBeyond = vec_t(F(2e-5), F(1), F(0)); // cos(theta) ~= 2e-5
        check(hlsl::approx::isPerpendicular(x, y, cosThetaEpsilon), type, "isPerpendicular(x, y)");
        check(hlsl::approx::isPerpendicular(x, yTiltedWithin, cosThetaEpsilon), type, "isPerpendicular within epsilon");
        check(!hlsl::approx::isPerpendicular(x, yTiltedBeyond, cosThetaEpsilon), type, "!isPerpendicular beyond epsilon");
        check(hlsl::approx::isPerpendicular(x * F(100), yTiltedWithin * F(0.01), cosThetaEpsilon), type, "isPerpendicular within epsilon, scaled");
        check(!hlsl::approx::isPerpendicular(x * F(100), yTiltedBeyond * F(0.01), cosThetaEpsilon), type, "!isPerpendicular beyond epsilon, scaled");
        check(!hlsl::approx::isPerpendicular(x, x, cosThetaEpsilon), type, "!isPerpendicular(x, x)");
        check(hlsl::approx::isPerpendicular(vec_t(F(0), F(0), F(0)), x, cosThetaEpsilon), type, "zero vector isPerpendicular to everything");

        const mat_t m = mat_t(F(1));
        mat_t mOneOff = m;
        mOneOff[2][1] = F(0.5);
        check(hlsl::approx::absRelEqual(m, m, F(0.1), F(0)), type, "matrix absRelEqual(m, m)");
        check(!hlsl::approx::absRelEqual(m, mOneOff, F(0.1), F(0)), type, "matrix !absRelEqual with one element off");
        check(!hlsl::utils::elementwiseAll(m, mOneOff, pred), type, "matrix !elementwiseAll with one element off");
        check(hlsl::utils::elementwiseAny(m, mOneOff, pred), type, "matrix elementwiseAny with one element off");
        check(!hlsl::utils::elementwiseNone(m, mOneOff, pred), type, "matrix !elementwiseNone with one element off");
        check(!hlsl::utils::elementwiseNone(m, mat_t(F(10)), pred), type, "matrix !elementwiseNone with only the diagonal off (off-diagonal zeros still match)");
        check(hlsl::utils::elementwiseNone(m, m + mat_t(F(10)) + mat_t(hlsl::vector<F, 3>(F(10)), hlsl::vector<F, 3>(F(10)), hlsl::vector<F, 3>(F(10))), pred), type, "matrix elementwiseNone with all elements off");
    };
    compositeTests.template operator()<float32_t>(check, "float32_t");
    compositeTests.template operator()<float64_t>(check, "float64_t");
}
