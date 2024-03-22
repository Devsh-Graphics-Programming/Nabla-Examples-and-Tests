// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

// ERM THIS IS PROBABLY WRONG, consult Arek!
#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <assert.h>
#include <nabla.h>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/type_traits.hlsl>
#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>
//#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>
#include <nbl/builtin/hlsl/mpl.hlsl>
#include <nbl/builtin/hlsl/bit.hlsl>

    // xoroshiro tests
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>

using namespace glm;
    // colorspace tests
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
#include <nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl>

#include <nbl/builtin/hlsl/limits.hlsl>

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"


using namespace nbl;
using namespace core;
using namespace system;
using namespace asset;
using namespace video;
using namespace ui;
using namespace nbl::hlsl;

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

class CompatibilityTest final : public nbl::application_templates::MonoDeviceApplication, public nbl::application_templates::MonoAssetManagerAndBuiltinResourceApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = application_templates::MonoAssetManagerAndBuiltinResourceApplication;
public:
    CompatibilityTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        system::IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;
    
        m_queue = m_device->getQueue(0, 0);
        m_commandPool = m_device->createCommandPool(m_queue->getFamilyIndex(), nbl::video::IGPUCommandPool::ECF_RESET_COMMAND_BUFFER_BIT);
        m_device->createCommandBuffers(
            m_commandPool.get(),
            video::IGPUCommandBuffer::EL_PRIMARY,
            1,
            &m_cmdbuf);


        video::IGPUObjectFromAssetConverter CPU2GPU;
        core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
        {
            IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            auto assetBundle = m_assetMgr->getAsset("../app_resources/test.comp.hlsl", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load shader!");

            // It would be super weird if loading a shader from a file produced more than 1 asset
            assert(assets.size() == 1);
            smart_refctd_ptr<ICPUSpecializedShader> source = IAsset::castDown<ICPUSpecializedShader>(assets[0]);
            auto stage = source->getStage();
            assert(stage = IShader::ESS_COMPUTE);

            nbl::video::IGPUObjectFromAssetConverter::SParams cvtParams = {};
            cvtParams.device = m_device.get();
            cvtParams.assetManager = m_assetMgr.get();
            video::IGPUObjectFromAssetConverter CPU2GPU;
            specializedShader = CPU2GPU.getGPUObjectsFromAssets(&source, &source + 1, cvtParams)->front();
            assert(specializedShader);
        }

		const uint32_t bindingCount = 4u;
		video::IGPUDescriptorSetLayout::SBinding bindings[bindingCount] = {};
		{
			bindings[0].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[1].type = asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE;
			bindings[2].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
			bindings[3].type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER;
		}
        
        for(int i = 0; i < bindingCount; ++i)
        {
            bindings[i].stageFlags = asset::IShader::ESS_COMPUTE;
            bindings[i].count = 1;
            bindings[i].binding = i;
        }
		m_descriptorSetLayout = m_device->createDescriptorSetLayout(bindings, bindings + bindingCount);
		asset::SPushConstantRange pcRange = {};
		pcRange.stageFlags = asset::IShader::ESS_COMPUTE;
		pcRange.offset = 0u;
		pcRange.size = 2 * sizeof(uint32_t);
		auto pipelineLayout = m_device->createPipelineLayout(&pcRange, &pcRange + 1, core::smart_refctd_ptr(m_descriptorSetLayout));
        m_pipeline = m_device->createComputePipeline(nullptr, std::move(pipelineLayout), core::smart_refctd_ptr(specializedShader));

        for (int i = 0; i < 2; ++i)
        {
            m_images[i] = m_device->createImage(nbl::video::IGPUImage::SCreationParams {
                {
                    .type = nbl::video::IGPUImage::E_TYPE::ET_2D,
                    .samples = nbl::video::IGPUImage::E_SAMPLE_COUNT_FLAGS::ESCF_1_BIT,
                    .format = nbl::asset::E_FORMAT::EF_R32G32B32A32_SFLOAT,
                    .extent = { 1920,1080,1 },
                    .mipLevels = 1,
                    .arrayLayers = 1,
                    .usage = nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_STORAGE_BIT 
                        | nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT
                        | nbl::video::IGPUImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT,
                }, {}, nbl::video::IGPUImage::TILING::LINEAR,
            });

            auto reqs = m_images[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getDeviceLocalMemoryTypeBits();
            m_device->allocate(reqs, m_images[i].get());

            m_imageViews[i] = m_device->createImageView(nbl::video::IGPUImageView::SCreationParams {
                .image = m_images[i],
                    .viewType = nbl::video::IGPUImageView::E_TYPE::ET_2D,
                    .format = nbl::asset::E_FORMAT::EF_R32G32B32A32_SFLOAT,
                    // .subresourceRange = { nbl::video::IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT, 0, 1, 0, 1 },
            });

            m_buffers[i] = m_device->createBuffer(nbl::video::IGPUBuffer::SCreationParams {
                {.size = reqs.size, .usage = 
                    nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT | 
                    nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_STORAGE_BUFFER_BIT,
                }
            });

            reqs = m_buffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            m_device->allocate(reqs, m_buffers[i].get());

            m_readbackBuffers[i] = m_device->createBuffer(nbl::video::IGPUBuffer::SCreationParams {
                {.size = reqs.size, .usage = nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT | nbl::video::IGPUBuffer::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT }
            });

            reqs = m_readbackBuffers[i]->getMemoryReqs();
            reqs.memoryTypeBits &= m_device->getPhysicalDevice()->getHostVisibleMemoryTypeBits();
            m_device->allocate(reqs, m_readbackBuffers[i].get());
        }

        core::smart_refctd_ptr<video::IDescriptorPool> descriptorPool = nullptr;
        {
            video::IDescriptorPool::SCreateInfo createInfo = {};
            createInfo.maxSets = 1;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_IMAGE)] = 2;
            createInfo.maxDescriptorCount[static_cast<uint32_t>(asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER)] = 2;
            descriptorPool = m_device->createDescriptorPool(std::move(createInfo));
        }

        m_descriptorSet = descriptorPool->createDescriptorSet(core::smart_refctd_ptr(m_descriptorSetLayout));


        video::IGPUDescriptorSet::SDescriptorInfo descriptorInfos[bindingCount] = {};
        video::IGPUDescriptorSet::SWriteDescriptorSet writeDescriptorSets[bindingCount] = {};
        
        for(int i = 0; i < bindingCount; ++i)
        {
            writeDescriptorSets[i].info = &descriptorInfos[i];
            writeDescriptorSets[i].dstSet = m_descriptorSet.get();
            writeDescriptorSets[i].binding = i;
            writeDescriptorSets[i].count = bindings[i].count;
            writeDescriptorSets[i].descriptorType = bindings[i].type;

            if(i<2)
            {
                descriptorInfos[i].desc = m_imageViews[i];
                descriptorInfos[i].info.image.imageLayout = asset::IImage::EL_GENERAL;
            }
            else
            {
                descriptorInfos[i].desc = m_buffers[i-2];
                descriptorInfos[i].info.buffer.size = ~0ull;
            }
        }

        m_device->updateDescriptorSets(bindingCount, writeDescriptorSets, 0u, nullptr);
        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override
    {
        cpu_tests();

        if (m_fence)
            m_device->blockForFences(1u, &m_fence.get());
        else
            m_fence = m_device->createFence(static_cast<nbl::video::IGPUFence::E_CREATE_FLAGS>(0));

        m_device->resetFences(1u, &m_fence.get());

        m_cmdbuf->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        m_cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

        video::IGPUCommandBuffer::SImageMemoryBarrier layoutTransBarriers[2] = {
            {
            .barrier = { .srcAccessMask = static_cast<asset::E_ACCESS_FLAGS>(0), .dstAccessMask = asset::EAF_SHADER_WRITE_BIT, },
            .oldLayout = asset::IImage::EL_UNDEFINED,
            .newLayout = asset::IImage::EL_GENERAL,
            .srcQueueFamilyIndex = ~0u,
            .dstQueueFamilyIndex = ~0u,
            .image = core::smart_refctd_ptr<video::IGPUImage>(m_images[0]),
            .subresourceRange = {
                .aspectMask = asset::IImage::EAF_COLOR_BIT,
                .baseMipLevel = 0u,
                .levelCount = 1u,
                .baseArrayLayer = 0u,
                .layerCount = 1u,
                }
            }
        };

        layoutTransBarriers[1] = layoutTransBarriers[0];
        layoutTransBarriers[1].image = m_images[1];

        m_cmdbuf->pipelineBarrier(
            asset::EPSF_TOP_OF_PIPE_BIT,
            asset::EPSF_COMPUTE_SHADER_BIT,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            2u, layoutTransBarriers);

        const uint32_t pushConstants[2] = { 1920, 1080 };
        const video::IGPUDescriptorSet* set = m_descriptorSet.get();
        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->bindDescriptorSets(asset::EPBP_COMPUTE, m_pipeline->getLayout(), 0u, 1u, &set);
        m_cmdbuf->dispatch(240, 135, 1u);
        
        for (int i = 0; i < 2; ++i)
        {
            layoutTransBarriers[i].barrier.srcAccessMask = layoutTransBarriers[i].barrier.dstAccessMask;
            layoutTransBarriers[i].barrier.dstAccessMask = asset::EAF_TRANSFER_READ_BIT;
            layoutTransBarriers[i].oldLayout = layoutTransBarriers[i].newLayout;
            layoutTransBarriers[i].newLayout = asset::IImage::EL_TRANSFER_SRC_OPTIMAL;
        }

        m_cmdbuf->pipelineBarrier(
            asset::EPSF_COMPUTE_SHADER_BIT,
            asset::EPSF_TRANSFER_BIT,
            static_cast<asset::E_DEPENDENCY_FLAGS>(0u),
            0u, nullptr,
            0u, nullptr,
            2u, layoutTransBarriers);

        nbl::asset::IImage::SBufferCopy copy = {
            .imageSubresource = { 
                .aspectMask = nbl::video::IGPUImage::E_ASPECT_FLAGS::EAF_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageExtent = {1920, 1080, 1},
        };
        
        bool succ = m_cmdbuf->copyImageToBuffer(m_images[0].get(), nbl::asset::IImage::EL_TRANSFER_SRC_OPTIMAL, m_readbackBuffers[0].get(), 1, &copy);
        succ &= m_cmdbuf->copyImageToBuffer(m_images[1].get(), nbl::asset::IImage::EL_TRANSFER_SRC_OPTIMAL, m_readbackBuffers[1].get(), 1, &copy);
        assert(succ);
        m_cmdbuf->end();
        
        {
            auto cmd = m_cmdbuf.get();
            IGPUQueue::SSubmitInfo info = {
                .commandBufferCount = 1,
                .commandBuffers = &cmd,
            };
            m_queue->submit(1, &info, m_fence.get());
        }

        m_device->blockForFences(1u, &m_fence.get());
        
        using res = std::array<std::array<std::array<float, 4>, 1080>, 1920>;
        res* ptrs[4] = {};
        
        static_assert(sizeof(res) == sizeof(float) * 4 * 1920 * 1080);

        for (int i = 0; i < 4; ++i)
        {
            auto mem = (i < 2 ? m_buffers[i] : m_readbackBuffers[i-2])->getBoundMemory();
            assert(mem->isMappable());
            m_device->mapMemory(nbl::video::IDeviceMemoryAllocation::MappedMemoryRange(mem, 0, mem->getAllocationSize()));
            ptrs[i] = (res*)mem->getMappedPointer();
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
            << std::bit_cast<u32>(buf[0][0][0]) << " " 
            << std::bit_cast<u32>(buf[0][0][1]) << " "
            << std::bit_cast<u32>(buf[0][0][2]) << " "
            << std::bit_cast<u32>(buf[0][0][3]) << " "
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
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline = nullptr;
    core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> m_descriptorSetLayout;
    core::smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_descriptorSet;

    core::smart_refctd_ptr<nbl::video::IGPUImage> m_images[2];
    core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_buffers[2];
    core::smart_refctd_ptr<nbl::video::IGPUBuffer> m_readbackBuffers[2];
    core::smart_refctd_ptr<nbl::video::IGPUImageView> m_imageViews[2];
    core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf = nullptr;
    video::IGPUQueue* m_queue;
    core::smart_refctd_ptr<video::IGPUFence> m_fence = nullptr;
    nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool> m_commandPool;
    
    // core::smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
    // video::IGPUObjectFromAssetConverter::SParams m_cpu2gpuParams;

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

template<class T>
constexpr auto limits_var(T obj)
{
    if constexpr (std::is_function_v<std::remove_pointer_t<T>>)
        return obj();
    else
        return obj;
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
    float32_t3x3 z_inv = inverse(z);
    auto mid = lerp(x,x,0.5f);
    auto w = transpose(y);
    

    // half test
    {

        float16_t MIN = 6.103515e-05F;
        float16_t MAX = 65504.0F;
        float16_t DENORM_MIN = 5.96046448e-08F;
        uint16_t  QUIET_NAN = 0x7FFF;
        uint16_t  SIGNALING_NAN = 0x7DFF;

// TODO: reenable after port to OpenEXR 3.0
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

}
