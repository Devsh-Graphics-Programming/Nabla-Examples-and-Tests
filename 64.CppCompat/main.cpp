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
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>
//#include <nbl/builtin/hlsl/cpp_compat/promote.hlsl>

    // xoroshiro tests
#include <nbl/builtin/hlsl/random/xoroshiro.hlsl>
    // colorspace tests
//#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
//#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
//#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>
//#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>
using namespace glm;
//#include <nbl/builtin/glsl/colorspace/encodeCIEXYZ.glsl>
//#include <nbl/builtin/glsl/colorspace/decodeCIEXYZ.glsl>

using namespace nbl;
using namespace core;
using namespace ui;
using namespace nbl::hlsl;

#include <nbl/builtin/hlsl/bit.hlsl>

//constexpr uint32_t COLOR_MATRIX_CNT = 14u;
//const std::array<float32_t3x3, COLOR_MATRIX_CNT> hlslColorMatrices = {
//    colorspace::scRGBtoXYZ, colorspace::Display_P3toXYZ, colorspace::DCI_P3toXYZ,
//    colorspace::BT2020toXYZ, colorspace::AdobeRGBtoXYZ, colorspace::ACES2065_1toXYZ,
//    colorspace::ACEScctoXYZ, colorspace::decode::XYZtoscRGB, colorspace::decode::XYZtoDisplay_P3,
//    colorspace::decode::XYZtoDCI_P3, colorspace::decode::XYZtoBT2020, colorspace::decode::XYZtoAdobeRGB,
//    colorspace::decode::XYZtoACES2065_1, colorspace::decode::XYZtoACEScc
//};
//const std::array<glm::mat3, COLOR_MATRIX_CNT> glslColorMatrices = {
//    nbl_glsl_scRGBtoXYZ, nbl_glsl_Display_P3toXYZ, nbl_glsl_DCI_P3toXYZ,
//    nbl_glsl_BT2020toXYZ, nbl_glsl_AdobeRGBtoXYZ, nbl_glsl_ACES2065_1toXYZ,
//    nbl_glsl_ACEScctoXYZ, nbl_glsl_XYZtoscRGB, nbl_glsl_XYZtoDisplay_P3,
//    nbl_glsl_XYZtoDCI_P3, nbl_glsl_XYZtoBT2020, nbl_glsl_XYZtoAdobeRGB,
//    nbl_glsl_XYZtoACES2065_1, nbl_glsl_XYZtoACEScc
//};
//
//void testColorMatrices()
//{
//    constexpr std::array<float32_t3, 3> unitVectors = {
//        float32_t3(1.0f, 0.0f, 0.0f),
//        float32_t3(0.0f, 1.0f, 0.0f),
//        float32_t3(0.0f, 0.0f, 1.0f)
//    };
//
//    for (uint32_t matrixIdx = 0u; matrixIdx < COLOR_MATRIX_CNT; matrixIdx++)
//    {
//        const auto& hlslMatrix = hlslColorMatrices[matrixIdx];
//        const auto& glslMatrix = glslColorMatrices[matrixIdx];
//
//        for (uint32_t i = 0u; i < 3u; i++)
//        {
//            // TODO: remove when tests are done
//            std::cout << (glslMatrix[i] == mul(hlslMatrix, unitVectors[i])) << ',';
//            std::cout << (mul(hlslMatrix, unitVectors[i]) == glslMatrix * unitVectors[i]) << ',';
//
//            assert(glslMatrix[i] == mul(hlslMatrix, unitVectors[i]));
//            assert(mul(hlslMatrix, unitVectors[i]) == glslMatrix * unitVectors[i]);
//        }
//
//        std::cout << std::endl;
//    }
//}

//bool areVectorsEqual(const float32_t3& lhs, const float32_t3& rhs)
//{
//    const float32_t3 epsilonVec = float32_t3(std::exp2(-10));
//    return glm::all(glm::abs(lhs - rhs) < epsilonVec);
//}

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

#include "../common/CommonAPI.h"

class CompatibilityTest : public ApplicationBase
{
public:
    void onAppInitialized_impl() override
    {
        CommonAPI::InitParams initParams;
        initParams.windowCb = core::smart_refctd_ptr<CommonAPI::CommonAPIEventCallback>(this);
        initParams.apiType = video::EAT_VULKAN;
        initParams.appName = { "HLSL-CPP compatibility test" };
        
        auto initOutput = CommonAPI::InitWithDefaultExt(std::move(initParams));
        m_system = std::move(initOutput.system);
        m_logicalDevice = std::move(initOutput.logicalDevice);
        m_assetManager = std::move(initOutput.assetManager);
        m_cpu2gpuParams = std::move(initOutput.cpu2gpuParams);
        m_logger = std::move(initOutput.logger);
        m_queues = std::move(initOutput.queues);

        commandPools = std::move(initOutput.commandPools);
        const auto& computeCommandPools = commandPools[CommonAPI::InitOutput::EQT_COMPUTE];

        m_logicalDevice->createCommandBuffers(
            computeCommandPools[0].get(),
            video::IGPUCommandBuffer::EL_PRIMARY,
            1,
            &m_cmdbuf);

        core::smart_refctd_ptr<video::IGPUPipelineLayout> pipelineLayout =
            m_logicalDevice->createPipelineLayout();

        video::IGPUObjectFromAssetConverter CPU2GPU;
        const char* pathToShader = "D:/repos/Nabla/examples_tests/64.CppCompat/test.hlsl"; // TODO: XD
        core::smart_refctd_ptr<video::IGPUSpecializedShader> specializedShader = nullptr;
        {
            asset::IAssetLoader::SAssetLoadParams params = {};
            params.logger = m_logger.get();
            auto specShader_cpu = core::smart_refctd_ptr_static_cast<asset::ICPUSpecializedShader>(*m_assetManager->getAsset(pathToShader, params).getContents().begin());
            specializedShader = CPU2GPU.getGPUObjectsFromAssets(&specShader_cpu, &specShader_cpu + 1, m_cpu2gpuParams)->front();
        }
        assert(specializedShader);

        m_pipeline = m_logicalDevice->createComputePipeline(nullptr,
            core::smart_refctd_ptr(pipelineLayout), core::smart_refctd_ptr(specializedShader));

        m_semaphores[0] = m_logicalDevice->createSemaphore();
        m_semaphores[1] = m_logicalDevice->createSemaphore();
    }

    void onAppTerminated_impl() override
    {
        m_logicalDevice->waitIdle();
    }

    void workLoopBody() override
    {
        waitForFrame(1u, m_fence);
        m_logicalDevice->resetFences(1u, &m_fence.get());

        m_cmdbuf->reset(video::IGPUCommandBuffer::ERF_RELEASE_RESOURCES_BIT);
        m_cmdbuf->begin(video::IGPUCommandBuffer::EU_ONE_TIME_SUBMIT_BIT);

        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->dispatch(1u, 1u, 1u);
        m_cmdbuf->end();

            //TODO: fix semaphores
        CommonAPI::Submit(
            m_logicalDevice.get(),
            m_cmdbuf.get(),
            m_queues[CommonAPI::InitOutput::EQT_COMPUTE],
            m_semaphores[0].get(),
            m_semaphores[1].get(),
            m_fence.get());

        m_keepRunning = false;
    }

    bool keepRunning() override
    {
        return m_keepRunning;
    }

    static void runTests(int argc, char** argv)
    {
        CommonAPI::main<CompatibilityTest>(argc, argv);
    }

    void setWindow(core::smart_refctd_ptr<nbl::ui::IWindow>&& wnd) override {}
    void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) override { m_system = std::move(system); }
    void setSurface(core::smart_refctd_ptr<video::ISurface>&& s) override {}
    void setFBOs(std::vector<core::smart_refctd_ptr<video::IGPUFramebuffer>>& f) override {}
    void setSwapchain(core::smart_refctd_ptr<video::ISwapchain>&& s) override {}
    nbl::ui::IWindow* getWindow() override { return nullptr; }
    video::IAPIConnection* getAPIConnection() override { return nullptr; }
    video::ILogicalDevice* getLogicalDevice()  override { return m_logicalDevice.get(); }
    video::IGPURenderpass* getRenderpass() override { return nullptr; }
    uint32_t getSwapchainImageCount() override { return 0u; }
    virtual nbl::asset::E_FORMAT getDepthFormat() override { return nbl::asset::E_FORMAT::EF_UNKNOWN;  }
    APP_CONSTRUCTOR(CompatibilityTest);

private:
    core::smart_refctd_ptr<nbl::system::ISystem> m_system;
    core::smart_refctd_ptr<nbl::video::ILogicalDevice> m_logicalDevice;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline = nullptr;
    core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf = nullptr;
    std::array<video::IGPUQueue*, CommonAPI::InitOutput::MaxQueuesCount> m_queues;
    core::smart_refctd_ptr<video::IGPUSemaphore> m_semaphores[2u] = {nullptr};
    core::smart_refctd_ptr<video::IGPUFence> m_fence = nullptr;
    std::array<std::array<nbl::core::smart_refctd_ptr<nbl::video::IGPUCommandPool>, CommonAPI::InitOutput::MaxFramesInFlight>, CommonAPI::InitOutput::MaxQueuesCount> commandPools;
    core::smart_refctd_ptr<nbl::asset::IAssetManager> m_assetManager;
    video::IGPUObjectFromAssetConverter::SParams m_cpu2gpuParams;
    core::smart_refctd_ptr<nbl::system::ILogger> m_logger;

    bool m_keepRunning = true;
};

int main(int argc, char** argv)
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
    constexpr uint32_t2 state = uint32_t2(12u, 34u);
    Xoroshiro64Star xoroshiro64Star = Xoroshiro64Star::construct(state);
    xoroshiro64Star();
    Xoroshiro64StarStar xoroshiro64StarStar = Xoroshiro64StarStar::construct(state);
    xoroshiro64StarStar();
    
    //CompatibilityTest::runTests(argc, argv);

    auto zero = cross(x,x);
    auto lenX2 = dot(x,x);
    float32_t3x3 z_inv = inverse(z);
    auto mid = lerp(x,x,0.5f);
    auto w = transpose(y);
}
