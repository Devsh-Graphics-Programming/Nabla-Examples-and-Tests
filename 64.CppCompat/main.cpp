// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#define _NBL_STATIC_LIB_
#include <iostream>
#include <cstdio>
#include <assert.h>
#include <nabla.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
#include <nbl/builtin/hlsl/barycentric/utils.hlsl>

    // colorspace
#include <nbl/builtin/hlsl/colorspace/encodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/decodeCIEXYZ.hlsl>
#include <nbl/builtin/hlsl/colorspace/OETF.hlsl>
#include <nbl/builtin/hlsl/colorspace/EOTF.hlsl>

using namespace nbl;
using namespace core;
using namespace ui;
using namespace hlsl;

// encodeCIEXYZ.hlsl matrices
constexpr glm::mat3 nbl_glsl_scRGBtoXYZ = glm::mat3(
    glm::vec3(0.412391f, 0.212639f, 0.019331f),
    glm::vec3(0.357584f, 0.715169f, 0.119195f),
    glm::vec3(0.180481f, 0.072192f, 0.950532f)
);

constexpr glm::mat3 nbl_glsl_Display_P3toXYZ = glm::mat3(
    glm::vec3(0.4865709486f, 0.2289745641f, 0.0000000000f),
    glm::vec3(0.2656676932f, 0.6917385218f, 0.0451133819f),
    glm::vec3(0.1982172852f, 0.0792869141f, 1.0439443689f)
);

constexpr glm::mat3 nbl_glsl_DCI_P3toXYZ = glm::mat3(
    glm::vec3(1.0f, 0.0f, 0.0f),
    glm::vec3(0.0f, 1.0f, 0.0f),
    glm::vec3(0.0f, 0.0f, 1.0f)
);

constexpr glm::mat3 nbl_glsl_BT2020toXYZ = glm::mat3(
    glm::vec3(0.636958f, 0.262700f, 0.000000f),
    glm::vec3(0.144617f, 0.677998f, 0.028073f),
    glm::vec3(0.168881f, 0.059302f, 1.060985f)
);

constexpr glm::mat3 nbl_glsl_AdobeRGBtoXYZ = glm::mat3(
    glm::vec3(0.5766690429f, 0.2973449753f, 0.0270313614f),
    glm::vec3(0.1855582379f, 0.6273635663f, 0.0706888525f),
    glm::vec3(0.1882286462f, 0.0752914585f, 0.9913375368f)
);

constexpr glm::mat3 nbl_glsl_ACES2065_1toXYZ = glm::mat3(
    glm::vec3(0.9525523959f, 0.3439664498f, 0.0000000000f),
    glm::vec3(0.0000000000f, 0.7281660966f, 0.0000000000f),
    glm::vec3(0.0000936786f, -0.0721325464f, 1.0088251844f)
);

constexpr glm::mat3 nbl_glsl_ACEScctoXYZ = glm::mat3(
    glm::vec3(0.6624541811f, 0.2722287168f, -0.0055746495f),
    glm::vec3(0.1340042065f, 0.6740817658f, 0.0040607335f),
    glm::vec3(0.1561876870f, 0.0536895174f, 1.0103391003f)
);

// decodeCIEXYZ.hlsl matrices
constexpr glm::mat3 nbl_glsl_XYZtoscRGB = glm::mat3(
    glm::vec3(3.240970f, -0.969244f, 0.055630f),
    glm::vec3(-1.537383f, 1.875968f, -0.203977f),
    glm::vec3(-0.498611f, 0.041555f, 1.056972f)
);

constexpr glm::mat3 nbl_glsl_XYZtoDisplay_P3 = glm::mat3(
    glm::vec3(2.4934969119f, -0.8294889696f, 0.0358458302f),
    glm::vec3(-0.9313836179f, 1.7626640603f, -0.0761723893f),
    glm::vec3(-0.4027107845f, 0.0236246858f, 0.9568845240f)
);

constexpr glm::mat3 nbl_glsl_XYZtoDCI_P3 = glm::mat3(
    glm::vec3(1.0, 0.0, 0.0),
    glm::vec3(0.0, 1.0, 0.0),
    glm::vec3(0.0, 0.0, 1.0)
);

constexpr glm::mat3 nbl_glsl_XYZtoBT2020 = glm::mat3(
    glm::vec3(1.716651f, -0.666684f, 0.017640f),
    glm::vec3(-0.355671f, 1.616481f, -0.042771f),
    glm::vec3(-0.253366f, 0.015769f, 0.942103f)
);

constexpr glm::mat3 nbl_glsl_XYZtoAdobeRGB = glm::mat3(
    glm::vec3(2.0415879038f, -0.9692436363f, 0.0134442806f),
    glm::vec3(-0.5650069743f, 1.8759675015f, -0.1183623922f),
    glm::vec3(-0.3447313508f, 0.0415550574f, 1.0151749944f)
);

constexpr glm::mat3 nbl_glsl_XYZtoACES2065_1 = glm::mat3(
    glm::vec3(1.0498110175f, -0.4959030231f, 0.0000000000f),
    glm::vec3(0.0000000000f, 1.3733130458f, 0.0000000000f),
    glm::vec3(-0.0000974845f, 0.0982400361f, 0.9912520182f)
);

constexpr glm::mat3 nbl_glsl_XYZtoACEScc = glm::mat3(
    glm::vec3(1.6410233797f, -0.6636628587f, 0.0117218943f),
    glm::vec3(-0.3248032942f, 1.6153315917f, -0.0082844420f),
    glm::vec3(-0.2364246952f, 0.0167563477f, 0.9883948585f)
);

constexpr uint32_t COLOR_MATRIX_CNT = 14u;
constexpr std::array<float3x3, COLOR_MATRIX_CNT> hlslColorMatrices = {
    colorspace::scRGBtoXYZ, colorspace::Display_P3toXYZ, colorspace::DCI_P3toXYZ,
    colorspace::BT2020toXYZ, colorspace::AdobeRGBtoXYZ, colorspace::ACES2065_1toXYZ,
    colorspace::ACEScctoXYZ, colorspace::decode::XYZtoscRGB, colorspace::decode::XYZtoDisplay_P3,
    colorspace::decode::XYZtoDCI_P3, colorspace::decode::XYZtoBT2020, colorspace::decode::XYZtoAdobeRGB,
    colorspace::decode::XYZtoACES2065_1, colorspace::decode::XYZtoACEScc
};
constexpr std::array<glm::mat3, COLOR_MATRIX_CNT> glslColorMatrices = {
    nbl_glsl_scRGBtoXYZ, nbl_glsl_Display_P3toXYZ, nbl_glsl_DCI_P3toXYZ,
    nbl_glsl_BT2020toXYZ, nbl_glsl_AdobeRGBtoXYZ, nbl_glsl_ACES2065_1toXYZ,
    nbl_glsl_ACEScctoXYZ, nbl_glsl_XYZtoscRGB, nbl_glsl_XYZtoDisplay_P3,
    nbl_glsl_XYZtoDCI_P3, nbl_glsl_XYZtoBT2020, nbl_glsl_XYZtoAdobeRGB,
    nbl_glsl_XYZtoACES2065_1, nbl_glsl_XYZtoACEScc
};

void testColorMatrices()
{
    constexpr std::array<float3, 3> unitVectors = {
        float3(1.0f, 0.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 0.0f, 1.0f)
    };

    for (uint32_t matrixIdx = 0u; matrixIdx < COLOR_MATRIX_CNT; matrixIdx++)
    {
        const auto& hlslMatrix = hlslColorMatrices[matrixIdx];
        const auto& glslMatrix = glslColorMatrices[matrixIdx];

        for (uint32_t i = 0u; i < 3u; i++)
        {
            // TODO: remove when tests are done
            std::cout << (glslMatrix[i] == mul(hlslMatrix, unitVectors[i])) << ',';
            std::cout << (mul(hlslMatrix, unitVectors[i]) == glslMatrix * unitVectors[i]) << ',';

            assert(glslMatrix[i] == mul(hlslMatrix, unitVectors[i]));
            assert(mul(hlslMatrix, unitVectors[i]) == glslMatrix * unitVectors[i]);
        }

        std::cout << std::endl;
    }
}

struct S {
    float3 f;
};

struct T {
    float    a;
    float3   b;
    S        c;
    float2x3 d;
    float2x3 e;
    int      f[3];
    float2   g[2];
    float4   h;
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
    float3 a = float3(1.0f, 2.0f, 3.0f);
    float3 b = float3(2.0f, 3.0f, 4.0f);
    b = a * 3.0f;
    bool3 asdf = bool3(true, false, true);
    pow(a, b);

    {
        float4x3 a;
        float3x4 b;
        float3 v;
        float4 u;
        mul(a, b);
        mul(b, a);
        mul(a, v);
        mul(v, b);
        mul(u, a);
        mul(b, u);

        float4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        a - a;
        b + b;
        static_assert(std::is_same_v<float4x4, decltype(mul(a, b))>);
        static_assert(std::is_same_v<float3x3, decltype(mul(b, a))>);
        static_assert(std::is_same_v<float4, decltype(mul(a, v))>);
        static_assert(std::is_same_v<float4, decltype(mul(v, b))>);
        static_assert(std::is_same_v<float3, decltype(mul(u, a))>);
        static_assert(std::is_same_v<float3, decltype(mul(b, u))>);

    }

    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() = float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() + float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(float4x4() - float4x4())>>);
    static_assert(std::is_same_v<float4x4, std::remove_cvref_t<decltype(mul(float4x4(), float4x4()))>>);

    static_assert(offsetof(T, a) == 0);
    static_assert(offsetof(T, b) == offsetof(T, a) + sizeof(T::a));
    static_assert(offsetof(T, c) == offsetof(T, b) + sizeof(T::b));
    static_assert(offsetof(T, d) == offsetof(T, c) + sizeof(T::c));
    static_assert(offsetof(T, e) == offsetof(T, d) + sizeof(T::d));
    static_assert(offsetof(T, f) == offsetof(T, e) + sizeof(T::e));
    static_assert(offsetof(T, g) == offsetof(T, f) + sizeof(T::f));
    static_assert(offsetof(T, h) == offsetof(T, g) + sizeof(T::g));
    
    float3 x;
    float2x3 y;
    float3x3 z;
    barycentric::reconstructBarycentrics(x, y);
    barycentric::reconstructBarycentrics(x, z);

    // color matrix tests:
    testColorMatrices();
    
    // promote.hlsl tests:

        // promote scalar to vector
    float3 v0 = nbl::hlsl::promote<float3, float>(2.0f);
        // promote scalar to matrix
    float3x3 m0 = nbl::hlsl::promote<float3x3, float>(2.0f);

        // TODO?: promote vector to matrix
    //glm::mat3 m1 = nbl::hlsl::promote<glm::mat3, glm::vec3>(glm::vec3(1.0f, 2.0f, 3.0f));

    // test vector comparison operators
    {
        float3 a = float3(1.0f, 2.0f, 3.0f);
        float3 b = float3(0.5f, 0.5f, 0.5f);
        assert(glm::all(a > b));
        assert(glm::all(b < a));

        b = float3(0.5f, 2.0f, 0.5f);
        assert(glm::all(a >= b));
        assert(glm::all(b <= a));
    }

    // test functions from EOTF.hlsl
    // TODO[Przemek]: tests function output
    float3 TEST_VEC = float3(0.1f, 0.2f, 0.3f);

    colorspace::eotf::identity<float3>(TEST_VEC);
    colorspace::eotf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    colorspace::eotf::sRGB<float3>(TEST_VEC);
    colorspace::eotf::Display_P3<float3>(TEST_VEC);
    colorspace::eotf::DCI_P3_XYZ<float3>(TEST_VEC);
    colorspace::eotf::SMPTE_170M<float3>(TEST_VEC);
    colorspace::eotf::SMPTE_ST2084<float3>(TEST_VEC);
    colorspace::eotf::HDR10_HLG<float3>(TEST_VEC);
    colorspace::eotf::AdobeRGB<float3>(TEST_VEC);
    colorspace::eotf::Gamma_2_2<float3>(TEST_VEC);
    colorspace::eotf::ACEScc<float3>(TEST_VEC);
    colorspace::eotf::ACEScct<float3>(TEST_VEC);

    // test functions from OETF.hlsl
    colorspace::oetf::identity<float3>(TEST_VEC);
    colorspace::oetf::impl_shared_2_4<float3>(TEST_VEC, 0.5f);
    colorspace::oetf::sRGB<float3>(TEST_VEC);
    colorspace::oetf::Display_P3<float3>(TEST_VEC);
    colorspace::oetf::DCI_P3_XYZ<float3>(TEST_VEC);
    colorspace::oetf::SMPTE_170M<float3>(TEST_VEC);
    colorspace::oetf::SMPTE_ST2084<float3>(TEST_VEC);
    colorspace::oetf::HDR10_HLG<float3>(TEST_VEC);
    colorspace::oetf::AdobeRGB<float3>(TEST_VEC);
    colorspace::oetf::Gamma_2_2<float3>(TEST_VEC);
    colorspace::oetf::ACEScc<float3>(TEST_VEC);
    colorspace::oetf::ACEScct<float3>(TEST_VEC);
    
    CompatibilityTest::runTests(argc, argv);
}
