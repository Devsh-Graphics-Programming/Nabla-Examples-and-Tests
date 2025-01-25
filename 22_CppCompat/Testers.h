#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_TESTERS_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_TESTERS_INCLUDED_

#include <nabla.h>
#include "app_resources/common.hlsl"
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;

static constexpr float MaxAllowedError = 0.001f;
#define VERIFY_TEST_VALUE(MEMBER_NAME)\
if (std::abs(expectedTestValues.MEMBER_NAME - testValues.MEMBER_NAME) > MaxAllowedError)\
{\
    std::cout << "nbl::hlsl::" #MEMBER_NAME << " produced incorrect output! test value: " << testValues.MEMBER_NAME << " expected value: " << expectedTestValues.MEMBER_NAME << std::endl;\
    assert(false);\
}

#define VERIFY_TEST_VECTOR_VALUE(MEMBER_NAME)\
if (memcmp(&expectedTestValues.MEMBER_NAME, &testValues.MEMBER_NAME, sizeof(decltype(testValues.MEMBER_NAME))) != 0)\
{\
    std::cout << "nbl::hlsl::" #MEMBER_NAME << " produced incorrect output! test value: " <<\
    testValues.MEMBER_NAME.x << ' ' << testValues.MEMBER_NAME.y << ' ' << testValues.MEMBER_NAME.z <<\
    " expected value: " << expectedTestValues.MEMBER_NAME.x << ' ' << expectedTestValues.MEMBER_NAME.y << ' ' << expectedTestValues.MEMBER_NAME.z << std::endl;\
    assert(false);\
}

class ITester 
{
public:
    virtual ~ITester()
    {
        m_outputBufferAllocation.memory->unmap();
    };

    struct PipelineSetupData
    {
        core::smart_refctd_ptr<video::ILogicalDevice> device;
        core::smart_refctd_ptr<video::CVulkanConnection> api;
        core::smart_refctd_ptr<asset::IAssetManager> assetMgr;
        core::smart_refctd_ptr<system::ILogger> logger;
        video::IPhysicalDevice* physicalDevice;
        uint32_t computeFamilyIndex;
    };

    template<typename InputStruct, typename OutputStruct>
    void setupPipeline(const PipelineSetupData& pipleineSetupData)
    {
        // setting up pipeline in the constructor
        m_device = core::smart_refctd_ptr(pipleineSetupData.device);
        m_physicalDevice = pipleineSetupData.physicalDevice;
        m_api = core::smart_refctd_ptr(pipleineSetupData.api);
        m_assetMgr = core::smart_refctd_ptr(pipleineSetupData.assetMgr);
        m_logger = core::smart_refctd_ptr(pipleineSetupData.logger);
        m_queueFamily = pipleineSetupData.computeFamilyIndex;
        m_semaphoreCounter = 0;
        m_semaphore = m_device->createSemaphore(0);
        m_cmdpool = m_device->createCommandPool(m_queueFamily, video::IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
        if (!m_cmdpool->createCommandBuffers(video::IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
            ;//TODO: app.logFail("Failed to create Command Buffers!\n");

        // Load shaders, set up pipeline
        core::smart_refctd_ptr<video::IGPUShader> shader;
        {
            asset::IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            // this time we load a shader directly from a file
            auto assetBundle = m_assetMgr->getAsset("app_resources/tgmathTest.comp.hlsl", lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
            {
                ;//TODO: app.logFail("Could not load shader!");
                assert(0);
            }

            // It would be super weird if loading a shader from a file produced more than 1 asset
            assert(assets.size() == 1);
            core::smart_refctd_ptr<asset::ICPUShader> source = asset::IAsset::castDown<asset::ICPUShader>(assets[0]);

            auto* compilerSet = m_assetMgr->getCompilerSet();

            asset::IShaderCompiler::SCompilerOptions options = {};
            options.stage = source->getStage();
            options.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
            options.spirvOptimizer = nullptr;
            options.debugInfoFlags |= asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
            options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
            options.preprocessorOptions.logger = m_logger.get();
            options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(source->getContentType())->getDefaultIncludeFinder();

            auto spirv = compilerSet->compileToSPIRV(source.get(), options);

            video::ILogicalDevice::SShaderCreationParameters params{};
            params.cpushader = spirv.get();
            shader = m_device->createShader(params);
        }

        if (!shader)
            ;//TODO: app.logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

        video::IGPUDescriptorSetLayout::SBinding bindings[2] = {
            {
                .binding = 0,
                .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                .stageFlags = ShaderStage::ESS_COMPUTE,
                .count = 1
            },
            {
                .binding = 1,
                .type = asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                .createFlags = video::IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                .stageFlags = ShaderStage::ESS_COMPUTE,
                .count = 1
            }
        };

        core::smart_refctd_ptr<video::IGPUDescriptorSetLayout> dsLayout = m_device->createDescriptorSetLayout(bindings);
        if (!dsLayout)
            ;//TODO: app.logFail("Failed to create a Descriptor Layout!\n");

        m_pplnLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(dsLayout));
        if (!m_pplnLayout)
            ;//TODO: app.logFail("Failed to create a Pipeline Layout!\n");

        {
            video::IGPUComputePipeline::SCreationParams params = {};
            params.layout = m_pplnLayout.get();
            params.shader.entryPoint = "main";
            params.shader.shader = shader.get();
            if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                ;//TODO: app.logFail("Failed to create pipelines (compile & link shaders)!\n");
        }

        // Allocate memory of the input buffer
        {
            constexpr size_t BufferSize = sizeof(TgmathIntputTestValues);

            video::IGPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            params.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            core::smart_refctd_ptr<video::IGPUBuffer> inputBuff = m_device->createBuffer(std::move(params));
            if (!inputBuff)
                ;//TODO: app.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

            inputBuff->setObjectDebugName("emulated_float64_t output buffer");

            video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = inputBuff->getMemoryReqs();
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            m_inputBufferAllocation = m_device->allocate(reqs, inputBuff.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!m_inputBufferAllocation.isValid())
                ;//TODO: app.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

            assert(inputBuff->getBoundMemory().memory == m_inputBufferAllocation.memory.get());
            core::smart_refctd_ptr<video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

            m_ds = pool->createDescriptorSet(core::smart_refctd_ptr(dsLayout));
            {
                video::IGPUDescriptorSet::SDescriptorInfo info[1];
                info[0].desc = core::smart_refctd_ptr(inputBuff);
                info[0].info.buffer = { .offset = 0,.size = BufferSize };
                video::IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                    {.dstSet = m_ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
                };
                m_device->updateDescriptorSets(writes, {});
            }
        }

        // Allocate memory of the output buffer
        {
            constexpr size_t BufferSize = sizeof(OutputStruct);

            video::IGPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            params.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            core::smart_refctd_ptr<video::IGPUBuffer> outputBuff = m_device->createBuffer(std::move(params));
            if (!outputBuff)
                ;//TODO: app.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

            outputBuff->setObjectDebugName("emulated_float64_t output buffer");

            video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            m_outputBufferAllocation = m_device->allocate(reqs, outputBuff.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!m_outputBufferAllocation.isValid())
                ;//TODO: app.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

            assert(outputBuff->getBoundMemory().memory == m_outputBufferAllocation.memory.get());
            core::smart_refctd_ptr<video::IDescriptorPool> pool = m_device->createDescriptorPoolForDSLayouts(video::IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

            {
                video::IGPUDescriptorSet::SDescriptorInfo info[1];
                info[0].desc = core::smart_refctd_ptr(outputBuff);
                info[0].info.buffer = { .offset = 0,.size = BufferSize };
                video::IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                    {.dstSet = m_ds.get(),.binding = 1,.arrayElement = 0,.count = 1,.info = info}
                };
                m_device->updateDescriptorSets(writes, {});
            }
        }

        if (!m_outputBufferAllocation.memory->map({ 0ull,m_outputBufferAllocation.memory->getAllocationSize() }, video::IDeviceMemoryAllocation::EMCAF_READ))
            ;//TODO: app.logFail("Failed to map the Device Memory!\n");

        // if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
        const video::ILogicalDevice::MappedMemoryRange memoryRange(m_outputBufferAllocation.memory.get(), 0ull, m_outputBufferAllocation.memory->getAllocationSize());
        if (!m_outputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(video::IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        assert(memoryRange.valid() && memoryRange.length >= sizeof(OutputStruct));

        m_queue = m_device->getQueue(m_queueFamily, 0);
    }

protected:
    uint32_t m_queueFamily;
    core::smart_refctd_ptr<video::ILogicalDevice> m_device;
    core::smart_refctd_ptr<video::CVulkanConnection> m_api;
    video::IPhysicalDevice* m_physicalDevice;
    core::smart_refctd_ptr<asset::IAssetManager> m_assetMgr;
    core::smart_refctd_ptr<system::ILogger> m_logger;
    video::IDeviceMemoryAllocator::SAllocation m_inputBufferAllocation = {};
    video::IDeviceMemoryAllocator::SAllocation m_outputBufferAllocation = {};
    core::smart_refctd_ptr<video::IGPUCommandBuffer> m_cmdbuf = nullptr;
    core::smart_refctd_ptr<video::IGPUCommandPool> m_cmdpool = nullptr;
    core::smart_refctd_ptr<video::IGPUDescriptorSet> m_ds = nullptr;
    core::smart_refctd_ptr<video::IGPUPipelineLayout> m_pplnLayout = nullptr;
    core::smart_refctd_ptr<video::IGPUComputePipeline> m_pipeline;
    core::smart_refctd_ptr<video::ISemaphore> m_semaphore;
    video::IQueue* m_queue;
    uint64_t m_semaphoreCounter;
    
    template<typename InputStruct, typename OutputStruct>
    OutputStruct dispatch(const InputStruct& input)
    {
        // Update input buffer
        if (!m_inputBufferAllocation.memory->map({ 0ull,m_inputBufferAllocation.memory->getAllocationSize() }, video::IDeviceMemoryAllocation::EMCAF_READ))
            __debugbreak();//TODO: app.logFail("Failed to map the Device Memory!\n");

        const video::ILogicalDevice::MappedMemoryRange memoryRange(m_inputBufferAllocation.memory.get(), 0ull, m_inputBufferAllocation.memory->getAllocationSize());
        if (!m_inputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(video::IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        std::memcpy(static_cast<InputStruct*>(m_inputBufferAllocation.memory->getMappedPointer()), &input, sizeof(InputStruct));

        m_inputBufferAllocation.memory->unmap();

        // record command buffer
        m_cmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
        m_cmdbuf->begin(video::IGPUCommandBuffer::USAGE::NONE);
        m_cmdbuf->beginDebugMarker("test", core::vector4df_SIMD(0, 1, 0, 1));
        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
        m_cmdbuf->dispatch(16, 1, 1);
        m_cmdbuf->endDebugMarker();
        m_cmdbuf->end();

        video::IQueue::SSubmitInfo submitInfos[1] = {};
        const video::IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
        submitInfos[0].commandBuffers = cmdbufs;
        const video::IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = m_semaphore.get(), .value = ++m_semaphoreCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
        submitInfos[0].signalSemaphores = signals;

        m_api->startCapture();
        m_queue->submit(submitInfos);
        m_api->endCapture();

        m_device->waitIdle();
        OutputStruct output;
        std::memcpy(&output, static_cast<OutputStruct*>(m_outputBufferAllocation.memory->getMappedPointer()), sizeof(OutputStruct));
        m_device->waitIdle();

        return output;
    }
};

class CTgmathTester final : public ITester
{
public:
    void performTests()
    {
        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_real_distribution<float> realDistributionNeg(-50.0f, -1.0f);
        std::uniform_real_distribution<float> realDistributionPos(1.0f, 50.0f);
        std::uniform_real_distribution<float> realDistributionZeroToOne(0.0f, 1.0f);
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<int> coinFlipDistribution(0, 1);

        // Set input thest values that will be used in both CPU and GPU tests
        TgmathIntputTestValues commonTestInputValues;
        commonTestInputValues.floor = realDistribution(mt);
        commonTestInputValues.lerpX = realDistributionNeg(mt);
        commonTestInputValues.lerpY = realDistributionPos(mt);
        commonTestInputValues.lerpA = realDistributionZeroToOne(mt);
        commonTestInputValues.isnan = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::quiet_NaN();
        commonTestInputValues.isinf = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::infinity();
        commonTestInputValues.powX = realDistribution(mt);
        commonTestInputValues.powY = realDistribution(mt);
        commonTestInputValues.exp = realDistribution(mt);
        commonTestInputValues.exp2 = realDistribution(mt);
        commonTestInputValues.log = realDistribution(mt);
        commonTestInputValues.absF = realDistribution(mt);
        commonTestInputValues.absI = intDistribution(mt);
        commonTestInputValues.sqrt = realDistribution(mt);
        commonTestInputValues.sin = realDistribution(mt);
        commonTestInputValues.cos = realDistribution(mt);
        commonTestInputValues.acos = realDistribution(mt);

        commonTestInputValues.floorVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.lerpXVec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
        commonTestInputValues.lerpYVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
        commonTestInputValues.lerpAVec = float32_t3(realDistributionZeroToOne(mt), realDistributionZeroToOne(mt), realDistributionZeroToOne(mt));
        commonTestInputValues.isnanVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.isinfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.powXVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.powYVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.expVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.exp2Vec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.logVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.absFVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.absIVec = int32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
        commonTestInputValues.sqrtVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.sinVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.cosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
        commonTestInputValues.acosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));

        // use std library functions to determine expected test values, the output of functions from tgmath.hlsl will be verified against these values
        TgmathTestValues expectedTestValues;
        expectedTestValues.floor = std::floor(commonTestInputValues.floor);
        expectedTestValues.lerp = std::lerp(commonTestInputValues.lerpX, commonTestInputValues.lerpY, commonTestInputValues.lerpA);
        expectedTestValues.isnan = std::isnan(commonTestInputValues.isnan);
        expectedTestValues.isinf = std::isinf(commonTestInputValues.isinf);
        expectedTestValues.pow = std::pow(commonTestInputValues.powX, commonTestInputValues.powY);
        expectedTestValues.exp = std::exp(commonTestInputValues.exp);
        expectedTestValues.exp2 = std::exp2(commonTestInputValues.exp2);
        expectedTestValues.log = std::log(commonTestInputValues.log);
        expectedTestValues.absF = std::abs(commonTestInputValues.absF);
        expectedTestValues.absI = std::abs(commonTestInputValues.absI);
        expectedTestValues.sqrt = std::sqrt(commonTestInputValues.sqrt);
        expectedTestValues.sin = std::sin(commonTestInputValues.sin);
        expectedTestValues.cos = std::cos(commonTestInputValues.cos);
        expectedTestValues.acos = std::acos(commonTestInputValues.acos);

        expectedTestValues.floorVec = float32_t3(std::floor(commonTestInputValues.floorVec.x), std::floor(commonTestInputValues.floorVec.y), std::floor(commonTestInputValues.floorVec.z));

        expectedTestValues.lerpVec.x = std::lerp(commonTestInputValues.lerpXVec.x, commonTestInputValues.lerpYVec.x, commonTestInputValues.lerpAVec.x);
        expectedTestValues.lerpVec.y = std::lerp(commonTestInputValues.lerpXVec.y, commonTestInputValues.lerpYVec.y, commonTestInputValues.lerpAVec.y);
        expectedTestValues.lerpVec.z = std::lerp(commonTestInputValues.lerpXVec.z, commonTestInputValues.lerpYVec.z, commonTestInputValues.lerpAVec.z);

        expectedTestValues.isnanVec = float32_t3(std::isnan(commonTestInputValues.isnanVec.x), std::isnan(commonTestInputValues.isnanVec.y), std::isnan(commonTestInputValues.isnanVec.z));
        expectedTestValues.isinfVec = float32_t3(std::isinf(commonTestInputValues.isinfVec.x), std::isinf(commonTestInputValues.isinfVec.y), std::isinf(commonTestInputValues.isinfVec.z));

        expectedTestValues.powVec.x = std::pow(commonTestInputValues.powXVec.x, commonTestInputValues.powYVec.x);
        expectedTestValues.powVec.y = std::pow(commonTestInputValues.powXVec.y, commonTestInputValues.powYVec.y);
        expectedTestValues.powVec.z = std::pow(commonTestInputValues.powXVec.z, commonTestInputValues.powYVec.z);

        expectedTestValues.expVec = float32_t3(std::exp(commonTestInputValues.expVec.x), std::exp(commonTestInputValues.expVec.y), std::exp(commonTestInputValues.expVec.z));
        expectedTestValues.exp2Vec = float32_t3(std::exp2(commonTestInputValues.exp2Vec.x), std::exp2(commonTestInputValues.exp2Vec.y), std::exp2(commonTestInputValues.exp2Vec.z));
        expectedTestValues.logVec = float32_t3(std::log(commonTestInputValues.logVec.x), std::log(commonTestInputValues.logVec.y), std::log(commonTestInputValues.logVec.z));
        expectedTestValues.absFVec = float32_t3(std::abs(commonTestInputValues.absFVec.x), std::abs(commonTestInputValues.absFVec.y), std::abs(commonTestInputValues.absFVec.z));
        expectedTestValues.absIVec = float32_t3(std::abs(commonTestInputValues.absIVec.x), std::abs(commonTestInputValues.absIVec.y), std::abs(commonTestInputValues.absIVec.z));
        expectedTestValues.sqrtVec = float32_t3(std::sqrt(commonTestInputValues.sqrtVec.x), std::sqrt(commonTestInputValues.sqrtVec.y), std::sqrt(commonTestInputValues.sqrtVec.z));
        expectedTestValues.cosVec = float32_t3(std::cos(commonTestInputValues.cosVec.x), std::cos(commonTestInputValues.cosVec.y), std::cos(commonTestInputValues.cosVec.z));
        expectedTestValues.sinVec = float32_t3(std::sin(commonTestInputValues.sinVec.x), std::sin(commonTestInputValues.sinVec.y), std::sin(commonTestInputValues.sinVec.z));
        expectedTestValues.acosVec = float32_t3(std::acos(commonTestInputValues.acosVec.x), std::acos(commonTestInputValues.acosVec.y), std::acos(commonTestInputValues.acosVec.z));

        // perform C++ side test of functions from tgmath.hlsl
        performCpuTests(commonTestInputValues, expectedTestValues);
        performGpuTests(commonTestInputValues, expectedTestValues);
    }

private:
    inline static constexpr int Iterations = 1u;

    void performCpuTests(const TgmathIntputTestValues& commonTestInputValues, const TgmathTestValues& expectedTestValues)
    {
        TgmathTestValues cpuTestValues;
        for (int i = 0; i < Iterations; ++i)
        {
            cpuTestValues.fillTestValues(commonTestInputValues);
            m_logger->log("CPU TESTS:", system::ILogger::ELL_PERFORMANCE);
            verifyTestValues(expectedTestValues, cpuTestValues);
        }
    }

    void performGpuTests(const TgmathIntputTestValues& commonTestInputValues, const TgmathTestValues& expectedTestValues)
    {
        TgmathTestValues gpuTestValues;
        for (int i = 0; i < Iterations; ++i)
        {
            gpuTestValues = dispatch<TgmathIntputTestValues, TgmathTestValues>(commonTestInputValues);
            m_logger->log("GPU TESTS:", system::ILogger::ELL_PERFORMANCE);
            verifyTestValues(expectedTestValues, gpuTestValues);
            __debugbreak();
        }
    }

    void verifyTestValues(const TgmathTestValues& expectedTestValues, const TgmathTestValues& testValues)
    {
        VERIFY_TEST_VALUE(floor);
        VERIFY_TEST_VALUE(lerp);
        VERIFY_TEST_VALUE(isnan);
        VERIFY_TEST_VALUE(isinf);
        VERIFY_TEST_VALUE(pow);
        VERIFY_TEST_VALUE(exp);
        VERIFY_TEST_VALUE(exp2);
        VERIFY_TEST_VALUE(log);
        VERIFY_TEST_VALUE(absF);
        VERIFY_TEST_VALUE(absI);
        VERIFY_TEST_VALUE(sqrt);
        VERIFY_TEST_VALUE(sin);
        VERIFY_TEST_VALUE(cos);
        VERIFY_TEST_VALUE(acos);

        VERIFY_TEST_VECTOR_VALUE(floorVec);
        VERIFY_TEST_VECTOR_VALUE(lerpVec);
        VERIFY_TEST_VECTOR_VALUE(isnanVec);
        VERIFY_TEST_VECTOR_VALUE(isinfVec);
        VERIFY_TEST_VECTOR_VALUE(powVec);
        VERIFY_TEST_VECTOR_VALUE(expVec);
        VERIFY_TEST_VECTOR_VALUE(exp2Vec);
        VERIFY_TEST_VECTOR_VALUE(logVec);
        VERIFY_TEST_VECTOR_VALUE(absFVec);
        VERIFY_TEST_VECTOR_VALUE(absIVec);
        VERIFY_TEST_VECTOR_VALUE(sqrtVec);
        VERIFY_TEST_VECTOR_VALUE(cosVec);
        VERIFY_TEST_VECTOR_VALUE(sinVec);
        VERIFY_TEST_VECTOR_VALUE(acosVec);
    }
};

#undef VERIFY_TEST_VALUE
#undef VERIFY_TEST_VECTOR_VALUE

#endif