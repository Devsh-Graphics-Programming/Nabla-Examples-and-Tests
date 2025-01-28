#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_TESTERS_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_TESTERS_INCLUDED_

#include <nabla.h>
#include "app_resources/common.hlsl"
#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

using namespace nbl;

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

    enum class TestType
    {
        CPU,
        GPU
    };

    template<typename T>
    void verifyTestValue(const std::string& memberName, const T& expectedVal, const T& testVal, const TestType testType)
    {
        if (std::abs(double(expectedVal) - double(testVal)) <= MaxAllowedError)
            return;

        std::stringstream ss;
        switch (testType)
        {
        case TestType::CPU:
            ss << "CPU TEST ERROR:\n";
        case TestType::GPU:
            ss << "GPU TEST ERROR:\n";
        }

        ss << "nbl::hlsl::" << memberName << " produced incorrect output! test value: " << testVal << " expected value: " << expectedVal << '\n';

        if (memberName == "pow")
        {
            auto a = expectedVal;
            auto b = testVal;
            std::cout << std::bitset<32>(reinterpret_cast<uint32_t&>(a)) << std::endl;
            std::cout << std::bitset<32>(reinterpret_cast<uint32_t&>(b)) << std::endl;
        }

        m_logger->log(ss.str().c_str(), system::ILogger::ELL_ERROR);
    }

    template<typename T>
    void verifyTestVector3dValue(const std::string& memberName, const nbl::hlsl::vector<T, 3>& expectedVal, const nbl::hlsl::vector<T, 3>& testVal, const TestType testType)
    {
        static constexpr float MaxAllowedError = 0.1f;
        if (std::abs(double(expectedVal.x) - double(testVal.x)) <= MaxAllowedError ||
            std::abs(double(expectedVal.y) - double(testVal.y)) <= MaxAllowedError ||
            std::abs(double(expectedVal.z) - double(testVal.z)) <= MaxAllowedError)
            return;

        std::stringstream ss;
        switch (testType)
        {
        case TestType::CPU:
            ss << "CPU TEST ERROR:\n";
        case TestType::GPU:
            ss << "GPU TEST ERROR:\n";
        }

        ss << "nbl::hlsl::" << memberName << " produced incorrect output! test value: " <<
            testVal.x << ' ' << testVal.y << ' ' << testVal.z <<
            " expected value: " << expectedVal.x << ' ' << expectedVal.y << ' ' << expectedVal.z << '\n';

        m_logger->log(ss.str().c_str(), system::ILogger::ELL_ERROR);
    }

    template<typename T>
    void verifyTestMatrix3x3Value(const std::string& memberName, const nbl::hlsl::matrix<T, 3, 3>& expectedVal, const nbl::hlsl::matrix<T, 3, 3>& testVal, const TestType testType)
    {
        for (int i = 0; i < 3; ++i)
        {
            auto expectedValRow = expectedVal[i];
            auto testValRow = testVal[i];
            verifyTestVector3dValue(memberName, expectedValRow, testValRow, testType);
        }
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

private:
    static constexpr float MaxAllowedError = 0.001f;
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
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<int> coinFlipDistribution(0, 1);

        m_logger->log("tgmath.hlsl TESTS:", system::ILogger::ELL_PERFORMANCE);
        for (int i = 0; i < Iterations; ++i)
        {
            // Set input thest values that will be used in both CPU and GPU tests
            TgmathIntputTestValues commonTestInputValues;
            commonTestInputValues.floor = realDistribution(mt);
            commonTestInputValues.lerpX = realDistributionNeg(mt);
            commonTestInputValues.lerpY = realDistributionPos(mt);
            commonTestInputValues.lerpA = realDistributionZeroToOne(mt);
            commonTestInputValues.isnan = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::quiet_NaN();
            commonTestInputValues.isinf = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::infinity();
            commonTestInputValues.powX = realDistributionSmall(mt);
            commonTestInputValues.powY = realDistributionSmall(mt);
            commonTestInputValues.exp = realDistributionSmall(mt);
            commonTestInputValues.exp2 = realDistributionSmall(mt);
            commonTestInputValues.log = realDistribution(mt);
            commonTestInputValues.absF = realDistribution(mt);
            commonTestInputValues.absI = intDistribution(mt);
            commonTestInputValues.sqrt = realDistribution(mt);
            commonTestInputValues.sin = realDistribution(mt);
            commonTestInputValues.cos = realDistribution(mt);
            commonTestInputValues.acos = realDistribution(mt);
            commonTestInputValues.modf = realDistribution(mt);

            commonTestInputValues.floorVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.lerpXVec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
            commonTestInputValues.lerpYVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            commonTestInputValues.lerpAVec = float32_t3(realDistributionZeroToOne(mt), realDistributionZeroToOne(mt), realDistributionZeroToOne(mt));
            commonTestInputValues.isnanVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.isinfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.powXVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            commonTestInputValues.powYVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            commonTestInputValues.expVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            commonTestInputValues.exp2Vec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            commonTestInputValues.logVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.absFVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.absIVec = int32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
            commonTestInputValues.sqrtVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.sinVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.cosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.acosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.modfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));

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
            {
                float tmp;
                expectedTestValues.modf = std::modf(commonTestInputValues.modf, &tmp);
            }

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
            {
                float tmp;
                expectedTestValues.modfVec = float32_t3(std::modf(commonTestInputValues.modfVec.x, &tmp), std::modf(commonTestInputValues.modfVec.y, &tmp), std::modf(commonTestInputValues.modfVec.z, &tmp));
            }

            performCpuTests(commonTestInputValues, expectedTestValues);
            performGpuTests(commonTestInputValues, expectedTestValues);
        }
        m_logger->log("tgmath.hlsl TESTS DONE.", system::ILogger::ELL_PERFORMANCE);
    }

private:
    inline static constexpr int Iterations = 100u;

    void performCpuTests(const TgmathIntputTestValues& commonTestInputValues, const TgmathTestValues& expectedTestValues)
    {
        TgmathTestValues cpuTestValues;
        cpuTestValues.fillTestValues(commonTestInputValues);
        verifyTestValues(expectedTestValues, cpuTestValues, ITester::TestType::CPU);
        
    }

    void performGpuTests(const TgmathIntputTestValues& commonTestInputValues, const TgmathTestValues& expectedTestValues)
    {
        TgmathTestValues gpuTestValues;
        gpuTestValues = dispatch<TgmathIntputTestValues, TgmathTestValues>(commonTestInputValues);
        verifyTestValues(expectedTestValues, gpuTestValues, ITester::TestType::GPU);
    }

    void verifyTestValues(const TgmathTestValues& expectedTestValues, const TgmathTestValues& testValues, ITester::TestType testType)
    {
        verifyTestValue("floor", expectedTestValues.floor, testValues.floor, testType);
        verifyTestValue("lerp", expectedTestValues.lerp, testValues.lerp, testType);
        verifyTestValue("isnan", expectedTestValues.isnan, testValues.isnan, testType);
        verifyTestValue("isinf", expectedTestValues.isinf, testValues.isinf, testType);
        verifyTestValue("pow", expectedTestValues.pow, testValues.pow, testType);
        verifyTestValue("exp", expectedTestValues.exp, testValues.exp, testType);
        verifyTestValue("exp2", expectedTestValues.exp2, testValues.exp2, testType);
        verifyTestValue("log", expectedTestValues.log, testValues.log, testType);
        verifyTestValue("absF", expectedTestValues.absF, testValues.absF, testType);
        verifyTestValue("absI", expectedTestValues.absI, testValues.absI, testType);
        verifyTestValue("sqrt", expectedTestValues.sqrt, testValues.sqrt, testType);
        verifyTestValue("sin", expectedTestValues.sin, testValues.sin, testType);
        verifyTestValue("cos", expectedTestValues.cos, testValues.cos, testType);
        verifyTestValue("acos", expectedTestValues.acos, testValues.acos, testType);
        verifyTestValue("modf", expectedTestValues.modf, testValues.modf, testType);

        verifyTestVector3dValue("floorVec", expectedTestValues.floorVec, testValues.floorVec, testType);
        verifyTestVector3dValue("lerpVec", expectedTestValues.lerpVec, testValues.lerpVec, testType);
        verifyTestVector3dValue("isnanVec", expectedTestValues.isnanVec, testValues.isnanVec, testType);
        verifyTestVector3dValue("isinfVec", expectedTestValues.isinfVec, testValues.isinfVec, testType);
        verifyTestVector3dValue("powVec", expectedTestValues.powVec, testValues.powVec, testType);
        verifyTestVector3dValue("expVec", expectedTestValues.expVec, testValues.expVec, testType);
        verifyTestVector3dValue("exp2Vec", expectedTestValues.exp2Vec, testValues.exp2Vec, testType);
        verifyTestVector3dValue("logVec", expectedTestValues.logVec, testValues.logVec, testType);
        verifyTestVector3dValue("absFVec", expectedTestValues.absFVec, testValues.absFVec, testType);
        verifyTestVector3dValue("absIVec", expectedTestValues.absIVec, testValues.absIVec, testType);
        verifyTestVector3dValue("sqrtVec", expectedTestValues.sqrtVec, testValues.sqrtVec, testType);
        verifyTestVector3dValue("sinVec", expectedTestValues.sinVec, testValues.sinVec, testType);
        verifyTestVector3dValue("cosVec", expectedTestValues.cosVec, testValues.cosVec, testType);
        verifyTestVector3dValue("acosVec", expectedTestValues.acosVec, testValues.acosVec, testType);
        verifyTestVector3dValue("modfVec", expectedTestValues.modfVec, testValues.modfVec, testType);
    }
};

class CIntrinsicsTester final : public ITester
{
public:
    void performTests()
    {
        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_real_distribution<float> realDistributionNeg(-50.0f, -1.0f);
        std::uniform_real_distribution<float> realDistributionPos(1.0f, 50.0f);
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<uint32_t> uintDistribution(0, 100);

        m_logger->log("intrinsics.hlsl TESTS:", system::ILogger::ELL_PERFORMANCE);
        for (int i = 0; i < Iterations; ++i)
        {
            // Set input thest values that will be used in both CPU and GPU tests
            IntrinsicsIntputTestValues commonTestInputValues;
            commonTestInputValues.bitCount = intDistribution(mt);
            commonTestInputValues.crossLhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.crossRhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.clampVal = realDistribution(mt);
            commonTestInputValues.clampMin = realDistributionNeg(mt);
            commonTestInputValues.clampMax = realDistributionPos(mt);
            commonTestInputValues.length = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.normalize = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.dotLhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.dotRhs = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.determinant = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            commonTestInputValues.findMSB = realDistribution(mt);
            commonTestInputValues.findLSB = realDistribution(mt);
            commonTestInputValues.inverse = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            commonTestInputValues.transpose = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            commonTestInputValues.mulLhs = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            commonTestInputValues.mulRhs = float32_t3x3(
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt),
                realDistribution(mt), realDistribution(mt), realDistribution(mt)
            );
            commonTestInputValues.minA = realDistribution(mt);
            commonTestInputValues.minB = realDistribution(mt);
            commonTestInputValues.maxA = realDistribution(mt);
            commonTestInputValues.maxB = realDistribution(mt);
            commonTestInputValues.rsqrt = realDistributionPos(mt);
            commonTestInputValues.bitReverse = realDistribution(mt);
            commonTestInputValues.frac = realDistribution(mt);

            commonTestInputValues.bitCountVec = int32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
            commonTestInputValues.clampValVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.clampMinVec = float32_t3(realDistributionNeg(mt), realDistributionNeg(mt), realDistributionNeg(mt));
            commonTestInputValues.clampMaxVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            commonTestInputValues.findMSBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            commonTestInputValues.findLSBVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            commonTestInputValues.minAVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.minBVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.maxAVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.maxBVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            commonTestInputValues.rsqrtVec = float32_t3(realDistributionPos(mt), realDistributionPos(mt), realDistributionPos(mt));
            commonTestInputValues.bitReverseVec = uint32_t3(uintDistribution(mt), uintDistribution(mt), uintDistribution(mt));
            commonTestInputValues.fracVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));

            // use std library or glm functions to determine expected test values, the output of functions from tgmath.hlsl will be verified against these values
            IntrinsicsTestValues expectedTestValues;
            expectedTestValues.bitCount = glm::bitCount(commonTestInputValues.bitCount);
            expectedTestValues.clamp = glm::clamp(commonTestInputValues.clampVal, commonTestInputValues.clampMin, commonTestInputValues.clampMax);
            expectedTestValues.length = glm::length(commonTestInputValues.length);
            expectedTestValues.dot = glm::dot(commonTestInputValues.dotLhs, commonTestInputValues.dotRhs);
            expectedTestValues.determinant = glm::determinant(reinterpret_cast<typename float32_t3x3::Base const&>(commonTestInputValues.determinant));
            expectedTestValues.findMSB = glm::findMSB(commonTestInputValues.findMSB);
            expectedTestValues.findLSB = glm::findLSB(commonTestInputValues.findLSB);
            expectedTestValues.min = glm::min(commonTestInputValues.minA, commonTestInputValues.minB);
            expectedTestValues.max = glm::max(commonTestInputValues.maxA, commonTestInputValues.maxB);
            expectedTestValues.rsqrt = (1.0f / std::sqrt(commonTestInputValues.rsqrt));

            expectedTestValues.frac = commonTestInputValues.frac - std::floor(commonTestInputValues.frac);
            expectedTestValues.bitReverse = glm::bitfieldReverse(commonTestInputValues.bitReverse);

            expectedTestValues.normalize = glm::normalize(commonTestInputValues.normalize);
            expectedTestValues.cross = glm::cross(commonTestInputValues.crossLhs, commonTestInputValues.crossRhs);
            expectedTestValues.bitCountVec = int32_t3(glm::bitCount(commonTestInputValues.bitCountVec.x), glm::bitCount(commonTestInputValues.bitCountVec.y), glm::bitCount(commonTestInputValues.bitCountVec.z));
            expectedTestValues.clampVec = float32_t3(
                glm::clamp(commonTestInputValues.clampValVec.x, commonTestInputValues.clampMinVec.x, commonTestInputValues.clampMaxVec.x),
                glm::clamp(commonTestInputValues.clampValVec.y, commonTestInputValues.clampMinVec.y, commonTestInputValues.clampMaxVec.y),
                glm::clamp(commonTestInputValues.clampValVec.z, commonTestInputValues.clampMinVec.z, commonTestInputValues.clampMaxVec.z)
            );
            expectedTestValues.findMSBVec = glm::findMSB(commonTestInputValues.findMSBVec);
            expectedTestValues.findLSBVec = glm::findLSB(commonTestInputValues.findLSBVec);
            expectedTestValues.minVec = float32_t3(
                glm::min(commonTestInputValues.minAVec.x, commonTestInputValues.minBVec.x),
                glm::min(commonTestInputValues.minAVec.y, commonTestInputValues.minBVec.y),
                glm::min(commonTestInputValues.minAVec.z, commonTestInputValues.minBVec.z)
            );
            expectedTestValues.maxVec = float32_t3(
                glm::max(commonTestInputValues.maxAVec.x, commonTestInputValues.maxBVec.x),
                glm::max(commonTestInputValues.maxAVec.y, commonTestInputValues.maxBVec.y),
                glm::max(commonTestInputValues.maxAVec.z, commonTestInputValues.maxBVec.z)
            );
            expectedTestValues.rsqrtVec = float32_t3(1.0f / std::sqrt(commonTestInputValues.rsqrtVec.x), 1.0f / std::sqrt(commonTestInputValues.rsqrtVec.y), 1.0f / std::sqrt(commonTestInputValues.rsqrtVec.z));
            expectedTestValues.bitReverseVec = glm::bitfieldReverse(commonTestInputValues.bitReverseVec);
            expectedTestValues.fracVec = float32_t3(
                commonTestInputValues.fracVec.x - std::floor(commonTestInputValues.fracVec.x),
                commonTestInputValues.fracVec.y - std::floor(commonTestInputValues.fracVec.y),
                commonTestInputValues.fracVec.z - std::floor(commonTestInputValues.fracVec.z));

            auto mulGlm = nbl::hlsl::mul(commonTestInputValues.mulLhs, commonTestInputValues.mulRhs);
            expectedTestValues.mul = reinterpret_cast<float32_t3x3&>(mulGlm);
            auto transposeGlm = glm::transpose(reinterpret_cast<typename float32_t3x3::Base const&>(commonTestInputValues.transpose));
            expectedTestValues.transpose = reinterpret_cast<float32_t3x3&>(transposeGlm);
            auto inverseGlm = glm::inverse(reinterpret_cast<typename float32_t3x3::Base const&>(commonTestInputValues.inverse));
            expectedTestValues.inverse = reinterpret_cast<float32_t3x3&>(inverseGlm);

            performCpuTests(commonTestInputValues, expectedTestValues);
            //performGpuTests(commonTestInputValues, expectedTestValues);
        }
        m_logger->log("intrinsics.hlsl TESTS DONE.", system::ILogger::ELL_PERFORMANCE);
    }

private:
    inline static constexpr int Iterations = 100u;

    void performCpuTests(const IntrinsicsIntputTestValues& commonTestInputValues, const IntrinsicsTestValues& expectedTestValues)
    {
        IntrinsicsTestValues cpuTestValues;
        cpuTestValues.fillTestValues(commonTestInputValues);
        verifyTestValues(expectedTestValues, cpuTestValues, ITester::TestType::CPU);

    }

    void performGpuTests(const IntrinsicsIntputTestValues& commonTestInputValues, const IntrinsicsTestValues& expectedTestValues)
    {
        IntrinsicsTestValues gpuTestValues;
        gpuTestValues = dispatch<IntrinsicsIntputTestValues, IntrinsicsTestValues>(commonTestInputValues);
        verifyTestValues(expectedTestValues, gpuTestValues, ITester::TestType::GPU);
    }

    void verifyTestValues(const IntrinsicsTestValues& expectedTestValues, const IntrinsicsTestValues& testValues, ITester::TestType testType)
    {
        verifyTestValue("bitCount", expectedTestValues.bitCount, testValues.bitCount, testType);
        verifyTestValue("clamp", expectedTestValues.clamp, testValues.clamp, testType);
        verifyTestValue("length", expectedTestValues.length, testValues.length, testType);
        verifyTestValue("dot", expectedTestValues.dot, testValues.dot, testType);
        verifyTestValue("determinant", expectedTestValues.determinant, testValues.determinant, testType);
        verifyTestValue("findMSB", expectedTestValues.findMSB, testValues.findMSB, testType);
        verifyTestValue("findLSB", expectedTestValues.findLSB, testValues.findLSB, testType);
        //verifyTestValue("min", expectedTestValues.min, testValues.min, testType);
        //verifyTestValue("max", expectedTestValues.max, testValues.max, testType);
        verifyTestValue("rsqrt", expectedTestValues.rsqrt, testValues.rsqrt, testType);
        verifyTestValue("frac", expectedTestValues.frac, testValues.frac, testType);
        verifyTestValue("bitReverse", expectedTestValues.bitReverse, testValues.bitReverse, testType);

        verifyTestVector3dValue("normalize", expectedTestValues.normalize, testValues.normalize, testType);
        verifyTestVector3dValue("cross", expectedTestValues.cross, testValues.cross, testType);
        verifyTestVector3dValue("bitCountVec", expectedTestValues.bitCountVec, testValues.bitCountVec, testType);
        verifyTestVector3dValue("clampVec", expectedTestValues.clampVec, testValues.clampVec, testType);
        verifyTestVector3dValue("findMSBVec", expectedTestValues.findMSBVec, testValues.findMSBVec, testType);
        verifyTestVector3dValue("findLSBVec", expectedTestValues.findLSBVec, testValues.findLSBVec, testType);
        //verifyTestVector3dValue("minVec", expectedTestValues.minVec, testValues.minVec, testType);
        //verifyTestVector3dValue("maxVec", expectedTestValues.maxVec, testValues.maxVec, testType);
        verifyTestVector3dValue("rsqrtVec", expectedTestValues.rsqrtVec, testValues.rsqrtVec, testType);
        verifyTestVector3dValue("bitReverseVec", expectedTestValues.bitReverseVec, testValues.bitReverseVec, testType);
        verifyTestVector3dValue("fracVec", expectedTestValues.fracVec, testValues.fracVec, testType);

        verifyTestMatrix3x3Value("mul", expectedTestValues.mul, testValues.mul, testType);
        verifyTestMatrix3x3Value("transpose", expectedTestValues.transpose, testValues.transpose, testType);
        verifyTestMatrix3x3Value("inverse", expectedTestValues.inverse, testValues.inverse, testType);
    }
};

#undef VERIFY_TEST_VALUE
#undef VERIFY_TEST_VECTOR_VALUE

#endif