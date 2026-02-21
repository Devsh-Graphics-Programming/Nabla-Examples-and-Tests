#ifndef _NBL_COMMON_I_TESTER_INCLUDED_
#define _NBL_COMMON_I_TESTER_INCLUDED_

#include <nabla.h>
#include <nbl/system/to_string.h>
#include <ranges>
#include <nbl/builtin/hlsl/testing/relative_approx_compare.hlsl>

using namespace nbl;

#include <nbl/builtin/hlsl/ieee754.hlsl>

template<typename InputTestValues, typename TestResults, typename TestExecutor>
class ITester
{
public:
    struct PipelineSetupData
    {
        std::string shaderKey;
        core::smart_refctd_ptr<video::ILogicalDevice> device;
        core::smart_refctd_ptr<video::CVulkanConnection> api;
        core::smart_refctd_ptr<asset::IAssetManager> assetMgr;
        core::smart_refctd_ptr<system::ILogger> logger;
        video::IPhysicalDevice* physicalDevice;
        uint32_t computeFamilyIndex;
    };

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
            logFail("Failed to create Command Buffers!\n");

        // Load shaders, set up pipeline
        core::smart_refctd_ptr<asset::IShader> shader;
        {
            asset::IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = "app_resources"; // virtual root
            auto assetBundle = m_assetMgr->getAsset(pipleineSetupData.shaderKey.data(), lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty())
                return logFail("Could not load shader!");

            // It would be super weird if loading a shader from a file produced more than 1 asset
            assert(assets.size() == 1);
            core::smart_refctd_ptr<asset::IShader> source = asset::IAsset::castDown<asset::IShader>(assets[0]);

            shader = m_device->compileShader({ source.get() });
        }

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
            logFail("Failed to create a Descriptor Layout!\n");

        m_pplnLayout = m_device->createPipelineLayout({}, core::smart_refctd_ptr(dsLayout));
        if (!m_pplnLayout)
            logFail("Failed to create a Pipeline Layout!\n");

        {
            video::IGPUComputePipeline::SCreationParams params = {};
            params.layout = m_pplnLayout.get();
            params.shader.entryPoint = "main";
            params.shader.shader = shader.get();
            if (!m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                logFail("Failed to create pipelines (compile & link shaders)!\n");
        }

        // Allocate memory of the input buffer
        {
            const size_t BufferSize = sizeof(InputTestValues) * m_testIterationCount;

            video::IGPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            params.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            core::smart_refctd_ptr<video::IGPUBuffer> inputBuff = m_device->createBuffer(std::move(params));
            if (!inputBuff)
                logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

            inputBuff->setObjectDebugName("emulated_float64_t output buffer");

            video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = inputBuff->getMemoryReqs();
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            m_inputBufferAllocation = m_device->allocate(reqs, inputBuff.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!m_inputBufferAllocation.isValid())
                logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

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
            const size_t BufferSize = sizeof(TestResults) * m_testIterationCount;

            video::IGPUBuffer::SCreationParams params = {};
            params.size = BufferSize;
            params.usage = video::IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
            core::smart_refctd_ptr<video::IGPUBuffer> outputBuff = m_device->createBuffer(std::move(params));
            if (!outputBuff)
                logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

            outputBuff->setObjectDebugName("emulated_float64_t output buffer");

            video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
            reqs.memoryTypeBits &= m_physicalDevice->getHostVisibleMemoryTypeBits();

            m_outputBufferAllocation = m_device->allocate(reqs, outputBuff.get(), video::IDeviceMemoryAllocation::EMAF_NONE);
            if (!m_outputBufferAllocation.isValid())
                logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

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
            logFail("Failed to map the Device Memory!\n");

        // if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
        const video::ILogicalDevice::MappedMemoryRange memoryRange(m_outputBufferAllocation.memory.get(), 0ull, m_outputBufferAllocation.memory->getAllocationSize());
        if (!m_outputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(video::IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        assert(memoryRange.valid() && memoryRange.length >= sizeof(TestResults));

        m_queue = m_device->getQueue(m_queueFamily, 0);
    }

    bool performTestsAndVerifyResults(const std::string& logFileName)
    {
        m_logFile.open(logFileName, std::ios::out | std::ios::trunc);
        if (!m_logFile.is_open())
            m_logger->log("Failed to open log file!", system::ILogger::ELL_ERROR);

        core::vector<InputTestValues> inputTestValues;
        core::vector<TestResults> exceptedTestResults;

        inputTestValues.reserve(m_testIterationCount);
        exceptedTestResults.reserve(m_testIterationCount);

        m_logger->log("TESTS:", system::ILogger::ELL_PERFORMANCE);
        for (int i = 0; i < m_testIterationCount; ++i)
        {
            // Set input thest values that will be used in both CPU and GPU tests
            InputTestValues testInput = generateInputTestValues();
            // use std library or glm functions to determine expected test values, the output of functions from intrinsics.hlsl will be verified against these values
            TestResults expected = determineExpectedResults(testInput);

            inputTestValues.push_back(testInput);
            exceptedTestResults.push_back(expected);
        }

        core::vector<TestResults> cpuTestResults = performCpuTests(inputTestValues);
        core::vector<TestResults> gpuTestResults = performGpuTests(inputTestValues);

        bool pass = verifyAllTestResults(cpuTestResults, gpuTestResults, exceptedTestResults);

        m_logger->log("TESTS DONE.", system::ILogger::ELL_PERFORMANCE);
        reloadSeed();

        m_logFile.close();
        return pass;
    }

    virtual ~ITester()
    {
        m_outputBufferAllocation.memory->unmap();
    };

protected:
    enum class TestType
    {
        CPU,
        GPU
    };

    /**
    * @param testBatchCount one test batch is equal to m_WorkgroupSize, so number of tests performed will be m_WorkgroupSize * testbatchCount
    */
    ITester(const uint32_t testBatchCount)
        : m_testBatchCount(testBatchCount), m_testIterationCount(testBatchCount * m_WorkgroupSize)
    {
        reloadSeed();
    };

    virtual bool verifyTestResults(const TestResults& expectedTestValues, const TestResults& testValues, const size_t testIteration, const uint32_t seed, TestType testType) = 0;

    virtual InputTestValues generateInputTestValues() = 0;

    virtual TestResults determineExpectedResults(const InputTestValues& testInput) = 0;

    std::mt19937& getRandomEngine()
    {
        return m_mersenneTwister;
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

    void dispatchGpuTests(const core::vector<InputTestValues>& input, core::vector<TestResults>& output)
    {
        // Update input buffer
        if (!m_inputBufferAllocation.memory->map({ 0ull,m_inputBufferAllocation.memory->getAllocationSize() }, video::IDeviceMemoryAllocation::EMCAF_READ))
            logFail("Failed to map the Device Memory!\n");

        const video::ILogicalDevice::MappedMemoryRange memoryRange(m_inputBufferAllocation.memory.get(), 0ull, m_inputBufferAllocation.memory->getAllocationSize());
        if (!m_inputBufferAllocation.memory->getMemoryPropertyFlags().hasFlags(video::IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
            m_device->invalidateMappedMemoryRanges(1, &memoryRange);

        assert(m_testIterationCount == input.size());
        const size_t inputDataSize = sizeof(InputTestValues) * m_testIterationCount;
        std::memcpy(static_cast<InputTestValues*>(m_inputBufferAllocation.memory->getMappedPointer()), input.data(), inputDataSize);

        m_inputBufferAllocation.memory->unmap();

        // record command buffer
        const uint32_t dispatchSizeX = m_testBatchCount;
        m_cmdbuf->reset(video::IGPUCommandBuffer::RESET_FLAGS::NONE);
        m_cmdbuf->begin(video::IGPUCommandBuffer::USAGE::NONE);
        m_cmdbuf->beginDebugMarker("test", core::vector4df_SIMD(0, 1, 0, 1));
        m_cmdbuf->bindComputePipeline(m_pipeline.get());
        m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
        m_cmdbuf->dispatch(dispatchSizeX, 1, 1);
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

        // save test results
        assert(m_testIterationCount == output.size());
        const size_t outputDataSize = sizeof(TestResults) * m_testIterationCount;
        std::memcpy(output.data(), static_cast<TestResults*>(m_outputBufferAllocation.memory->getMappedPointer()), outputDataSize);

        m_device->waitIdle();
    }

    template<typename T>
    bool verifyTestValue(const std::string& memberName, const T& expectedVal, const T& testVal,
        const size_t testIteration, const uint32_t seed, const TestType testType, const float64_t maxAllowedDifference = 0.0)
    {
        if (compareTestValues<T>(expectedVal, testVal, maxAllowedDifference))
            return true;

        printTestFail<T>(memberName, expectedVal, testVal, testIteration, seed, testType);
        return false;
    }

    template<typename T>
    void printTestFail(const std::string& memberName, const T& expectedVal, const T& testVal,
        const size_t testIteration, const uint32_t seed, const TestType testType)
    {
        std::stringstream ss;
        switch (testType)
        {
        case TestType::CPU:
            ss << "CPU TEST ERROR:\n";
            break;
        case TestType::GPU:
            ss << "GPU TEST ERROR:\n";
        }

        ss << "nbl::hlsl::" << memberName << " produced incorrect output!" << '\n';
        ss << "TEST ITERATION INDEX: " << testIteration << " SEED: " << seed << '\n';
        ss << "EXPECTED VALUE: " << system::to_string(expectedVal) << " TEST VALUE: " << system::to_string(testVal) << '\n';

        m_logger->log("%s", system::ILogger::ELL_ERROR, ss.str().c_str());
        m_logFile << ss.str() << '\n';
    }

private:
    template<typename... Args>
    inline void logFail(const char* msg, Args&&... args)
    {
        m_logger->log(msg, system::ILogger::ELL_ERROR, std::forward<Args>(args)...);
        exit(-1);
    }

    core::vector<TestResults> performCpuTests(const core::vector<InputTestValues>& inputTestValues)
    {
        core::vector<TestResults> output(m_testIterationCount);
        TestExecutor testExecutor;

        auto iterations = std::views::iota(0ull, m_testIterationCount);
        std::for_each(std::execution::par_unseq, iterations.begin(), iterations.end(),
            [&](size_t i)
            {
                testExecutor(inputTestValues[i], output[i]);
            }
        );

        return output;
    }

    core::vector<TestResults> performGpuTests(const core::vector<InputTestValues>& inputTestValues)
    {
        core::vector<TestResults> output(m_testIterationCount);
        dispatchGpuTests(inputTestValues, output);

        return output;
    }

    bool verifyAllTestResults(const core::vector<TestResults>& cpuTestReults, const core::vector<TestResults>& gpuTestReults, const core::vector<TestResults>& exceptedTestReults)
    {
        bool pass = true;
        for (int i = 0; i < m_testIterationCount; ++i)
        {
            pass = verifyTestResults(exceptedTestReults[i], cpuTestReults[i], i, m_seed, ITester::TestType::CPU) && pass;
            pass = verifyTestResults(exceptedTestReults[i], gpuTestReults[i], i, m_seed, ITester::TestType::GPU) && pass;
        }
        return pass;
    }

    void reloadSeed()
    {
        std::random_device rd;
        m_seed = rd();
        m_mersenneTwister = std::mt19937(m_seed);
    }

    template<typename T>
    bool compareTestValues(const T& lhs, const T& rhs, const float64_t maxAllowedDifference)
    {
        return lhs == rhs;
    }
    template<typename T> requires concepts::FloatingPointLikeScalar<T> || concepts::FloatingPointLikeVectorial<T> || (concepts::Matricial<T> && concepts::FloatingPointLikeScalar<typename nbl::hlsl::matrix_traits<T>::scalar_type>)
    bool compareTestValues(const T& lhs, const T& rhs, const float64_t maxAllowedDifference)
    {
        return nbl::hlsl::testing::relativeApproxCompare(lhs, rhs, maxAllowedDifference);
    }

    const size_t m_testIterationCount;
    const uint32_t m_testBatchCount;
    static constexpr size_t m_WorkgroupSize = 256u;
    // seed will change after every call to performTestsAndVerifyResults()
    std::mt19937 m_mersenneTwister;
    uint32_t m_seed;
    std::ofstream m_logFile;
};

#endif