#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_I_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_I_TESTER_INCLUDED_


#include "nbl/examples/examples.hpp"

#include "app_resources/common.hlsl"
#include "nbl/asset/metadata/CHLSLMetadata.h"


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
        std::string testShaderPath;

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
            logFail("Failed to create Command Buffers!\n");

        // Load shaders, set up pipeline
        core::smart_refctd_ptr<asset::IShader> shader;
        auto shaderStage = ESS_UNKNOWN;
        {
            asset::IAssetLoader::SAssetLoadParams lp = {};
            lp.logger = m_logger.get();
            lp.workingDirectory = ""; // virtual root
            auto assetBundle = m_assetMgr->getAsset(pipleineSetupData.testShaderPath, lp);
            const auto assets = assetBundle.getContents();
            if (assets.empty() || assetBundle.getAssetType() != asset::IAsset::ET_SHADER)
            {
                logFail("Could not load shader!");
                assert(0);
            }

            // It would be super weird if loading a shader from a file produced more than 1 asset
            assert(assets.size() == 1);
            core::smart_refctd_ptr<asset::IShader> source = asset::IAsset::castDown<asset::IShader>(assets[0]);
            const auto hlslMetadata = static_cast<const asset::CHLSLMetadata*>(assetBundle.getMetadata());
            shaderStage = hlslMetadata->shaderStages->front();

            auto* compilerSet = m_assetMgr->getCompilerSet();

            asset::IShaderCompiler::SCompilerOptions options = {};
            options.stage = shaderStage;
            options.preprocessorOptions.targetSpirvVersion = m_device->getPhysicalDevice()->getLimits().spirvVersion;
            options.spirvOptimizer = nullptr;
            options.debugInfoFlags |= asset::IShaderCompiler::E_DEBUG_INFO_FLAGS::EDIF_SOURCE_BIT;
            options.preprocessorOptions.sourceIdentifier = source->getFilepathHint();
            options.preprocessorOptions.logger = m_logger.get();
            options.preprocessorOptions.includeFinder = compilerSet->getShaderCompiler(source->getContentType())->getDefaultIncludeFinder();

            shader = compilerSet->compileToSPIRV(source.get(), options);
        }

        if (!shader)
            logFail("Failed to create a GPU Shader, seems the Driver doesn't like the SPIR-V we're feeding it!\n");

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
            constexpr size_t BufferSize = sizeof(InputStruct);

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
            constexpr size_t BufferSize = sizeof(OutputStruct);

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
        static constexpr float MaxAllowedError = 0.1f;
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

        m_logger->log(ss.str().c_str(), system::ILogger::ELL_ERROR);
    }

    template<typename T>
    void verifyTestVector3dValue(const std::string& memberName, const nbl::hlsl::vector<T, 3>& expectedVal, const nbl::hlsl::vector<T, 3>& testVal, const TestType testType)
    {
        static constexpr float MaxAllowedError = 0.1f;
        if (std::abs(double(expectedVal.x) - double(testVal.x)) <= MaxAllowedError &&
            std::abs(double(expectedVal.y) - double(testVal.y)) <= MaxAllowedError &&
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
            logFail("Failed to map the Device Memory!\n");

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
        m_cmdbuf->dispatch(1, 1, 1);
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
    template<typename... Args>
    inline void logFail(const char* msg, Args&&... args)
    {
        m_logger->log(msg, system::ILogger::ELL_ERROR, std::forward<Args>(args)...);
        exit(-1);
    }
};

#endif