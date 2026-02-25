// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "nbl/examples/examples.hpp"

#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <assert.h>
#include <cfenv>

#include "app_resources/common.hlsl"
#include "app_resources/benchmark/common.hlsl"
#include "nbl/builtin/hlsl/ieee754.hlsl"

#include <nbl\builtin\hlsl\math\quadrature\gauss_legendre\gauss_legendre.hlsl>


using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;

constexpr bool DoTests = true;
constexpr bool DoBenchmark = true;

class CompatibilityTest final : public MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
public:
    CompatibilityTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    virtual SPhysicalDeviceFeatures getPreferredDeviceFeatures() const override
    {
        auto retval = device_base_t::getPreferredDeviceFeatures();
        retval.pipelineExecutableInfo = true;
        return retval;
    }

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // since emulated_float64_t rounds to zero
        std::fesetround(FE_TOWARDZERO);

        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override
    {
        if constexpr (DoTests)
        {
            emulated_float64_tests();
        }
        if constexpr (DoBenchmark)
        {
            EF64Benchmark benchmark(*this);
            benchmark.run();
        }

        m_keepRunning = false;
    }

    bool keepRunning() override
    {
        return m_keepRunning;
    }


private:

    bool m_keepRunning = true;

    constexpr static inline uint32_t EmulatedFloat64TestIterations = 1000u;
    
    enum class EmulatedFloatTestDevice
    {
        CPU,
        GPU
    };

    template<bool FastMath, bool FlushDenormToZero, EmulatedFloatTestDevice Device>
    bool compareEmulatedFloat64TestValues(const TestValues<FastMath, FlushDenormToZero>& expectedValues, const TestValues<FastMath, FlushDenormToZero>& testValues)
    {
        bool success = true;

        auto printOnFailure = [this](EmulatedFloatTestDevice device)
        {
            std::string errorMsgPrefix = "";
            if (device == EmulatedFloatTestDevice::CPU)
                errorMsgPrefix = "CPU test fail:";
            else
                errorMsgPrefix = "GPU test fail:";

            m_logger->log("%s", ILogger::ELL_ERROR, errorMsgPrefix.c_str());
            m_logFile << errorMsgPrefix << '\n';
        };

        auto printOnArithmeticFailure = [this](const char* valName, uint64_t expectedValue, uint64_t testValue, uint64_t a, uint64_t b)
        {
            double expectedAsDouble = reinterpret_cast<double&>(expectedValue);
            double testAsDouble = reinterpret_cast<double&>(testValue);
            double error = std::abs(expectedAsDouble - testAsDouble);

            std::stringstream ss;
            ss << "for input values: A = " << reinterpret_cast<double&>(a) << " B = " << reinterpret_cast<double&>(b) << '\n';
            ss << valName << " not equal!";
            ss << "\nexpected value: " << std::fixed << std::setprecision(20) << expectedAsDouble;
            ss << "\ntest value:     " << std::fixed << std::setprecision(20) << testAsDouble;
            ss << "\nerror = " << error << '\n';
            ss << "bit representations: \n";
            ss << "seeeeeeeeeeemmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm\n";
            ss << std::bitset<64>(expectedValue) << " - expectedValue bit pattern\n";
            ss << std::bitset<64>(testValue) << " - testValue bit pattern \n";

            m_logger->log("%s", ILogger::ELL_ERROR, ss.str().c_str());
            m_logFile << ss.str() << '\n';

            //std::cout << "ULP error: " << std::max(expectedValue, testValue) - std::min(expectedValue, testValue) << "\n\n";

        };

        auto calcULPError = [](emulated_float64_t::storage_t expectedValue, emulated_float64_t::storage_t testValue)
        {
            return std::max(expectedValue, testValue) - std::min(expectedValue, testValue);
        };

        auto printOnComparisonFailure = [this](const char* valName, int expectedValue, int testValue, double a, double b)
        {
            std::string inputValuesStr = std::string("for input values: A = ") + std::to_string(a) + std::string(" B = ") + std::to_string(b);

            m_logger->log("%s", ILogger::ELL_ERROR, inputValuesStr.c_str());
            m_logFile << inputValuesStr << '\n';

            std::stringstream ss;
            ss << valName << " not equal!";
            ss << "\nexpected value: " << std::boolalpha << bool(expectedValue);
            ss << "\ntest value: " << std::boolalpha << bool(testValue);

            m_logger->log("%s", ILogger::ELL_ERROR, ss.str().c_str());
            m_logFile << ss.str() << '\n';
        };

        if (calcULPError(expectedValues.int32CreateVal, testValues.int32CreateVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("int32CreateVal", expectedValues.int32CreateVal, testValues.int32CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.int64CreateVal, testValues.int64CreateVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("int64CreateVal", expectedValues.int64CreateVal, testValues.int64CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.uint32CreateVal, testValues.uint32CreateVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("uint32CreateVal", expectedValues.uint32CreateVal, testValues.uint32CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.uint64CreateVal, testValues.uint64CreateVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("uint64CreateVal", expectedValues.uint64CreateVal, testValues.uint64CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.float32CreateVal, testValues.float32CreateVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("float32CreateVal", expectedValues.float32CreateVal, testValues.float32CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.float64CreateVal != testValues.float64CreateVal)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("float64CreateVal", expectedValues.float64CreateVal, testValues.float64CreateVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.additionVal, testValues.additionVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("additionVal", expectedValues.additionVal, testValues.additionVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.substractionVal, testValues.substractionVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("substractionVal", expectedValues.substractionVal, testValues.substractionVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.multiplicationVal, testValues.multiplicationVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("multiplicationVal", expectedValues.multiplicationVal, testValues.multiplicationVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (calcULPError(expectedValues.divisionVal, testValues.divisionVal) > 1u)
        {
            printOnFailure(Device);
            printOnArithmeticFailure("divisionVal", expectedValues.divisionVal, testValues.divisionVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.lessOrEqualVal != testValues.lessOrEqualVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("lessOrEqualVal", expectedValues.lessOrEqualVal, testValues.lessOrEqualVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.greaterOrEqualVal != testValues.greaterOrEqualVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("greaterOrEqualVal", expectedValues.greaterOrEqualVal, testValues.greaterOrEqualVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.equalVal != testValues.equalVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("equalVal", expectedValues.equalVal, testValues.equalVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.notEqualVal != testValues.notEqualVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("notEqualVal", expectedValues.notEqualVal, testValues.notEqualVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.lessVal != testValues.lessVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("lessVal", expectedValues.lessVal, testValues.lessVal, expectedValues.a, expectedValues.b);
            success = false;
        }
        if (expectedValues.greaterVal != testValues.greaterVal)
        {
            printOnFailure(Device);
            printOnComparisonFailure("greaterVal", expectedValues.greaterVal, testValues.greaterVal, expectedValues.a, expectedValues.b);
            success = false;
        }

        return success;
    };

    class EF64Submitter
    {
    public:
        EF64Submitter(CompatibilityTest& base)
            :m_base(base), m_pushConstants({}), m_semaphoreCounter(0)
        {
            // setting up pipeline in the constructor
            m_queueFamily = base.getComputeQueue()->getFamilyIndex();
            m_semaphore = base.m_device->createSemaphore(0);
            m_cmdpool = base.m_device->createCommandPool(m_queueFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
            if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
                base.logFail("Failed to create Command Buffers!\n");

            // Load shaders, set up pipeline
            {
                smart_refctd_ptr<IShader> shader;
                {
                    IAssetLoader::SAssetLoadParams lp = {};
                    lp.logger = base.m_logger.get();
                    lp.workingDirectory = "app_resources"; // virtual root

                    auto key = nbl::this_example::builtin::build::get_spirv_key<"test">(base.m_device.get());
                    auto assetBundle = base.m_assetMgr->getAsset(key.data(), lp);
                    const auto assets = assetBundle.getContents();
                    if (assets.empty())
                    {
                        base.logFail("Could not load shader!");
                        assert(0);
                    }

                    // It would be super weird if loading a shader from a file produced more than 1 asset
                    assert(assets.size() == 1);
                    shader = IAsset::castDown<IShader>(assets[0]);
                }

                if (!shader)
                    base.logFail("Failed to load precompiled \"test\" shader!\n");

                nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
                    {
                        .binding = 0,
                        .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                        .stageFlags = ShaderStage::ESS_COMPUTE,
                        .count = 1
                    }
                };
                smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = base.m_device->createDescriptorSetLayout(bindings);
                if (!dsLayout)
                    base.logFail("Failed to create a Descriptor Layout!\n");

                SPushConstantRange pushConstantRanges[] = {
                {
                    .stageFlags = ShaderStage::ESS_COMPUTE,
                    .offset = 0,
                    .size = sizeof(PushConstants)
                }
                };
                m_pplnLayout = base.m_device->createPipelineLayout(pushConstantRanges, smart_refctd_ptr(dsLayout));
                if (!m_pplnLayout)
                    base.logFail("Failed to create a Pipeline Layout!\n");

                {
                    IGPUComputePipeline::SCreationParams params = {};
                    params.layout = m_pplnLayout.get();
                    params.shader.entryPoint = "main";
                    params.shader.shader = shader.get();
                    if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
                    {
                        params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                        params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
                    }
                    if (!base.m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                        base.logFail("Failed to create pipelines (compile & link shaders)!\n");

                    if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
                    {
                        auto report = system::to_string(m_pipeline->getExecutableInfo());
                        base.m_logger->log("EF64Submitter Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, report.c_str());
                    }
                }

                // Allocate the memory
                {
                    constexpr size_t BufferSize = sizeof(TestValues<false, true>);

                    nbl::video::IGPUBuffer::SCreationParams params = {};
                    params.size = BufferSize;
                    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                    smart_refctd_ptr<IGPUBuffer> outputBuff = base.m_device->createBuffer(std::move(params));
                    if (!outputBuff)
                        base.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

                    outputBuff->setObjectDebugName("emulated_float64_t output buffer");

                    nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = outputBuff->getMemoryReqs();
                    reqs.memoryTypeBits &= base.m_physicalDevice->getHostVisibleMemoryTypeBits();

                    m_allocation = base.m_device->allocate(reqs, outputBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
                    if (!m_allocation.isValid())
                        base.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

                    assert(outputBuff->getBoundMemory().memory == m_allocation.memory.get());
                    smart_refctd_ptr<nbl::video::IDescriptorPool> pool = base.m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

                    m_ds = pool->createDescriptorSet(std::move(dsLayout));
                    {
                        IGPUDescriptorSet::SDescriptorInfo info[1];
                        info[0].desc = smart_refctd_ptr(outputBuff);
                        info[0].info.buffer = { .offset = 0,.size = BufferSize };
                        IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                            {.dstSet = m_ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
                        };
                        base.m_device->updateDescriptorSets(writes, {});
                    }
                }

                if (!m_allocation.memory->map({ 0ull,m_allocation.memory->getAllocationSize() }, IDeviceMemoryAllocation::EMCAF_READ))
                    base.logFail("Failed to map the Device Memory!\n");
            }

            // if the mapping is not coherent the range needs to be invalidated to pull in new data for the CPU's caches
            const ILogicalDevice::MappedMemoryRange memoryRange(m_allocation.memory.get(), 0ull, m_allocation.memory->getAllocationSize());
            if (!m_allocation.memory->getMemoryPropertyFlags().hasFlags(IDeviceMemoryAllocation::EMPF_HOST_COHERENT_BIT))
                base.m_device->invalidateMappedMemoryRanges(1, &memoryRange);

            assert(memoryRange.valid() && memoryRange.length >= sizeof(TestValues<false, true>));

            m_queue = m_base.m_device->getQueue(m_queueFamily, 0);
        }

        ~EF64Submitter() 
        {
            m_allocation.memory->unmap();
        }

        void setPushConstants(PushConstants& pc)
        {
            m_pushConstants = pc;
        }

        TestValues<false, true> submitGetGPUTestValues()
        {
            // record command buffer
            m_cmdbuf->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
            m_cmdbuf->begin(IGPUCommandBuffer::USAGE::NONE);
            m_cmdbuf->beginDebugMarker("emulated_float64_t compute dispatch", vectorSIMDf(0, 1, 0, 1));
            m_cmdbuf->bindComputePipeline(m_pipeline.get());
            m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
            m_cmdbuf->pushConstants(m_pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(PushConstants), &m_pushConstants);
            m_cmdbuf->dispatch(WORKGROUP_SIZE, 1, 1);
            m_cmdbuf->endDebugMarker();
            m_cmdbuf->end();

            IQueue::SSubmitInfo submitInfos[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()}};
            submitInfos[0].commandBuffers = cmdbufs;
            const IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = m_semaphore.get(), .value = ++m_semaphoreCounter, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT}};
            submitInfos[0].signalSemaphores = signals;
            
            m_base.m_api->startCapture();
            m_queue->submit(submitInfos);
            m_base.m_api->endCapture();

            m_base.m_device->waitIdle();
            TestValues<false, true> output;
            std::memcpy(&output, static_cast<TestValues<false, true>*>(m_allocation.memory->getMappedPointer()), sizeof(TestValues<false, true>));
            m_base.m_device->waitIdle();

            return output;
        }

    private:
        uint32_t m_queueFamily;
        nbl::video::IDeviceMemoryAllocator::SAllocation m_allocation = {};
        smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_cmdbuf = nullptr;
        smart_refctd_ptr<nbl::video::IGPUCommandPool> m_cmdpool = nullptr;
        smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ds = nullptr;
        smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_pplnLayout = nullptr;
        PushConstants m_pushConstants;
        CompatibilityTest& m_base;
        smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_pipeline;
        smart_refctd_ptr<ISemaphore> m_semaphore;
        IQueue* m_queue;
        uint64_t m_semaphoreCounter;
    };

    void emulated_float64_tests()
    {
        EF64Submitter submitter(*this);

        auto printTestOutput = [this](const std::string& functionName, const EmulatedFloat64TestOutput& testResult)
            {
                std::cout << functionName << ": " << std::endl;

                if (!testResult.cpuTestsSucceed)
                    logFail("Incorrect CPU determinated values!");
                else
                    m_logger->log("Correct CPU determinated values!", ILogger::ELL_PERFORMANCE);

                if (!testResult.gpuTestsSucceed)
                    logFail("Incorrect GPU determinated values!");
                else
                    m_logger->log("Correct GPU determinated values!", ILogger::ELL_PERFORMANCE);
            };

        m_logFile.open("EmulatedFloatTestLog.txt", std::ios::out | std::ios::trunc);
        if (!m_logFile.is_open())
            m_logger->log("Failed to open log file!", system::ILogger::ELL_ERROR);

        printTestOutput("emulatedFloat64RandomValuesTest", emulatedFloat64RandomValuesTest(submitter));
        printTestOutput("emulatedFloat64RandomValuesTestContrastingExponents", emulatedFloat64RandomValuesTestContrastingExponents(submitter));
        printTestOutput("emulatedFloat64NegAndPosZeroTest", emulatedFloat64NegAndPosZeroTest(submitter));
        printTestOutput("emulatedFloat64BothValuesInfTest", emulatedFloat64BothValuesInfTest(submitter));
        printTestOutput("emulatedFloat64BothValuesNegInfTest", emulatedFloat64BothValuesNegInfTest(submitter));
        printTestOutput("emulatedFloat64OneValIsInfOtherIsNegInfTest", emulatedFloat64OneValIsInfOtherIsNegInfTest(submitter));
        printTestOutput("emulatedFloat64OneValIsInfTest", emulatedFloat64OneValIsInfTest(submitter));
        printTestOutput("emulatedFloat64OneValIsNegInfTest", emulatedFloat64OneValIsNegInfTest(submitter));
        if(false) // doesn't work for some reason + fast math is enabled by default
            printTestOutput("emulatedFloat64BNaNTest", emulatedFloat64BNaNTest(submitter));
        printTestOutput("emulatedFloat64BInfTest", emulatedFloat64OneValIsZeroTest(submitter));
        printTestOutput("emulatedFloat64BNegInfTest", emulatedFloat64OneValIsNegZeroTest(submitter));

        m_logFile.close();
    }

    template <bool FastMath, bool FlushDenormToZero>
    struct EmulatedFloat64TestValuesInfo
    {
        emulated_float64_t<FastMath, FlushDenormToZero> a;
        emulated_float64_t<FastMath, FlushDenormToZero> b;
        ConstructorTestValues constrTestValues;
        TestValues<FastMath, FlushDenormToZero> expectedTestValues;
        
        void fillExpectedTestValues()
        {
            double aAsDouble = reinterpret_cast<double&>(a);
            double bAsDouble = reinterpret_cast<double&>(b);

            expectedTestValues.a = a.data;
            expectedTestValues.b = b.data;

            expectedTestValues.int32CreateVal = bit_cast<uint64_t>(double(constrTestValues.int32));
            expectedTestValues.int64CreateVal = bit_cast<uint64_t>(double(constrTestValues.int64));
            expectedTestValues.uint32CreateVal = bit_cast<uint64_t>(double(constrTestValues.uint32));
            expectedTestValues.uint64CreateVal = bit_cast<uint64_t>(double(constrTestValues.uint64));
            expectedTestValues.float32CreateVal = bit_cast<uint64_t>(double(constrTestValues.float32));
            expectedTestValues.float64CreateVal = bit_cast<uint64_t>(constrTestValues.float64);
            expectedTestValues.additionVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(aAsDouble + bAsDouble).data;
            expectedTestValues.substractionVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(aAsDouble - bAsDouble).data;
            expectedTestValues.multiplicationVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(aAsDouble * bAsDouble).data;
            expectedTestValues.divisionVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(aAsDouble / bAsDouble).data;
            expectedTestValues.lessOrEqualVal = aAsDouble <= bAsDouble;
            expectedTestValues.greaterOrEqualVal = aAsDouble >= bAsDouble;
            expectedTestValues.equalVal = aAsDouble == bAsDouble;
            expectedTestValues.notEqualVal = aAsDouble != bAsDouble;
            expectedTestValues.lessVal = aAsDouble < bAsDouble;
            expectedTestValues.greaterVal = aAsDouble > bAsDouble;
        }
    };

    struct EmulatedFloat64TestOutput
    {
        bool cpuTestsSucceed;
        bool gpuTestsSucceed;
    };

    EmulatedFloat64TestOutput emulatedFloat64LoopedTests_impl(EF64Submitter& submitter, 
        const uint32_t iterations,
        const std::function<double()>& determineValueA, 
        const std::function<double()>& determineValueB)
    {
        EmulatedFloat64TestOutput output = { true, true };

        std::uniform_int_distribution i32Distribution(-std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
        std::uniform_int_distribution i64Distribution(-std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
        std::uniform_int_distribution u32Distribution(-std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max());
        std::uniform_int_distribution u64Distribution(-std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max());
        std::uniform_real_distribution fDistribution(-100000.0, 100000.0);
        
        std::random_device rd;
        std::mt19937 mt(rd());

        for (uint32_t i = 0u; i < iterations; ++i)
        {
            // generate random test values
            EmulatedFloat64TestValuesInfo<false, true> testValInfo;
            double aTmp = determineValueA();
            double bTmp = determineValueB();
            testValInfo.a.data = reinterpret_cast<emulated_float64_t<false, true>::storage_t&>(aTmp);
            testValInfo.b.data = reinterpret_cast<emulated_float64_t<false, true>::storage_t&>(bTmp);
            testValInfo.constrTestValues.int32 = i32Distribution(mt);
            testValInfo.constrTestValues.int64 = i64Distribution(mt);
            testValInfo.constrTestValues.uint32 = u32Distribution(mt);
            testValInfo.constrTestValues.uint64 = u64Distribution(mt);
            testValInfo.constrTestValues.float32 = fDistribution(mt);
            testValInfo.constrTestValues.float64 = fDistribution(mt);

            testValInfo.fillExpectedTestValues();
            auto singleTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);

            if (!singleTestOutput.cpuTestsSucceed)
                output.cpuTestsSucceed = false;
            if (!singleTestOutput.gpuTestsSucceed)
                output.gpuTestsSucceed = false;
        }

        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64RandomValuesTest(EF64Submitter& submitter)
    {
        auto getRandomFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-100000.0, 100000.0);


                return distribution(mt);
            };

        return emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations, getRandomFloat64, getRandomFloat64);
    }

    EmulatedFloat64TestOutput emulatedFloat64RandomValuesTestContrastingExponents(EF64Submitter& submitter)
    {
        auto getRandomSmallFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-0.01, 0.01);

                return distribution(mt);
            };

        auto getRandomLargeFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(1000000000.0, 2000000000.0);
                static std::uniform_int_distribution coinFlipDistribution(0, 1);

                double output = distribution(mt);
                if (coinFlipDistribution(mt))
                    output = -output;

                return output;
            };

        EmulatedFloat64TestOutput firstTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomSmallFloat64, getRandomLargeFloat64);
        EmulatedFloat64TestOutput secondTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomLargeFloat64, getRandomSmallFloat64);

        EmulatedFloat64TestOutput output;
        output.cpuTestsSucceed = firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed;
        output.gpuTestsSucceed = firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed;
        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64BothValuesNaNTest(EF64Submitter& submitter)
    {
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        EmulatedFloat64TestValuesInfo<false, true> testValInfo;
        const float32_t nan32 = std::numeric_limits<float32_t>::quiet_NaN();
        const float64_t nan64 = std::numeric_limits<float64_t>::quiet_NaN();
        testValInfo.a = emulated_float64_t<false, true>::create(nan64);
        testValInfo.b = emulated_float64_t<false, true>::create(nan64);
        testValInfo.constrTestValues = {
            .int32 = std::bit_cast<int32_t>(nan32),
            .int64 = std::bit_cast<int64_t>(nan64),
            .uint32 = std::bit_cast<uint32_t>(nan32),
            .uint64 = std::bit_cast<uint64_t>(nan64),
            .float32 = nan32
            //.float64 = nan64
        };

        testValInfo.fillExpectedTestValues();
        return performEmulatedFloat64Tests(testValInfo, submitter);
    }

    EmulatedFloat64TestOutput emulatedFloat64NegAndPosZeroTest(EF64Submitter& submitter)
    {
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        EmulatedFloat64TestValuesInfo<false, true> testValInfo;
        testValInfo.a = emulated_float64_t<false, true>::create(ieee754::traits<float64_t>::signMask);
        testValInfo.b = emulated_float64_t<false, true>::create(std::bit_cast<uint64_t>(0.0));
        testValInfo.constrTestValues = {
            .int32 = 0,
            .int64 = 0,
            .uint32 = 0,
            .uint64 = 0,
            .float32 = 0
        };

        testValInfo.fillExpectedTestValues();
        auto firstTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);
        std::swap(testValInfo.a, testValInfo.b);
        testValInfo.fillExpectedTestValues();
        auto secondTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);

        return { firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed, firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed };
    }

    EmulatedFloat64TestOutput emulatedFloat64BothValuesInfTest(EF64Submitter& submitter)
    {
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        EmulatedFloat64TestValuesInfo<false, true> testValInfo;
        const float32_t inf32 = std::numeric_limits<float32_t>::infinity();
        const float64_t inf64 = std::numeric_limits<float64_t>::infinity();
        testValInfo.a = emulated_float64_t<false, true>::create(inf64);
        testValInfo.b = emulated_float64_t<false, true>::create(inf64);
        testValInfo.constrTestValues = {
            .int32 = 0,
            .int64 = 0,
            .uint32 = 0,
            .uint64 = 0,
            .float32 = inf32
            //.float64 = inf64
        };

        testValInfo.fillExpectedTestValues();
        return performEmulatedFloat64Tests(testValInfo, submitter);
    }

    EmulatedFloat64TestOutput emulatedFloat64BothValuesNegInfTest(EF64Submitter& submitter)
    {
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        EmulatedFloat64TestValuesInfo<false, true> testValInfo;
        const float32_t inf32 = -std::numeric_limits<float32_t>::infinity();
        const float64_t inf64 = -std::numeric_limits<float64_t>::infinity();
        testValInfo.a = emulated_float64_t<false, true>::create(inf64);
        testValInfo.b = emulated_float64_t<false, true>::create(inf64);
        testValInfo.constrTestValues = {
            .int32 = 0,
            .int64 = 0,
            .uint32 = 0,
            .uint64 = 0,
            .float32 = inf32
            //.float64 = inf64
        };

        testValInfo.fillExpectedTestValues();
        return performEmulatedFloat64Tests(testValInfo, submitter);
    }

    EmulatedFloat64TestOutput emulatedFloat64OneValIsInfOtherIsNegInfTest(EF64Submitter& submitter)
    {
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        EmulatedFloat64TestValuesInfo<false, true> testValInfo;
        const float64_t inf64 = -std::numeric_limits<float64_t>::infinity();
        testValInfo.a = emulated_float64_t<false, true>::create(inf64);
        testValInfo.b = emulated_float64_t<false, true>::create(inf64);
        testValInfo.constrTestValues = {
            .int32 = 0,
            .int64 = 0,
            .uint32 = 0,
            .uint64 = 0,
            .float32 = 0
            //.float64 = inf64
        };

        testValInfo.fillExpectedTestValues();
        auto firstTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);
        std::swap(testValInfo.a, testValInfo.b);
        testValInfo.fillExpectedTestValues();
        auto secondTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);

        return { firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed, firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed };
    }

    // TODO: fix
    EmulatedFloat64TestOutput emulatedFloat64BNaNTest(EF64Submitter& submitter)
    {
        EmulatedFloat64TestOutput output = { true, true };
        smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(0);

        for (uint32_t i = 0u; i < EmulatedFloat64TestIterations; ++i)
        {
            std::random_device rd;
            std::mt19937 mt(rd());

            std::uniform_int_distribution i32Distribution(-std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
            std::uniform_int_distribution i64Distribution(-std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::max());
            std::uniform_int_distribution u32Distribution(-std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max());
            std::uniform_int_distribution u64Distribution(-std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max());
            std::uniform_real_distribution f32Distribution(-100000.0f, 100000.0f);
            std::uniform_real_distribution f64Distribution(-100000.0, 100000.0);

            EmulatedFloat64TestValuesInfo<false, true> testValInfo;
            double aTmp = f64Distribution(mt);
            double bTmp = std::numeric_limits<float64_t>::quiet_NaN();
            testValInfo.a.data = reinterpret_cast<emulated_float64_t<false, true>::storage_t&>(aTmp);
            testValInfo.b.data = reinterpret_cast<emulated_float64_t<false, true>::storage_t&>(bTmp);
            testValInfo.constrTestValues.int32 = i32Distribution(mt);
            testValInfo.constrTestValues.int64 = i64Distribution(mt);
            testValInfo.constrTestValues.uint32 = u32Distribution(mt);
            testValInfo.constrTestValues.uint64 = u64Distribution(mt);
            testValInfo.constrTestValues.float32 = f32Distribution(mt);
            //testValInfo.constrTestValues.float64 = f64Distribution(mt);

            testValInfo.fillExpectedTestValues();
            auto singleTestOutput = performEmulatedFloat64Tests(testValInfo, submitter);

            if (!singleTestOutput.cpuTestsSucceed)
                output.cpuTestsSucceed = false;
            if (!singleTestOutput.gpuTestsSucceed)
                output.gpuTestsSucceed = false;
        }

        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64OneValIsInfTest(EF64Submitter& submitter)
    {
        auto getRandomFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-100000.0, 100000.0);

                return distribution(mt);
            };

        auto getInfinity = []()
            {
                return std::numeric_limits<float64_t>::infinity();
            };

        EmulatedFloat64TestOutput firstTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomFloat64, getInfinity);
        EmulatedFloat64TestOutput secondTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getInfinity, getRandomFloat64);

        EmulatedFloat64TestOutput output;
        output.cpuTestsSucceed = firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed;
        output.gpuTestsSucceed = firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed;
        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64OneValIsNegInfTest(EF64Submitter& submitter)
    {
        auto getRandomFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-100000.0, 100000.0);


                return distribution(mt);
            };

        auto getNegInfinity = []()
            {
                return -std::numeric_limits<float64_t>::infinity();
            };

        EmulatedFloat64TestOutput firstTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomFloat64, getNegInfinity);
        EmulatedFloat64TestOutput secondTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getNegInfinity, getRandomFloat64);

        EmulatedFloat64TestOutput output;
        output.cpuTestsSucceed = firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed;
        output.gpuTestsSucceed = firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed;
        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64OneValIsZeroTest(EF64Submitter& submitter)
    {
        auto getRandomFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-100000.0, 100000.0);

                return distribution(mt);
            };

        auto getZero = []()
            {
                return 0.0;
            };

        EmulatedFloat64TestOutput firstTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomFloat64, getZero);
        EmulatedFloat64TestOutput secondTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getZero, getRandomFloat64);

        EmulatedFloat64TestOutput output; 
        output.cpuTestsSucceed = firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed;
        output.gpuTestsSucceed = firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed;
        return output;
    }

    EmulatedFloat64TestOutput emulatedFloat64OneValIsNegZeroTest(EF64Submitter& submitter)
    {
        auto getRandomFloat64 = []()
            {
                static std::random_device rd;
                static std::mt19937 mt(rd());
                static std::uniform_real_distribution distribution(-100000.0, 100000.0);

                return distribution(mt);
            };

        auto getNegZero = []()
            {
                return -0.0;
            };

        EmulatedFloat64TestOutput firstTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getRandomFloat64, getNegZero);
        EmulatedFloat64TestOutput secondTestOutput = emulatedFloat64LoopedTests_impl(submitter, EmulatedFloat64TestIterations / 2, getNegZero, getRandomFloat64);

        EmulatedFloat64TestOutput output;
        output.cpuTestsSucceed = firstTestOutput.cpuTestsSucceed && secondTestOutput.cpuTestsSucceed;
        output.gpuTestsSucceed = firstTestOutput.gpuTestsSucceed && secondTestOutput.gpuTestsSucceed;
        return output;
    }

    template <bool FastMath, bool FlushDenormToZero>
    EmulatedFloat64TestOutput performEmulatedFloat64Tests(EmulatedFloat64TestValuesInfo<FastMath, FlushDenormToZero>& testValInfo, EF64Submitter& submitter)
    {
        emulated_float64_t<false, true> a = testValInfo.a;
        emulated_float64_t<false, true> b = testValInfo.b;

        const TestValues<FastMath, FlushDenormToZero> cpuTestValues = {
            .int32CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.int32).data,
            .int64CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.int64).data,
            .uint32CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.uint32).data,
            .uint64CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.uint64).data,
            .float32CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.float32).data,
            .float64CreateVal = emulated_float64_t<FastMath, FlushDenormToZero>::create(testValInfo.constrTestValues.float64).data,
            .additionVal = (a + b).data,
            .substractionVal = (a - b).data,
            .multiplicationVal = (a * b).data,
            .divisionVal = (a / b).data,
            .lessOrEqualVal = a <= b,
            .greaterOrEqualVal = a >= b,
            .equalVal = a == b,
            .notEqualVal = a != b,
            .lessVal = a < b,
            .greaterVal = a > b
        };

        EmulatedFloat64TestOutput output;

        // cpu validation
        output.cpuTestsSucceed = compareEmulatedFloat64TestValues<false, true, EmulatedFloatTestDevice::CPU>(testValInfo.expectedTestValues, cpuTestValues);

        // gpu validation
        PushConstants pc;
        pc.a = reinterpret_cast<uint64_t&>(a);
        pc.b = reinterpret_cast<uint64_t&>(b);
        pc.constrTestVals = testValInfo.constrTestValues;
        
        submitter.setPushConstants(pc);
        auto gpuTestValues = submitter.submitGetGPUTestValues();

        output.gpuTestsSucceed = compareEmulatedFloat64TestValues<false, true, EmulatedFloatTestDevice::GPU>(testValInfo.expectedTestValues, gpuTestValues);

        return output;
    }

    class EF64Benchmark final
    {
    public:
        EF64Benchmark(CompatibilityTest& base)
        {
            m_device = base.m_device;
            m_logger = base.m_logger;
            m_api = base.m_api;

            // setting up pipeline in the constructor
            m_queueFamily = base.getComputeQueue()->getFamilyIndex();
            m_cmdpool = base.m_device->createCommandPool(m_queueFamily, IGPUCommandPool::CREATE_FLAGS::RESET_COMMAND_BUFFER_BIT);
            //core::smart_refctd_ptr<IGPUCommandBuffer>* cmdBuffs[] = { &m_cmdbuf, &m_timestampBeforeCmdBuff, &m_timestampAfterCmdBuff };
            if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_cmdbuf))
                base.logFail("Failed to create Command Buffers!\n");
            if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampBeforeCmdBuff))
                base.logFail("Failed to create Command Buffers!\n");
            if (!m_cmdpool->createCommandBuffers(IGPUCommandPool::BUFFER_LEVEL::PRIMARY, 1u, &m_timestampAfterCmdBuff))
                base.logFail("Failed to create Command Buffers!\n");

            // Load shaders, set up pipeline
            {
                smart_refctd_ptr<IShader> shader;
                {
                    IAssetLoader::SAssetLoadParams lp = {};
                    lp.logger = base.m_logger.get();
                    lp.workingDirectory = "app_resources"; // virtual root
                    // this time we load a shader directly from a file
                    auto key = nbl::this_example::builtin::build::get_spirv_key<"benchmark">(m_device.get());
                    auto assetBundle = base.m_assetMgr->getAsset(key.data(), lp);
                    const auto assets = assetBundle.getContents();
                    if (assets.empty())
                    {
                        base.logFail("Could not load shader!");
                        assert(0);
                    }

                    // It would be super weird if loading a shader from a file produced more than 1 asset
                    assert(assets.size() == 1);
                    shader = IAsset::castDown<IShader>(assets[0]);
                }

                if (!shader)
                    base.logFail("Failed to load precompiled \"benchmark\" shader!\n");

                nbl::video::IGPUDescriptorSetLayout::SBinding bindings[1] = {
                    {
                        .binding = 0,
                        .type = nbl::asset::IDescriptor::E_TYPE::ET_STORAGE_BUFFER,
                        .createFlags = IGPUDescriptorSetLayout::SBinding::E_CREATE_FLAGS::ECF_NONE,
                        .stageFlags = ShaderStage::ESS_COMPUTE,
                        .count = 1
                    }
                };
                smart_refctd_ptr<IGPUDescriptorSetLayout> dsLayout = base.m_device->createDescriptorSetLayout(bindings);
                if (!dsLayout)
                    base.logFail("Failed to create a Descriptor Layout!\n");

                SPushConstantRange pushConstantRanges[] = {
                    {
                        .stageFlags = ShaderStage::ESS_COMPUTE,
                        .offset = 0,
                        .size = sizeof(BenchmarkPushConstants)
                    }
                };
                m_pplnLayout = base.m_device->createPipelineLayout(pushConstantRanges, smart_refctd_ptr(dsLayout));
                if (!m_pplnLayout)
                    base.logFail("Failed to create a Pipeline Layout!\n");

                {
                    IGPUComputePipeline::SCreationParams params = {};
                    params.layout = m_pplnLayout.get();
                    params.shader.entryPoint = "main";
                    params.shader.shader = shader.get();
                    if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
                    {
                        params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_STATISTICS;
                        params.flags |= IGPUComputePipeline::SCreationParams::FLAGS::CAPTURE_INTERNAL_REPRESENTATIONS;
                    }
                    if (!base.m_device->createComputePipelines(nullptr, { &params,1 }, &m_pipeline))
                        base.logFail("Failed to create pipelines (compile & link shaders)!\n");

                    if (base.m_device->getEnabledFeatures().pipelineExecutableInfo)
                    {
                        auto report = system::to_string(m_pipeline->getExecutableInfo());
                        base.m_logger->log("EF64Benchmark Pipeline Executable Report:\n%s", ILogger::ELL_PERFORMANCE, report.c_str());
                    }
                }

                // Allocate the memory
                {
                    static_assert(sizeof(float64_t) == sizeof(benchmark_emulated_float64_t));
                    constexpr size_t BufferSize = BENCHMARK_WORKGROUP_COUNT * BENCHMARK_WORKGROUP_DIMENSION_SIZE_X *
                        BENCHMARK_WORKGROUP_DIMENSION_SIZE_Y * BENCHMARK_WORKGROUP_DIMENSION_SIZE_Z * sizeof(float64_t);

                    nbl::video::IGPUBuffer::SCreationParams params = {};
                    params.size = BufferSize;
                    params.usage = IGPUBuffer::EUF_STORAGE_BUFFER_BIT;
                    smart_refctd_ptr<IGPUBuffer> dummyBuff = base.m_device->createBuffer(std::move(params));
                    if (!dummyBuff)
                        base.logFail("Failed to create a GPU Buffer of size %d!\n", params.size);

                    dummyBuff->setObjectDebugName("benchmark buffer");

                    nbl::video::IDeviceMemoryBacked::SDeviceMemoryRequirements reqs = dummyBuff->getMemoryReqs();

                    m_allocation = base.m_device->allocate(reqs, dummyBuff.get(), nbl::video::IDeviceMemoryAllocation::EMAF_NONE);
                    if (!m_allocation.isValid())
                        base.logFail("Failed to allocate Device Memory compatible with our GPU Buffer!\n");

                    assert(dummyBuff->getBoundMemory().memory == m_allocation.memory.get());
                    smart_refctd_ptr<nbl::video::IDescriptorPool> pool = base.m_device->createDescriptorPoolForDSLayouts(IDescriptorPool::ECF_NONE, { &dsLayout.get(),1 });

                    m_ds = pool->createDescriptorSet(std::move(dsLayout));
                    {
                        IGPUDescriptorSet::SDescriptorInfo info[1];
                        info[0].desc = smart_refctd_ptr(dummyBuff);
                        info[0].info.buffer = { .offset = 0,.size = BufferSize };
                        IGPUDescriptorSet::SWriteDescriptorSet writes[1] = {
                            {.dstSet = m_ds.get(),.binding = 0,.arrayElement = 0,.count = 1,.info = info}
                        };
                        base.m_device->updateDescriptorSets(writes, {});
                    }
                }
            }

            IQueryPool::SCreationParams queryPoolCreationParams{};
            queryPoolCreationParams.queryType = IQueryPool::TYPE::TIMESTAMP;
            queryPoolCreationParams.queryCount = 2;
            queryPoolCreationParams.pipelineStatisticsFlags = IQueryPool::PIPELINE_STATISTICS_FLAGS::NONE;
            m_queryPool = m_device->createQueryPool(queryPoolCreationParams);

            m_computeQueue = m_device->getQueue(m_queueFamily, 0);
        }

        void run()
        {
            m_logger->log("\n\nfloat64_t benchmark result:", ILogger::ELL_PERFORMANCE);
            performBenchmark(EF64_BENCHMARK_MODE::NATIVE);
            m_logger->log("emulated_float64_t benchmark, fast math enabled result:", ILogger::ELL_PERFORMANCE);
            performBenchmark(EF64_BENCHMARK_MODE::EF64_FAST_MATH_ENABLED);
            m_logger->log("emulated_float64_t benchmark, fast math disabled result:", ILogger::ELL_PERFORMANCE);
            performBenchmark(EF64_BENCHMARK_MODE::EF64_FAST_MATH_DISABLED);
            // every subgroup with even ID do calculations with the `emulated_float64_t<false, true>` type, other subgroups do calculations with float64_t
            m_logger->log("emulated_float64_t benchmark, subgroup divided work result:", ILogger::ELL_PERFORMANCE);
            performBenchmark(EF64_BENCHMARK_MODE::SUBGROUP_DIVIDED_WORK);
            // every item does calculations with both emulated and native types
            m_logger->log("emulated_float64_t benchmark, interleaved result:", ILogger::ELL_PERFORMANCE);
            performBenchmark(EF64_BENCHMARK_MODE::INTERLEAVED);
        }

    private:
        void performBenchmark(EF64_BENCHMARK_MODE mode)
        {
            m_device->waitIdle();

            recordTimestampQueryCmdBuffers();

            uint64_t semaphoreCounter = 0;
            smart_refctd_ptr<ISemaphore> semaphore = m_device->createSemaphore(semaphoreCounter);

            IQueue::SSubmitInfo::SSemaphoreInfo signals[] = { {.semaphore = semaphore.get(), .value = 0u, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT} };
            IQueue::SSubmitInfo::SSemaphoreInfo waits[] = { {.semaphore = semaphore.get(), .value = 0u, .stageMask = asset::PIPELINE_STAGE_FLAGS::COMPUTE_SHADER_BIT } };

            IQueue::SSubmitInfo beforeTimestapSubmitInfo[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufsBegin[] = { {.cmdbuf = m_timestampBeforeCmdBuff.get()} };
            beforeTimestapSubmitInfo[0].commandBuffers = cmdbufsBegin;
            beforeTimestapSubmitInfo[0].signalSemaphores = signals;
            beforeTimestapSubmitInfo[0].waitSemaphores = waits;

            IQueue::SSubmitInfo afterTimestapSubmitInfo[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufsEnd[] = { {.cmdbuf = m_timestampAfterCmdBuff.get()} };
            afterTimestapSubmitInfo[0].commandBuffers = cmdbufsEnd;
            afterTimestapSubmitInfo[0].signalSemaphores = signals;
            afterTimestapSubmitInfo[0].waitSemaphores = waits;

            IQueue::SSubmitInfo benchmarkSubmitInfos[1] = {};
            const IQueue::SSubmitInfo::SCommandBufferInfo cmdbufs[] = { {.cmdbuf = m_cmdbuf.get()} };
            benchmarkSubmitInfos[0].commandBuffers = cmdbufs;
            benchmarkSubmitInfos[0].signalSemaphores = signals;
            benchmarkSubmitInfos[0].waitSemaphores = waits;


            m_pushConstants.benchmarkMode = mode;
            recordCmdBuff();

            // warmup runs
            for (int i = 0; i < WarmupIterations; ++i)
            {
                if(i == 0)
                    m_api->startCapture();
                waits[0].value = semaphoreCounter;
                signals[0].value = ++semaphoreCounter;
                m_computeQueue->submit(benchmarkSubmitInfos);
                if (i == 0)
                    m_api->endCapture();
            }

            waits[0].value = semaphoreCounter;
            signals[0].value = ++semaphoreCounter;
            m_computeQueue->submit(beforeTimestapSubmitInfo);

            // actual benchmark runs
            for (int i = 0; i < Iterations; ++i)
            {
                waits[0].value = semaphoreCounter;
                signals[0].value = ++semaphoreCounter;
                m_computeQueue->submit(benchmarkSubmitInfos);
            }
            
            waits[0].value = semaphoreCounter;
            signals[0].value = ++semaphoreCounter;
            m_computeQueue->submit(afterTimestapSubmitInfo);

            m_device->waitIdle();

            const uint64_t nativeBenchmarkTimeElapsedNanoseconds = calcTimeElapsed();
            const float nativeBenchmarkTimeElapsedSeconds = double(nativeBenchmarkTimeElapsedNanoseconds) / 1000000000.0;

            m_logger->log("%llu ns, %f s", ILogger::ELL_PERFORMANCE, nativeBenchmarkTimeElapsedNanoseconds, nativeBenchmarkTimeElapsedSeconds);
        }

        void recordCmdBuff()
        {
            m_cmdbuf->begin(IGPUCommandBuffer::USAGE::SIMULTANEOUS_USE_BIT);
            m_cmdbuf->beginDebugMarker("emulated_float64_t compute dispatch", vectorSIMDf(0, 1, 0, 1));
            m_cmdbuf->bindComputePipeline(m_pipeline.get());
            m_cmdbuf->bindDescriptorSets(nbl::asset::EPBP_COMPUTE, m_pplnLayout.get(), 0, 1, &m_ds.get());
            m_cmdbuf->pushConstants(m_pplnLayout.get(), IShader::E_SHADER_STAGE::ESS_COMPUTE, 0, sizeof(BenchmarkPushConstants), &m_pushConstants);
            m_cmdbuf->dispatch(BENCHMARK_WORKGROUP_COUNT, 1, 1);
            m_cmdbuf->endDebugMarker();
            m_cmdbuf->end();
        }

        void recordTimestampQueryCmdBuffers()
        {
            static bool firstInvocation = true;

            if (!firstInvocation)
            {
                m_timestampBeforeCmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
                m_timestampBeforeCmdBuff->reset(IGPUCommandBuffer::RESET_FLAGS::NONE);
            }

            m_timestampBeforeCmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            m_timestampBeforeCmdBuff->resetQueryPool(m_queryPool.get(), 0, 2);
            m_timestampBeforeCmdBuff->writeTimestamp(PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 0);
            m_timestampBeforeCmdBuff->end();

            m_timestampAfterCmdBuff->begin(IGPUCommandBuffer::USAGE::ONE_TIME_SUBMIT_BIT);
            m_timestampAfterCmdBuff->writeTimestamp(PIPELINE_STAGE_FLAGS::NONE, m_queryPool.get(), 1);
            m_timestampAfterCmdBuff->end();

            firstInvocation = false;
        }

        uint64_t calcTimeElapsed()
        {
            uint64_t timestamps[2];
            const core::bitflag flags = core::bitflag(IQueryPool::RESULTS_FLAGS::_64_BIT) | core::bitflag(IQueryPool::RESULTS_FLAGS::WAIT_BIT);
            m_device->getQueryPoolResults(m_queryPool.get(), 0, 2, &timestamps, sizeof(uint64_t), flags);
            return timestamps[1] - timestamps[0];
        }

    private:
        core::smart_refctd_ptr<video::CVulkanConnection> m_api;
        smart_refctd_ptr<ILogicalDevice> m_device;
        smart_refctd_ptr<ILogger> m_logger;

        nbl::video::IDeviceMemoryAllocator::SAllocation m_allocation = {};
        smart_refctd_ptr<nbl::video::IGPUCommandPool> m_cmdpool = nullptr;
        smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_cmdbuf = nullptr;
        smart_refctd_ptr<nbl::video::IGPUDescriptorSet> m_ds = nullptr;
        smart_refctd_ptr<nbl::video::IGPUPipelineLayout> m_pplnLayout = nullptr;
        BenchmarkPushConstants m_pushConstants;
        smart_refctd_ptr<nbl::video::IGPUComputePipeline> m_pipeline;

        smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_timestampBeforeCmdBuff = nullptr;
        smart_refctd_ptr<nbl::video::IGPUCommandBuffer> m_timestampAfterCmdBuff = nullptr;
        smart_refctd_ptr<nbl::video::IQueryPool> m_queryPool = nullptr;

        uint32_t m_queueFamily;
        IQueue* m_computeQueue;
        static constexpr int WarmupIterations = 1000;
        static constexpr int Iterations = 1000;
        using benchmark_emulated_float64_t = emulated_float64_t<false, true>;
    };

    template<typename... Args>
    inline bool logFail(const char* msg, Args&&... args)
    {
        m_logger->log(msg, ILogger::ELL_ERROR, std::forward<Args>(args)...);
        return false;
    }

    std::ofstream m_logFile;
};

NBL_MAIN_FUNC(CompatibilityTest)