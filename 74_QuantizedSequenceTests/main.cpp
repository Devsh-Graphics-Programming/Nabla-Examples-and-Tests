// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "app_resources/common.hlsl"

#include "CQuantizedSequenceTester.h"

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

class QuantizedSequenceTest final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
public:
    QuantizedSequenceTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        {
            CQuantizedSequenceTester::PipelineSetupData pplnSetupData;
            pplnSetupData.device = m_device;
            pplnSetupData.api = m_api;
            pplnSetupData.assetMgr = m_assetMgr;
            pplnSetupData.logger = m_logger;
            pplnSetupData.physicalDevice = m_physicalDevice;
            pplnSetupData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            pplnSetupData.shaderKey = nbl::this_example::builtin::build::get_spirv_key<"quantizedSequenceTest">(m_device.get());

            CQuantizedSequenceTester quantizedSequenceTester(8);
            quantizedSequenceTester.setupPipeline(pplnSetupData);
            if (!quantizedSequenceTester.performTestsAndVerifyResults("QuantizedSequenceTestLog.txt"))
                return false;
        }
       
        // In contrast to fences, we just need one semaphore to rule all dispatches
        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override {}

    bool keepRunning() override { return false; }
};

NBL_MAIN_FUNC(QuantizedSequenceTest)
