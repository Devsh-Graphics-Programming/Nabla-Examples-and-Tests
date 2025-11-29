// Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include <nabla.h>
#include <iostream>
#include <cstdio>
#include <assert.h>

#include "nbl/application_templates/MonoDeviceApplication.hpp"
#include "nbl/examples/common/BuiltinResourcesApplication.hpp"

#include "app_resources/common.hlsl"
#include "CTester.h"

using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::video;
using namespace nbl::examples;
using namespace nbl::application_templates;

class MortonTest final : public MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
public:
    MortonTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {
    }

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;
        
        CTester::PipelineSetupData pplnSetupData;
        pplnSetupData.device = m_device;
        pplnSetupData.api = m_api;
        pplnSetupData.assetMgr = m_assetMgr;
        pplnSetupData.logger = m_logger;
        pplnSetupData.physicalDevice = m_physicalDevice;
        pplnSetupData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
        // Some tests with mortons with emulated uint storage were cut off, it should be fine since each tested on their own produces correct results for each operator
        // Blocked by https://github.com/KhronosGroup/SPIRV-Tools/issues/6104
        {
            CTester mortonTester(100);
            pplnSetupData.testCommonDataPath = "testCommon.hlsl";
            mortonTester.setupPipeline<InputTestValues, TestValues>(pplnSetupData);
            mortonTester.performTestsAndVerifyResults();
        }

        return true;
    }

    void onAppTerminated_impl() override
    {
        m_device->waitIdle();
    }

    void workLoopBody() override
    {
        m_keepRunning = false;
    }

    bool keepRunning() override
    {
        return m_keepRunning;
    }


private:
    bool m_keepRunning = true;
};

NBL_MAIN_FUNC(MortonTest)