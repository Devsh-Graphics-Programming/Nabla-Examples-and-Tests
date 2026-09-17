// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "app_resources/common.hlsl"

#include "CQuaternionTester.h"

#include "nbl/builtin/hlsl/math/quaternions.hlsl"
#include "nbl/builtin/hlsl/approx/abs_rel.hlsl"
#include "nbl/builtin/hlsl/approx/orientation.hlsl"

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

// Deterministic checks that `RuntimeTraits` (and therefore `quaternion::create(matrix)`) accepts rotation matrices
// whose basis dot products are only approximately zero, plain and uniformly scaled.
// Ported from the camera probe 9b/9c in 09_GeometryCreator.
bool runtimeTraitsTests(system::ILogger* logger)
{
    bool pass = true;
    auto check = [&](const bool condition, const char* type, const char* what)
    {
        if (!condition)
        {
            logger->log("RuntimeTraits test failed [%s]: %s", system::ILogger::ELL_ERROR, type, what);
            pass = false;
        }
    };

    auto testType = [&]<typename F>(const char* type)
    {
        using quat_t = hlsl::math::quaternion<F>;
        using vec3_t = hlsl::vector<F, 3>;
        using mat_t = hlsl::matrix<F, 3, 3>;
        using traits_t = hlsl::math::linalg::RuntimeTraits<mat_t>;

        auto isFinite = [](const quat_t q)
        {
            for (uint32_t i = 0; i < 4; ++i)
                if (hlsl::isnan(q.data[i]) || hlsl::isinf(q.data[i]))
                    return false;
            return true;
        };

        auto testRotation = [&](const quat_t q, const char* what)
        {
            const mat_t basis = hlsl::_static_cast<mat_t>(q);
            // the cast has to produce the matrix you can `mul` with, i.e. the basis vectors sit in the columns:
            // `mul(basis, v) == q.transformVector(v)`. Everything that stores an orientation as a matrix depends on it,
            // and without this check a transposed cast still passes every orthonormality and round-trip test below.
            {
                for (uint32_t i = 0; i < 3; ++i)
                {
                    vec3_t axis = hlsl::promote<vec3_t>(F(0));
                    axis[i] = F(1);
                    check(hlsl::approx::absRelEqual<vec3_t>(hlsl::mul(basis, axis), q.transformVector(axis, true), F(1e-5), F(1e-5)), type, what);
                }
            }
            {
                const traits_t rt = traits_t::create(basis);
                check(rt.invertible, type, what);
                check(rt.orthogonal, type, what);
                check(rt.orthonormal, type, what);
                const quat_t recovered = quat_t::createFromRotationMatrix(basis, true);
                check(isFinite(recovered), type, what);
                check(hlsl::approx::orientationEqual(q.data, recovered.data, F(1e-5)), type, what);
            }
            // uniformly scaled, dot products between rows grow with the scale too
            mat_t scaled = basis;
            scaled *= F(100);
            {
                const traits_t rt = traits_t::create(scaled);
                check(rt.orthogonal, type, what);
                check(!rt.orthonormal, type, what);
                check(hlsl::approx::absRelEqual(rt.uniformColumnSqNorm, F(1e4), F(0), F(1e-5)), type, what);
                const quat_t recovered = quat_t::createFromRotationMatrix(scaled, true);
                check(isFinite(recovered), type, what);
                check(hlsl::approx::orientationEqual(q.data, recovered.data, F(1e-5)), type, what);
            }
        };

        // axis aligned, basis dot products come out as exact zeros
        testRotation(quat_t::createFromAxisAngle(vec3_t(F(0), F(0), F(1)), hlsl::numbers::pi<F> * F(0.5)), "90deg about Z");
        // generic, basis dot products are only approximately zero
        testRotation(quat_t::createFromAxisAngle(hlsl::normalize(vec3_t(F(1), F(2), F(3))), hlsl::numbers::pi<F> * F(37.0 / 180.0)), "37deg about normalize(1,2,3)");
    };
    testType.template operator()<float32_t>("float32_t");
    testType.template operator()<float64_t>("float64_t");

    return pass;
}

class QuaternionTest final : public application_templates::MonoDeviceApplication, public BuiltinResourcesApplication
{
    using device_base_t = application_templates::MonoDeviceApplication;
    using asset_base_t = BuiltinResourcesApplication;
public:
    QuaternionTest(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
        IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}

    bool onAppInitialized(smart_refctd_ptr<ISystem>&& system) override
    {
        // Remember to call the base class initialization!
        if (!device_base_t::onAppInitialized(smart_refctd_ptr(system)))
            return false;
        if (!asset_base_t::onAppInitialized(std::move(system)))
            return false;

        if (!runtimeTraitsTests(m_logger.get()))
            return false;

        {
            CQuaternionTester::PipelineSetupData pplnSetupData;
            pplnSetupData.device = m_device;
            pplnSetupData.api = m_api;
            pplnSetupData.assetMgr = m_assetMgr;
            pplnSetupData.logger = m_logger;
            pplnSetupData.physicalDevice = m_physicalDevice;
            pplnSetupData.computeFamilyIndex = getComputeQueue()->getFamilyIndex();
            pplnSetupData.shaderKey = nbl::this_example::builtin::build::get_spirv_key<"quaternionTest">(m_device.get());

            CQuaternionTester quaternionTester(8);
            quaternionTester.setupPipeline(pplnSetupData);
            if (!quaternionTester.performTestsAndVerifyResults("QuaternionTestLog.txt"))
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

NBL_MAIN_FUNC(QuaternionTest)
