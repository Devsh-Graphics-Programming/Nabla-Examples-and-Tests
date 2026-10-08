// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

#include "app_resources/common.hlsl"

#include "CQuaternionTester.h"

#include "nbl/builtin/hlsl/math/quaternions.hlsl"
#include "nbl/builtin/hlsl/approx/abs_rel.hlsl"
#include "nbl/builtin/hlsl/approx/orientation.hlsl"
#include "nbl/ext/Cameras/CCameraMathUtilities.hpp"

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

// Deterministic checks of the Euler angle builder and of the camera extension's math utilities built on it.
bool eulerAndCameraMathTests(system::ILogger* logger)
{
    bool pass = true;
    auto check = [&](const bool condition, const char* type, const char* what)
    {
        if (!condition)
        {
            logger->log("Euler and camera math test failed [%s]: %s", system::ILogger::ELL_ERROR, type, what);
            pass = false;
        }
    };

    auto testType = [&]<typename F>(const char* type)
    {
        using quat_t = hlsl::math::quaternion<F>;
        using vec3_t = hlsl::vector<F, 3>;
        using vec4_t = hlsl::vector<F, 4>;
        using mat4_t = hlsl::matrix<F, 4, 4>;
        using math_utils_t = ext::cameras::CCameraMathUtilities;

        const F tolerance = F(1e-5);
        const F degToRad = hlsl::numbers::pi<F> / F(180);
        const vec3_t right = vec3_t(F(1), F(0), F(0));
        const vec3_t up = vec3_t(F(0), F(1), F(0));
        const vec3_t forward = vec3_t(F(0), F(0), F(1));

        auto vecEqual = [&](const vec3_t lhs, const vec3_t rhs)
        {
            return hlsl::approx::absRelEqual<vec3_t>(lhs, rhs, tolerance, tolerance);
        };
        // `q` and `-q` are the same rotation, so either sign matches
        auto quatEqual = [&](const quat_t lhs, const quat_t rhs)
        {
            return hlsl::approx::absRelEqual<vec4_t>(lhs.data, rhs.data, tolerance, tolerance) ||
                hlsl::approx::absRelEqual<vec4_t>(lhs.data, -rhs.data, tolerance, tolerance);
        };
        auto axisAngle = [](const vec3_t axis, const F angle) { return quat_t::createFromAxisAngle(axis, angle); };

        // (pitch, yaw, roll) in degrees; pitch stays inside (-90, 90) so all three angles can be recovered
        const vec3_t anglesDeg[] = {
            vec3_t(F(0), F(0), F(0)),
            vec3_t(F(30), F(0), F(0)),
            vec3_t(F(0), F(40), F(0)),
            vec3_t(F(0), F(0), F(50)),
            vec3_t(F(20), F(35), F(15)),
            vec3_t(F(-75), F(170), F(-120)),
            vec3_t(F(80), F(-150), F(160)),
            vec3_t(F(-45), F(90), F(90))
        };
        for (const auto& angles : anglesDeg)
        {
            const vec3_t rad = angles * degToRad;
            const quat_t q = quat_t::createFromYawPitchRoll(rad.y, rad.x, rad.z);
            // yaw * pitch * roll: applied to a vector, roll acts first, then pitch, then yaw
            check(quatEqual(q, axisAngle(up, rad.y) * axisAngle(right, rad.x) * axisAngle(forward, rad.z)), type, "yaw pitch roll builder composes yaw * pitch * roll");
            check(vecEqual(math_utils_t::getPitchYawRollRadians(q), rad), type, "pitch yaw roll extractor inverts the yaw pitch roll builder");
        }
        {
            // yaw 90 and roll 90: the camera turns to face +X and its horizon tilts, roll stays a roll
            const quat_t q = quat_t::createFromYawPitchRoll(F(90) * degToRad, F(0), F(90) * degToRad);
            check(vecEqual(q.transformVector(forward, true), right), type, "yaw 90, roll 90 faces +X");
            check(vecEqual(q.transformVector(up, true), forward), type, "yaw 90, roll 90 has up along +Z");
        }
        {
            const vec3_t position = vec3_t(F(1), F(2), F(3));
            const vec3_t target = vec3_t(F(-2), F(0.5), F(7));
            quat_t q;
            check(math_utils_t::tryCreateQuaternionFromLookAt(position, target, up, q), type, "look-at succeeds");
            check(vecEqual(q.transformVector(forward, true), hlsl::normalize(target - position)), type, "look-at faces the target");
            check(hlsl::abs(q.transformVector(right, true).y) <= tolerance, type, "look-at keeps the horizon level");
            check(q.transformVector(up, true).y > F(0), type, "look-at keeps up above the horizon");

            // straight down with up = +Y: the up hint is parallel to forward, so the fallback axis decides the roll
            quat_t down;
            check(math_utils_t::tryCreateQuaternionFromLookAt(vec3_t(F(0), F(5), F(0)), vec3_t(F(0), F(0), F(0)), up, down), type, "look-at straight down succeeds");
            check(vecEqual(down.transformVector(forward, true), -up), type, "look-at straight down faces -Y");
            check(vecEqual(down.transformVector(right, true), right), type, "look-at straight down keeps right along +X");
            check(vecEqual(down.transformVector(up, true), forward), type, "look-at straight down has up along +Z");
        }
        {
            const vec3_t translation = vec3_t(F(4), F(-5), F(6));
            const quat_t rotation = axisAngle(hlsl::normalize(vec3_t(F(1), F(2), F(3))), F(37) * degToRad);
            const mat4_t transform = math_utils_t::composeTransformMatrix(translation, rotation, vec3_t(F(2), F(0.5), F(3)));
            vec3_t outTranslation;
            quat_t outRotation;
            check(math_utils_t::tryExtractPositionAndQuaternionFromTransform(transform, outTranslation, outRotation), type, "extraction of a scaled rigid transform succeeds");
            check(vecEqual(outTranslation, translation), type, "extraction returns the translation");
            check(quatEqual(outRotation, rotation), type, "extraction returns the rotation");

            // the up column gains a component along the right column, so the basis is not orthogonal
            mat4_t sheared = transform;
            sheared[0].y += F(0.5);
            sheared[1].y += F(0.5);
            check(!math_utils_t::tryExtractPositionAndQuaternionFromTransform(sheared, outTranslation, outRotation), type, "extraction rejects a sheared transform");

            // a negative scale on one axis mirrors the basis, which no quaternion can hold
            const mat4_t mirrored = math_utils_t::composeTransformMatrix(translation, rotation, vec3_t(F(-1), F(1), F(1)));
            check(!math_utils_t::tryExtractPositionAndQuaternionFromTransform(mirrored, outTranslation, outRotation), type, "extraction rejects a mirrored transform");
        }
        {
            using mat3_t = hlsl::matrix<F, 3, 3>;
            using traits_t = hlsl::math::linalg::RuntimeTraits<mat3_t>;
            auto scalarEqual = [&](const F lhs, const F rhs) { return hlsl::approx::absRelEqual<F>(lhs, rhs, tolerance, tolerance); };

            const mat3_t rotation = hlsl::_static_cast<mat3_t>(axisAngle(hlsl::normalize(vec3_t(F(1), F(2), F(3))), F(37) * degToRad));
            check(scalarEqual(traits_t::create(rotation).determinant, F(1)), type, "a rotation has determinant 1");
            mat3_t scaled = rotation;
            scaled *= F(2);
            check(scalarEqual(traits_t::create(scaled).determinant, F(8)), type, "a rotation scaled by 2 has determinant 8");

            // orthonormal, but a mirror: the determinant is what tells it apart from a rotation
            const mat3_t mirror = mat3_t(vec3_t(F(-1), F(0), F(0)), vec3_t(F(0), F(1), F(0)), vec3_t(F(0), F(0), F(1)));
            const traits_t mirrorTraits = traits_t::create(mirror);
            check(mirrorTraits.orthonormal, type, "a mirror is orthonormal");
            check(scalarEqual(mirrorTraits.determinant, F(-1)), type, "a mirror has determinant -1");
            check(!math_utils_t::isFiniteQuaternion(quat_t::createFromRotationMatrix(mirror, true)), type, "a mirror does not convert to a quaternion");
            mat3_t negativeScale = rotation;
            negativeScale *= F(-2);
            check(!math_utils_t::isFiniteQuaternion(quat_t::createFromRotationMatrix(negativeScale, true)), type, "a negative uniform scale does not convert to a quaternion");
            check(math_utils_t::isFiniteQuaternion(quat_t::createFromRotationMatrix(scaled, true)), type, "a positive uniform scale still converts to a quaternion");
        }
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
        if (!eulerAndCameraMathTests(m_logger.get()))
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
