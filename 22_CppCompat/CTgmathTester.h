#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_TGMATH_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_TGMATH_TESTER_INCLUDED_


#include "nbl/examples/examples.hpp"

#include "app_resources/common.hlsl"
#include "ITester.h"

#include "nbl/builtin/hlsl/math/quaternions.hlsl"


using namespace nbl;

class CTgmathTester final : public ITester
{
public:
    void performTests()
    {
        std::random_device rd;
        std::mt19937 mt(rd());

        std::uniform_real_distribution<float> realDistributionNeg(-50.0f, -1.0f);
        std::uniform_real_distribution<float> realDistributionPos(1.0f, 50.0f);
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<int> coinFlipDistribution(0, 1);

        m_logger->log("tgmath.hlsl TESTS:", system::ILogger::ELL_PERFORMANCE);
        for (int i = 0; i < Iterations; ++i)
        {
            // Set input thest values that will be used in both CPU and GPU tests
            TgmathIntputTestValues testInput;
            testInput.floor = realDistribution(mt);
            testInput.isnan = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::quiet_NaN();
            testInput.isinf = coinFlipDistribution(mt) ? realDistribution(mt) : std::numeric_limits<float>::infinity();
            testInput.powX = realDistributionSmall(mt);
            testInput.powY = realDistributionSmall(mt);
            testInput.exp = realDistributionSmall(mt);
            testInput.exp2 = realDistributionSmall(mt);
            testInput.log = realDistribution(mt);
            testInput.log2 = realDistribution(mt);
            testInput.absF = realDistribution(mt);
            testInput.absI = intDistribution(mt);
            testInput.sqrt = realDistribution(mt);
            testInput.sin = realDistribution(mt);
            testInput.cos = realDistribution(mt);
            testInput.tan = realDistribution(mt);
            testInput.asin = realDistribution(mt);
            testInput.atan = realDistribution(mt);
            testInput.sinh = realDistribution(mt);
            testInput.cosh = realDistribution(mt);
            testInput.tanh = realDistribution(mt);
            testInput.asinh = realDistribution(mt);
            testInput.acosh = realDistribution(mt);
            testInput.atanh = realDistribution(mt);
            testInput.atan2X = realDistribution(mt);
            testInput.atan2Y = realDistribution(mt);
            testInput.acos = realDistribution(mt);
            testInput.modf = realDistribution(mt);
            testInput.round = realDistribution(mt);
            testInput.roundEven = coinFlipDistribution(mt) ? realDistributionSmall(mt) : (static_cast<float32_t>(intDistribution(mt) / 2) + 0.5f);
            testInput.trunc = realDistribution(mt);
            testInput.ceil = realDistribution(mt);
            testInput.fmaX = realDistribution(mt);
            testInput.fmaY = realDistribution(mt);
            testInput.fmaZ = realDistribution(mt);
            testInput.ldexpArg = realDistributionSmall(mt);
            testInput.ldexpExp = intDistribution(mt);
            testInput.erf = realDistribution(mt);
            testInput.erfInv = realDistribution(mt);

            testInput.floorVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.isnanVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.isinfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.powXVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.powYVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.expVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.exp2Vec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.logVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.log2Vec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.absFVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.absIVec = int32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
            testInput.sqrtVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.sinVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.cosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.tanVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.asinVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.atanVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.sinhVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.coshVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.tanhVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.asinhVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.acoshVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.atanhVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.atan2XVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.atan2YVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.acosVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.modfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.ldexpArgVec = float32_t3(realDistributionSmall(mt), realDistributionSmall(mt), realDistributionSmall(mt));
            testInput.ldexpExpVec = float32_t3(intDistribution(mt), intDistribution(mt), intDistribution(mt));
            testInput.erfVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.erfInvVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));

            testInput.modfStruct = realDistribution(mt);
            testInput.modfStructVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));
            testInput.frexpStruct = realDistribution(mt);
            testInput.frexpStructVec = float32_t3(realDistribution(mt), realDistribution(mt), realDistribution(mt));

            // use std library functions to determine expected test values, the output of functions from tgmath.hlsl will be verified against these values
            TgmathTestValues expected;
            expected.floor = std::floor(testInput.floor);
            expected.isnan = std::isnan(testInput.isnan);
            expected.isinf = std::isinf(testInput.isinf);
            expected.pow = std::pow(testInput.powX, testInput.powY);
            expected.exp = std::exp(testInput.exp);
            expected.exp2 = std::exp2(testInput.exp2);
            expected.log = std::log(testInput.log);
            expected.log2 = std::log2(testInput.log2);
            expected.absF = std::abs(testInput.absF);
            expected.absI = std::abs(testInput.absI);
            expected.sqrt = std::sqrt(testInput.sqrt);
            expected.sin = std::sin(testInput.sin);
            expected.cos = std::cos(testInput.cos);
            expected.acos = std::acos(testInput.acos);
            expected.tan = std::tan(testInput.tan);
            expected.asin = std::asin(testInput.asin);
            expected.atan = std::atan(testInput.atan);
            expected.sinh = std::sinh(testInput.sinh);
            expected.cosh = std::cosh(testInput.cosh);
            expected.tanh = std::tanh(testInput.tanh);
            expected.asinh = std::asinh(testInput.asinh);
            expected.acosh = std::acosh(testInput.acosh);
            expected.atanh = std::atanh(testInput.atanh);
            expected.atan2 = std::atan2(testInput.atan2Y, testInput.atan2X);
            expected.erf = std::erf(testInput.erf);
            {
                float tmp;
                expected.modf = std::modf(testInput.modf, &tmp);
            }
            expected.round = std::round(testInput.round);
            // TODO: uncomment when C++23
            //expected.roundEven = std::roundeven(testInput.roundEven);
            // TODO: remove when C++23
            auto roundeven = [](const float& val) -> float
                {
                    float tmp;
                    if (std::abs(std::modf(val, &tmp)) == 0.5f)
                    {
                        int32_t result = static_cast<int32_t>(val);
                        if (result % 2 != 0)
                            result >= 0 ? ++result : --result;
                        return result;
                    }

                    return std::round(val);
                };
            expected.roundEven = roundeven(testInput.roundEven);

            expected.trunc = std::trunc(testInput.trunc);
            expected.ceil = std::ceil(testInput.ceil);
            expected.fma = std::fma(testInput.fmaX, testInput.fmaY, testInput.fmaZ);
            expected.ldexp = std::ldexp(testInput.ldexpArg, testInput.ldexpExp);

            expected.floorVec = float32_t3(std::floor(testInput.floorVec.x), std::floor(testInput.floorVec.y), std::floor(testInput.floorVec.z));

            expected.isnanVec = float32_t3(std::isnan(testInput.isnanVec.x), std::isnan(testInput.isnanVec.y), std::isnan(testInput.isnanVec.z));
            expected.isinfVec = float32_t3(std::isinf(testInput.isinfVec.x), std::isinf(testInput.isinfVec.y), std::isinf(testInput.isinfVec.z));

            expected.powVec.x = std::pow(testInput.powXVec.x, testInput.powYVec.x);
            expected.powVec.y = std::pow(testInput.powXVec.y, testInput.powYVec.y);
            expected.powVec.z = std::pow(testInput.powXVec.z, testInput.powYVec.z);

            expected.expVec = float32_t3(std::exp(testInput.expVec.x), std::exp(testInput.expVec.y), std::exp(testInput.expVec.z));
            expected.exp2Vec = float32_t3(std::exp2(testInput.exp2Vec.x), std::exp2(testInput.exp2Vec.y), std::exp2(testInput.exp2Vec.z));
            expected.logVec = float32_t3(std::log(testInput.logVec.x), std::log(testInput.logVec.y), std::log(testInput.logVec.z));
            expected.log2Vec = float32_t3(std::log2(testInput.log2Vec.x), std::log2(testInput.log2Vec.y), std::log2(testInput.log2Vec.z));
            expected.absFVec = float32_t3(std::abs(testInput.absFVec.x), std::abs(testInput.absFVec.y), std::abs(testInput.absFVec.z));
            expected.absIVec = float32_t3(std::abs(testInput.absIVec.x), std::abs(testInput.absIVec.y), std::abs(testInput.absIVec.z));
            expected.sqrtVec = float32_t3(std::sqrt(testInput.sqrtVec.x), std::sqrt(testInput.sqrtVec.y), std::sqrt(testInput.sqrtVec.z));
            expected.cosVec = float32_t3(std::cos(testInput.cosVec.x), std::cos(testInput.cosVec.y), std::cos(testInput.cosVec.z));
            expected.sinVec = float32_t3(std::sin(testInput.sinVec.x), std::sin(testInput.sinVec.y), std::sin(testInput.sinVec.z));
            expected.tanVec = float32_t3(std::tan(testInput.tanVec.x), std::tan(testInput.tanVec.y), std::tan(testInput.tanVec.z));
            expected.asinVec = float32_t3(std::asin(testInput.asinVec.x), std::asin(testInput.asinVec.y), std::asin(testInput.asinVec.z));
            expected.atanVec = float32_t3(std::atan(testInput.atanVec.x), std::atan(testInput.atanVec.y), std::atan(testInput.atanVec.z));
            expected.sinhVec = float32_t3(std::sinh(testInput.sinhVec.x), std::sinh(testInput.sinhVec.y), std::sinh(testInput.sinhVec.z));
            expected.coshVec = float32_t3(std::cosh(testInput.coshVec.x), std::cosh(testInput.coshVec.y), std::cosh(testInput.coshVec.z));
            expected.tanhVec = float32_t3(std::tanh(testInput.tanhVec.x), std::tanh(testInput.tanhVec.y), std::tanh(testInput.tanhVec.z));
            expected.asinhVec = float32_t3(std::asinh(testInput.asinhVec.x), std::asinh(testInput.asinhVec.y), std::asinh(testInput.asinhVec.z));
            expected.acoshVec = float32_t3(std::acosh(testInput.acoshVec.x), std::acosh(testInput.acoshVec.y), std::acosh(testInput.acoshVec.z));
            expected.atanhVec = float32_t3(std::atanh(testInput.atanhVec.x), std::atanh(testInput.atanhVec.y), std::atanh(testInput.atanhVec.z));
            expected.atan2Vec = float32_t3(std::atan2(testInput.atan2YVec.x, testInput.atan2XVec.x), std::atan2(testInput.atan2YVec.y, testInput.atan2XVec.y), std::atan2(testInput.atan2YVec.z, testInput.atan2XVec.z));
            expected.acosVec = float32_t3(std::acos(testInput.acosVec.x), std::acos(testInput.acosVec.y), std::acos(testInput.acosVec.z));
            expected.erfVec = float32_t3(std::erf(testInput.erfVec.x), std::erf(testInput.erfVec.y), std::erf(testInput.erfVec.z));
            {
                float tmp;
                expected.modfVec = float32_t3(std::modf(testInput.modfVec.x, &tmp), std::modf(testInput.modfVec.y, &tmp), std::modf(testInput.modfVec.z, &tmp));
            }
            expected.roundVec = float32_t3(
                std::round(testInput.roundVec.x),
                std::round(testInput.roundVec.y),
                std::round(testInput.roundVec.z)
            );
            // TODO: uncomment when C++23
            //expected.roundEven = float32_t(
            //    std::roundeven(testInput.roundEvenVec.x),
            //    std::roundeven(testInput.roundEvenVec.y),
            //    std::roundeven(testInput.roundEvenVec.z)
            //    );
            // TODO: remove when C++23
            expected.roundEvenVec = float32_t3(
                roundeven(testInput.roundEvenVec.x),
                roundeven(testInput.roundEvenVec.y),
                roundeven(testInput.roundEvenVec.z)
            );

            expected.truncVec = float32_t3(std::trunc(testInput.truncVec.x), std::trunc(testInput.truncVec.y), std::trunc(testInput.truncVec.z));
            expected.ceilVec = float32_t3(std::ceil(testInput.ceilVec.x), std::ceil(testInput.ceilVec.y), std::ceil(testInput.ceilVec.z));
            expected.fmaVec = float32_t3(
                std::fma(testInput.fmaXVec.x, testInput.fmaYVec.x, testInput.fmaZVec.x),
                std::fma(testInput.fmaXVec.y, testInput.fmaYVec.y, testInput.fmaZVec.y),
                std::fma(testInput.fmaXVec.z, testInput.fmaYVec.z, testInput.fmaZVec.z)
            );
            expected.ldexpVec = float32_t3(
                std::ldexp(testInput.ldexpArgVec.x, testInput.ldexpExpVec.x),
                std::ldexp(testInput.ldexpArgVec.y, testInput.ldexpExpVec.y),
                std::ldexp(testInput.ldexpArgVec.z, testInput.ldexpExpVec.z)
            );

            {
                ModfOutput<float> expectedModfStructOutput;
                expectedModfStructOutput.fractionalPart = std::modf(testInput.modfStruct, &expectedModfStructOutput.wholeNumberPart);
                expected.modfStruct = expectedModfStructOutput;

                ModfOutput<float32_t3> expectedModfStructOutputVec;
                for (int i = 0; i < 3; ++i)
                    expectedModfStructOutputVec.fractionalPart[i] = std::modf(testInput.modfStructVec[i], &expectedModfStructOutputVec.wholeNumberPart[i]);
                expected.modfStructVec = expectedModfStructOutputVec;
            }

            {
                FrexpOutput<float> expectedFrexpStructOutput;
                expectedFrexpStructOutput.significand = std::frexp(testInput.frexpStruct, &expectedFrexpStructOutput.exponent);
                expected.frexpStruct = expectedFrexpStructOutput;

                FrexpOutput<float32_t3> expectedFrexpStructOutputVec;
                for (int i = 0; i < 3; ++i)
                    expectedFrexpStructOutputVec.significand[i] = std::frexp(testInput.frexpStructVec[i], &expectedFrexpStructOutputVec.exponent[i]);
                expected.frexpStructVec = expectedFrexpStructOutputVec;
            }

            performCpuTests(testInput, expected);
            performGpuTests(testInput, expected);
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
        // TODO: figure out input for functions: sinh, cosh so output isn't a crazy low number
        // very low numbers generate comparison errors

        verifyTestValue("floor", expectedTestValues.floor, testValues.floor, testType);
        verifyTestValue("isnan", expectedTestValues.isnan, testValues.isnan, testType);
        verifyTestValue("isinf", expectedTestValues.isinf, testValues.isinf, testType);
        verifyTestValue("pow", expectedTestValues.pow, testValues.pow, testType);
        verifyTestValue("exp", expectedTestValues.exp, testValues.exp, testType);
        verifyTestValue("exp2", expectedTestValues.exp2, testValues.exp2, testType);
        verifyTestValue("log", expectedTestValues.log, testValues.log, testType);
        verifyTestValue("log2", expectedTestValues.log2, testValues.log2, testType);
        verifyTestValue("absF", expectedTestValues.absF, testValues.absF, testType);
        verifyTestValue("absI", expectedTestValues.absI, testValues.absI, testType);
        verifyTestValue("sqrt", expectedTestValues.sqrt, testValues.sqrt, testType);
        verifyTestValue("sin", expectedTestValues.sin, testValues.sin, testType);
        verifyTestValue("cos", expectedTestValues.cos, testValues.cos, testType);
        verifyTestValue("acos", expectedTestValues.acos, testValues.acos, testType);
        verifyTestValue("tan", expectedTestValues.tan, testValues.tan, testType);
        verifyTestValue("asin", expectedTestValues.asin, testValues.asin, testType);
        verifyTestValue("atan", expectedTestValues.atan, testValues.atan, testType);
        //verifyTestValue("sinh", expectedTestValues.sinh, testValues.sinh, testType);
        //verifyTestValue("cosh", expectedTestValues.cosh, testValues.cosh, testType);
        verifyTestValue("tanh", expectedTestValues.tanh, testValues.tanh, testType);
        verifyTestValue("asinh", expectedTestValues.asinh, testValues.asinh, testType);
        verifyTestValue("acosh", expectedTestValues.acosh, testValues.acosh, testType);
        verifyTestValue("atanh", expectedTestValues.atanh, testValues.atanh, testType);
        verifyTestValue("atan2", expectedTestValues.atan2, testValues.atan2, testType);
        verifyTestValue("modf", expectedTestValues.modf, testValues.modf, testType);
        verifyTestValue("round", expectedTestValues.round, testValues.round, testType);
        verifyTestValue("roundEven", expectedTestValues.roundEven, testValues.roundEven, testType);
        verifyTestValue("trunc", expectedTestValues.trunc, testValues.trunc, testType);
        verifyTestValue("ceil", expectedTestValues.ceil, testValues.ceil, testType);
        verifyTestValue("fma", expectedTestValues.fma, testValues.fma, testType);
        verifyTestValue("ldexp", expectedTestValues.ldexp, testValues.ldexp, testType);
        verifyTestValue("erf", expectedTestValues.erf, testValues.erf, testType);
        //verifyTestValue("erfInv", expectedTestValues.erfInv, testValues.erfInv, testType);

        verifyTestVector3dValue("floorVec", expectedTestValues.floorVec, testValues.floorVec, testType);
        verifyTestVector3dValue("isnanVec", expectedTestValues.isnanVec, testValues.isnanVec, testType);
        verifyTestVector3dValue("isinfVec", expectedTestValues.isinfVec, testValues.isinfVec, testType);
        verifyTestVector3dValue("powVec", expectedTestValues.powVec, testValues.powVec, testType);
        verifyTestVector3dValue("expVec", expectedTestValues.expVec, testValues.expVec, testType);
        verifyTestVector3dValue("exp2Vec", expectedTestValues.exp2Vec, testValues.exp2Vec, testType);
        verifyTestVector3dValue("logVec", expectedTestValues.logVec, testValues.logVec, testType);
        verifyTestVector3dValue("log2Vec", expectedTestValues.log2Vec, testValues.log2Vec, testType);
        verifyTestVector3dValue("absFVec", expectedTestValues.absFVec, testValues.absFVec, testType);
        verifyTestVector3dValue("absIVec", expectedTestValues.absIVec, testValues.absIVec, testType);
        verifyTestVector3dValue("sqrtVec", expectedTestValues.sqrtVec, testValues.sqrtVec, testType);
        verifyTestVector3dValue("sinVec", expectedTestValues.sinVec, testValues.sinVec, testType);
        verifyTestVector3dValue("cosVec", expectedTestValues.cosVec, testValues.cosVec, testType);
        verifyTestVector3dValue("acosVec", expectedTestValues.acosVec, testValues.acosVec, testType);
        verifyTestVector3dValue("modfVec", expectedTestValues.modfVec, testValues.modfVec, testType);
        verifyTestVector3dValue("roundVec", expectedTestValues.roundVec, testValues.roundVec, testType);
        verifyTestVector3dValue("roundEvenVec", expectedTestValues.roundEvenVec, testValues.roundEvenVec, testType);
        verifyTestVector3dValue("truncVec", expectedTestValues.truncVec, testValues.truncVec, testType);
        verifyTestVector3dValue("ceilVec", expectedTestValues.ceilVec, testValues.ceilVec, testType);
        verifyTestVector3dValue("fmaVec", expectedTestValues.fmaVec, testValues.fmaVec, testType);
        verifyTestVector3dValue("ldexp", expectedTestValues.ldexpVec, testValues.ldexpVec, testType);
        verifyTestVector3dValue("tanVec", expectedTestValues.tanVec, testValues.tanVec, testType);
        verifyTestVector3dValue("asinVec", expectedTestValues.asinVec, testValues.asinVec, testType);
        verifyTestVector3dValue("atanVec", expectedTestValues.atanVec, testValues.atanVec, testType);
        //verifyTestVector3dValue("sinhVec", expectedTestValues.sinhVec, testValues.sinhVec, testType);
        //verifyTestVector3dValue("coshVec", expectedTestValues.coshVec, testValues.coshVec, testType);
        verifyTestVector3dValue("tanhVec", expectedTestValues.tanhVec, testValues.tanhVec, testType);
        verifyTestVector3dValue("asinhVec", expectedTestValues.asinhVec, testValues.asinhVec, testType);
        verifyTestVector3dValue("acoshVec", expectedTestValues.acoshVec, testValues.acoshVec, testType);
        verifyTestVector3dValue("atanhVec", expectedTestValues.atanhVec, testValues.atanhVec, testType);
        verifyTestVector3dValue("atan2Vec", expectedTestValues.atan2Vec, testValues.atan2Vec, testType);
        verifyTestVector3dValue("erfVec", expectedTestValues.erfVec, testValues.erfVec, testType);
        //verifyTestVector3dValue("erfInvVec", expectedTestValues.erfInvVec, testValues.erfInvVec, testType);

        // verify output of struct producing functions
        verifyTestValue("modfStruct", expectedTestValues.modfStruct.fractionalPart, testValues.modfStruct.fractionalPart, testType);
        verifyTestValue("modfStruct", expectedTestValues.modfStruct.wholeNumberPart, testValues.modfStruct.wholeNumberPart, testType);
        verifyTestVector3dValue("modfStructVec", expectedTestValues.modfStructVec.fractionalPart, testValues.modfStructVec.fractionalPart, testType);
        verifyTestVector3dValue("modfStructVec", expectedTestValues.modfStructVec.wholeNumberPart, testValues.modfStructVec.wholeNumberPart, testType);

        verifyTestValue("frexpStruct", expectedTestValues.frexpStruct.significand, testValues.frexpStruct.significand, testType);
        verifyTestValue("frexpStruct", expectedTestValues.frexpStruct.exponent, testValues.frexpStruct.exponent, testType);
        verifyTestVector3dValue("frexpStructVec", expectedTestValues.frexpStructVec.significand, testValues.frexpStructVec.significand, testType);
        verifyTestVector3dValue("frexpStructVec", expectedTestValues.frexpStructVec.exponent, testValues.frexpStructVec.exponent, testType);


        {
            float32_t angle = 0.5;
            float32_t2 dir = float32_t2{ cos(angle), sin(angle) };
            float32_t3x3 rotateMat =
            {
                dir.x, -dir.y, 0.0,
                dir.y, dir.x,  0.0,
                0.0, 0.0, 1.0
            };

            float32_t scale = 100.0;
            float32_t3x3 scaleMat =
            {
                scale, 0.0, 0.0,
                0.0, scale, 0.0,
                0.0, 0.0, scale
            };

            float32_t3x3 expectedTransform = nbl::hlsl::mul(rotateMat, scaleMat);

            math::quaternion<float> quat = math::quaternion<float>::create(expectedTransform);
            float32_t3x3 testTransform = quat.constructMatrix();

            verifyTestMatrix3x3Value("quaternion create from matrix", expectedTransform, testTransform, testType);
        }
    }
};

#endif