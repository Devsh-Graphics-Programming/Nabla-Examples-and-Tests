#ifndef _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_TGMATH_TESTER_INCLUDED_
#define _NBL_EXAMPLES_TESTS_22_CPP_COMPAT_C_TGMATH_TESTER_INCLUDED_


#include "nbl/examples/examples.hpp"
#include "app_resources/common.hlsl"
#include "nbl/examples/Tester/ITester.h"

using namespace nbl;

class CTgmathTester final : public ITester<TgmathIntputTestValues, TgmathTestValues, TgmathTestExecutor>
{
    using base_t = ITester<TgmathIntputTestValues, TgmathTestValues, TgmathTestExecutor>;

public:
    CTgmathTester(const uint32_t testBatchCount)
        : base_t(testBatchCount) {};

private:
    TgmathIntputTestValues generateInputTestValues() override
    {
        std::uniform_real_distribution<float> realDistribution(-100.0f, 100.0f);
        std::uniform_real_distribution<float> realDistributionSmall(1.0f, 4.0f);
        std::uniform_int_distribution<int> intDistribution(-100, 100);
        std::uniform_int_distribution<int> coinFlipDistribution(0, 1);

        TgmathIntputTestValues testInput;
        testInput.floor = realDistribution(getRandomEngine());
        testInput.isnan = coinFlipDistribution(getRandomEngine()) ? realDistribution(getRandomEngine()) : std::numeric_limits<float>::quiet_NaN();
        testInput.isinf = coinFlipDistribution(getRandomEngine()) ? realDistribution(getRandomEngine()) : std::numeric_limits<float>::infinity();
        testInput.powX = realDistributionSmall(getRandomEngine());
        testInput.powY = realDistributionSmall(getRandomEngine());
        testInput.exp = realDistributionSmall(getRandomEngine());
        testInput.exp2 = realDistributionSmall(getRandomEngine());
        testInput.log = realDistribution(getRandomEngine());
        testInput.log2 = realDistribution(getRandomEngine());
        testInput.absF = realDistribution(getRandomEngine());
        testInput.absI = intDistribution(getRandomEngine());
        testInput.sqrt = realDistribution(getRandomEngine());
        testInput.sin = realDistribution(getRandomEngine());
        testInput.cos = realDistribution(getRandomEngine());
        testInput.tan = realDistribution(getRandomEngine());
        testInput.asin = realDistribution(getRandomEngine());
        testInput.atan = realDistribution(getRandomEngine());
        testInput.sinh = realDistribution(getRandomEngine());
        testInput.cosh = realDistribution(getRandomEngine());
        testInput.tanh = realDistribution(getRandomEngine());
        testInput.asinh = realDistribution(getRandomEngine());
        testInput.acosh = realDistribution(getRandomEngine());
        testInput.atanh = realDistribution(getRandomEngine());
        testInput.atan2X = realDistribution(getRandomEngine());
        testInput.atan2Y = realDistribution(getRandomEngine());
        testInput.acos = realDistribution(getRandomEngine());
        testInput.modf = realDistribution(getRandomEngine());
        testInput.round = realDistribution(getRandomEngine());
        testInput.roundEven = coinFlipDistribution(getRandomEngine()) ? realDistributionSmall(getRandomEngine()) : (static_cast<float32_t>(intDistribution(getRandomEngine()) / 2) + 0.5f);
        testInput.trunc = realDistribution(getRandomEngine());
        testInput.ceil = realDistribution(getRandomEngine());
        testInput.fmaX = realDistribution(getRandomEngine());
        testInput.fmaY = realDistribution(getRandomEngine());
        testInput.fmaZ = realDistribution(getRandomEngine());
        testInput.ldexpArg = realDistributionSmall(getRandomEngine());
        testInput.ldexpExp = intDistribution(getRandomEngine());
        testInput.erf = realDistribution(getRandomEngine());
        testInput.erfInv = realDistribution(getRandomEngine());

        testInput.floorVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.isnanVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.isinfVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.powXVec = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.powYVec = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.expVec = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.exp2Vec = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.logVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.log2Vec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.absFVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.absIVec = int32_t3(intDistribution(getRandomEngine()), intDistribution(getRandomEngine()), intDistribution(getRandomEngine()));
        testInput.sqrtVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.sinVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.cosVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.tanVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.asinVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.atanVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.sinhVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.coshVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.tanhVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.asinhVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.acoshVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.atanhVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.atan2XVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.atan2YVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.acosVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.modfVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.ldexpArgVec = float32_t3(realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()), realDistributionSmall(getRandomEngine()));
        testInput.ldexpExpVec = float32_t3(intDistribution(getRandomEngine()), intDistribution(getRandomEngine()), intDistribution(getRandomEngine()));
        testInput.erfVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.erfInvVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));

        testInput.modfStruct = realDistribution(getRandomEngine());
        testInput.modfStructVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));
        testInput.frexpStruct = realDistribution(getRandomEngine());
        testInput.frexpStructVec = float32_t3(realDistribution(getRandomEngine()), realDistribution(getRandomEngine()), realDistribution(getRandomEngine()));

        return testInput;
    }

    TgmathTestValues determineExpectedResults(const TgmathIntputTestValues& testInput) override
    {
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

        return expected;
    }

    bool verifyTestResults(const TgmathTestValues& expectedTestValues, const TgmathTestValues& testValues, const size_t testIteration, const uint32_t seed, TestType testType) override
    {
        // TODO: figure out input for functions: sinh, cosh so output isn't a crazy low number
        // very low numbers generate comparison errors

        bool pass = true;
        pass &= verifyTestValue("floor", expectedTestValues.floor, testValues.floor, testIteration, seed, testType);
        pass &= verifyTestValue("isnan", expectedTestValues.isnan, testValues.isnan, testIteration, seed, testType);
        pass &= verifyTestValue("isinf", expectedTestValues.isinf, testValues.isinf, testIteration, seed, testType);
        pass &= verifyTestValue("pow", expectedTestValues.pow, testValues.pow, testIteration, seed, testType, 0.0001);
        pass &= verifyTestValue("exp", expectedTestValues.exp, testValues.exp, testIteration, seed, testType);
        pass &= verifyTestValue("exp2", expectedTestValues.exp2, testValues.exp2, testIteration, seed, testType);
        pass &= verifyTestValue("log", expectedTestValues.log, testValues.log, testIteration, seed, testType);
        pass &= verifyTestValue("log2", expectedTestValues.log2, testValues.log2, testIteration, seed, testType);
        pass &= verifyTestValue("absF", expectedTestValues.absF, testValues.absF, testIteration, seed, testType);
        pass &= verifyTestValue("absI", expectedTestValues.absI, testValues.absI, testIteration, seed, testType);
        pass &= verifyTestValue("sqrt", expectedTestValues.sqrt, testValues.sqrt, testIteration, seed, testType);
        pass &= verifyTestValue("sin", expectedTestValues.sin, testValues.sin, testIteration, seed, testType);
        pass &= verifyTestValue("cos", expectedTestValues.cos, testValues.cos, testIteration, seed, testType);
        pass &= verifyTestValue("acos", expectedTestValues.acos, testValues.acos, testIteration, seed, testType);
        pass &= verifyTestValue("tan", expectedTestValues.tan, testValues.tan, testIteration, seed, testType);
        pass &= verifyTestValue("asin", expectedTestValues.asin, testValues.asin, testIteration, seed, testType);
        pass &= verifyTestValue("atan", expectedTestValues.atan, testValues.atan, testIteration, seed, testType);
        //pass &= verifyTestValue("sinh", expectedTestValues.sinh, testValues.sinh, testIteration, seed, testType);
        //pass &= verifyTestValue("cosh", expectedTestValues.cosh, testValues.cosh, testIteration, seed, testType);
        pass &= verifyTestValue("tanh", expectedTestValues.tanh, testValues.tanh, testIteration, seed, testType);
        pass &= verifyTestValue("asinh", expectedTestValues.asinh, testValues.asinh, testIteration, seed, testType);
        pass &= verifyTestValue("acosh", expectedTestValues.acosh, testValues.acosh, testIteration, seed, testType);
        pass &= verifyTestValue("atanh", expectedTestValues.atanh, testValues.atanh, testIteration, seed, testType);
        pass &= verifyTestValue("atan2", expectedTestValues.atan2, testValues.atan2, testIteration, seed, testType);
        pass &= verifyTestValue("modf", expectedTestValues.modf, testValues.modf, testIteration, seed, testType);
        pass &= verifyTestValue("round", expectedTestValues.round, testValues.round, testIteration, seed, testType);
        pass &= verifyTestValue("roundEven", expectedTestValues.roundEven, testValues.roundEven, testIteration, seed, testType);
        pass &= verifyTestValue("trunc", expectedTestValues.trunc, testValues.trunc, testIteration, seed, testType);
        pass &= verifyTestValue("ceil", expectedTestValues.ceil, testValues.ceil, testIteration, seed, testType);
        pass &= verifyTestValue("fma", expectedTestValues.fma, testValues.fma, testIteration, seed, testType);
        pass &= verifyTestValue("ldexp", expectedTestValues.ldexp, testValues.ldexp, testIteration, seed, testType);
        pass &= verifyTestValue("erf", expectedTestValues.erf, testValues.erf, testIteration, seed, testType);
        //pass &= verifyTestValue("erfInv", expectedTestValues.erfInv, testValues.erfInv, testIteration, seed, testType);

        pass &= verifyTestValue("floorVec", expectedTestValues.floorVec, testValues.floorVec, testIteration, seed, testType);
        pass &= verifyTestValue("isnanVec", expectedTestValues.isnanVec, testValues.isnanVec, testIteration, seed, testType);
        pass &= verifyTestValue("isinfVec", expectedTestValues.isinfVec, testValues.isinfVec, testIteration, seed, testType);
        pass &= verifyTestValue("powVec", expectedTestValues.powVec, testValues.powVec, testIteration, seed, testType, 0.0001);
        pass &= verifyTestValue("expVec", expectedTestValues.expVec, testValues.expVec, testIteration, seed, testType);
        pass &= verifyTestValue("exp2Vec", expectedTestValues.exp2Vec, testValues.exp2Vec, testIteration, seed, testType);
        pass &= verifyTestValue("logVec", expectedTestValues.logVec, testValues.logVec, testIteration, seed, testType);
        pass &= verifyTestValue("log2Vec", expectedTestValues.log2Vec, testValues.log2Vec, testIteration, seed, testType);
        pass &= verifyTestValue("absFVec", expectedTestValues.absFVec, testValues.absFVec, testIteration, seed, testType);
        pass &= verifyTestValue("absIVec", expectedTestValues.absIVec, testValues.absIVec, testIteration, seed, testType);
        pass &= verifyTestValue("sqrtVec", expectedTestValues.sqrtVec, testValues.sqrtVec, testIteration, seed, testType);
        pass &= verifyTestValue("sinVec", expectedTestValues.sinVec, testValues.sinVec, testIteration, seed, testType);
        pass &= verifyTestValue("cosVec", expectedTestValues.cosVec, testValues.cosVec, testIteration, seed, testType);
        pass &= verifyTestValue("acosVec", expectedTestValues.acosVec, testValues.acosVec, testIteration, seed, testType);
        pass &= verifyTestValue("modfVec", expectedTestValues.modfVec, testValues.modfVec, testIteration, seed, testType);
        pass &= verifyTestValue("roundVec", expectedTestValues.roundVec, testValues.roundVec, testIteration, seed, testType);
        pass &= verifyTestValue("roundEvenVec", expectedTestValues.roundEvenVec, testValues.roundEvenVec, testIteration, seed, testType);
        pass &= verifyTestValue("truncVec", expectedTestValues.truncVec, testValues.truncVec, testIteration, seed, testType);
        pass &= verifyTestValue("ceilVec", expectedTestValues.ceilVec, testValues.ceilVec, testIteration, seed, testType);
        pass &= verifyTestValue("fmaVec", expectedTestValues.fmaVec, testValues.fmaVec, testIteration, seed, testType);
        pass &= verifyTestValue("ldexp", expectedTestValues.ldexpVec, testValues.ldexpVec, testIteration, seed, testType);
        pass &= verifyTestValue("tanVec", expectedTestValues.tanVec, testValues.tanVec, testIteration, seed, testType);
        pass &= verifyTestValue("asinVec", expectedTestValues.asinVec, testValues.asinVec, testIteration, seed, testType);
        pass &= verifyTestValue("atanVec", expectedTestValues.atanVec, testValues.atanVec, testIteration, seed, testType);
        //pass &= verifyTestValue("sinhVec", expectedTestValues.sinhVec, testValues.sinhVec, testIteration, seed, testType);
        //pass &= verifyTestValue("coshVec", expectedTestValues.coshVec, testValues.coshVec, testIteration, seed, testType);
        pass &= verifyTestValue("tanhVec", expectedTestValues.tanhVec, testValues.tanhVec, testIteration, seed, testType);
        pass &= verifyTestValue("asinhVec", expectedTestValues.asinhVec, testValues.asinhVec, testIteration, seed, testType);
        pass &= verifyTestValue("acoshVec", expectedTestValues.acoshVec, testValues.acoshVec, testIteration, seed, testType);
        pass &= verifyTestValue("atanhVec", expectedTestValues.atanhVec, testValues.atanhVec, testIteration, seed, testType);
        pass &= verifyTestValue("atan2Vec", expectedTestValues.atan2Vec, testValues.atan2Vec, testIteration, seed, testType);
        pass &= verifyTestValue("erfVec", expectedTestValues.erfVec, testValues.erfVec, testIteration, seed, testType);
        //pass &= verifyTestValue("erfInvVec", expectedTestValues.erfInvVec, testValues.erfInvVec, testIteration, seed, testType);

        // verify output of struct producing functions
        pass &= verifyTestValue("modfStruct", expectedTestValues.modfStruct.fractionalPart, testValues.modfStruct.fractionalPart, testIteration, seed, testType);
        pass &= verifyTestValue("modfStruct", expectedTestValues.modfStruct.wholeNumberPart, testValues.modfStruct.wholeNumberPart, testIteration, seed, testType);
        pass &= verifyTestValue("modfStructVec", expectedTestValues.modfStructVec.fractionalPart, testValues.modfStructVec.fractionalPart, testIteration, seed, testType);
        pass &= verifyTestValue("modfStructVec", expectedTestValues.modfStructVec.wholeNumberPart, testValues.modfStructVec.wholeNumberPart, testIteration, seed, testType);

        pass &= verifyTestValue("frexpStruct", expectedTestValues.frexpStruct.significand, testValues.frexpStruct.significand, testIteration, seed, testType);
        pass &= verifyTestValue("frexpStruct", expectedTestValues.frexpStruct.exponent, testValues.frexpStruct.exponent, testIteration, seed, testType);
        pass &= verifyTestValue("frexpStructVec", expectedTestValues.frexpStructVec.significand, testValues.frexpStructVec.significand, testIteration, seed, testType);
        pass &= verifyTestValue("frexpStructVec", expectedTestValues.frexpStructVec.exponent, testValues.frexpStructVec.exponent, testIteration, seed, testType);
        return pass;
    }
};

#endif