#ifndef _CAD_EXAMPLE_CURVES_H_
#define _CAD_EXAMPLE_CURVES_H_

#include <nabla.h>
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
using namespace nbl::hlsl;

#include "common.hlsl"
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>

namespace curves
{
//TODO: move this to cpp-compat hlsl builtins
float64_t2 LineLineIntersection(const float64_t2& p1, const float64_t2& v1, const float64_t2& p2, const float64_t2& v2)
{
    float64_t denominator = v1.y * v2.x - v1.x * v2.y;
    float64_t2 diff = p1 - p2;
    float64_t numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-15 && abs(numerator) < 1e-15)
    {
        // are parallel and the same
        return (p1 + p2) / 2.0;
    }

    float64_t t = numerator / denominator;
    float64_t2 intersectionPoint = p1 + t * v1;
    return intersectionPoint;
}

//TODO: Move these bezier functions inside the bezier struct in hlsl
inline float64_t bezierYatT(const QuadraticBezierInfo& bezier, const float64_t t)
{
    const float64_t a = bezier.p[0].y - 2.0 * bezier.p[1].y + bezier.p[2].y;
    const float64_t b = 2.0 * (bezier.p[1].y - bezier.p[0].y);
    const float64_t c = bezier.p[0].y;
    return ((a * t) + b) * t + c; // computePosition at t1
}

// TODO: move this to cpp-compat hlsl builtins in math::equations::quadratics probably
// solve ax^2+bx+c=0
inline float64_t2 solveQuadraticRoot(const float64_t a, const float64_t b, const float64_t c)
{
    float64_t2 ret;

    const float64_t det = b * b - 4.0 * a * c;
    const float64_t detSqrt = sqrt(det);
    const float64_t rcp = 0.5 / a;
    const float64_t bOver2A = b * rcp;

    float64_t t0 = 0.0, t1 = 0.0;
    if (b >= 0)
    {
        ret[0] = -detSqrt * rcp - bOver2A;
        ret[1] = 2 * c / (-b - detSqrt);
    }
    else
    {
        ret[0] = 2 * c / (-b + detSqrt);
        ret[1] = +detSqrt * rcp - bOver2A;
    }
    
    return ret;
}

// returns nan if found X is outside of bounds or not found at all
inline float64_t bezierYatX(const QuadraticBezierInfo& bezier, float64_t x)
{
    const float64_t a = bezier.p[0].x - 2.0 * bezier.p[1].x + bezier.p[2].x;
    const float64_t b = 2.0 * (bezier.p[1].x - bezier.p[0].x);
    const float64_t c = bezier.p[0].x - x;

    float64_t2 roots = solveQuadraticRoot(a, b, c);

    // _NBL_DEBUG_BREAK_IF(!isnan(roots[0]) && !isnan(roots[1])); // should only have 1 solution

    if (roots[0] >= 0.0 && roots[0] <= 1.0)
        return bezierYatT(bezier, roots[0]);
    else if (roots[1] >= 0.0 && roots[1] <= 1.0)
        return bezierYatT(bezier, roots[1]);
    else
        return std::numeric_limits<double>::quiet_NaN();

}

inline QuadraticBezierInfo constructBezierWithTwoPointsAndTangents(float64_t2 P0, float64_t2 v0, float64_t2 P2, float64_t2 v2)
{
    QuadraticBezierInfo out = {};
    out.p[0] = P0;
    out.p[2] = P2;
    out.p[1] = LineLineIntersection(P0, v0, P2, v2);
    return out;
}

struct ParametricCurve
{
    //! compute position at t
    virtual float64_t2 computePosition(float64_t t) const = 0;

    //! compute unnormalized tangent vector at t
    virtual float64_t2 computeTangent(float64_t t) const = 0;

    //! compute differential arc length at t
    virtual float64_t differentialArcLen(float64_t t) const = 0;

    struct ArcLenIntegrand
    {
        const ParametricCurve* m_curve;

        ArcLenIntegrand(const ParametricCurve* curve)
            : m_curve(curve)
        {}

        inline float64_t operator()(const float64_t t) const
        {
            return m_curve->differentialArcLen(t);
        }
    };

    //! compute arc length by gauss legendere integration
    inline float64_t arcLen(float64_t t0, float64_t t1) const
    {
        constexpr uint16_t IntegrationOrder = 10u;
        return nbl::hlsl::math::quadrature::GaussLegendreIntegration<IntegrationOrder, double, ArcLenIntegrand>::calculateIntegral(ArcLenIntegrand(this), t0, t1);
    }

    //! compute inverse arc len using bisection search
    inline float64_t inverseArcLen_BisectionSearch(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold = 1e-4, const uint16_t iterationThreshold = 16u) const
    {
        float64_t xi = 0.0;
        float64_t low = min;
        float64_t high = max;
        for (uint16_t i = 0; i < iterationThreshold; ++i)
        {
            xi = (low + high) / 2.0;
            float64_t sum = arcLen(min, xi);
            float64_t integral = sum + arcLen(xi, max);

            // we could've done sum/integral - targetLen, but this is more robust as it avoids a divsion
            float64_t valueAtParamGuess = sum - targetLen * integral;

            if (abs(valueAtParamGuess) < cdfAccuracyThreshold * integral)
                return xi; // we found xi value that gives us a cdf of targetLen within cdfAccuracyThreshold
            else
            {
                if (valueAtParamGuess > 0.0)
                    high = xi;
                else
                    low = xi;
            }
        }

        return xi;
    }

    //! compute inverse arc len  
    inline float64_t inverseArcLen(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold = 1e-4) const
    {
        return inverseArcLen_BisectionSearch(targetLen, min, max, cdfAccuracyThreshold);
    }

    //! used in special cases when parametric curves need to find inflection point by using solvnig for root of signed curvature
    virtual float64_t2 computeSecondOrderDifferential(float64_t t) const
    {
        return float64_t2(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
    }

    // gets you the t point of inflection with errorThreshold accuracy
    // the curves we deal with have at most 1 inflection point
    // if there is no inflection point this function will return NaN
    virtual float64_t computeInflectionPoint(float64_t errorThreshold) const
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
};

// It's when t = x in a Parametric Curve
struct ExplicitCurve : public ParametricCurve
{
    virtual float64_t y(float64_t x) const = 0;
    virtual float64_t derivative(float64_t x) const = 0;

    float64_t differentialArcLen(float64_t x) const override
    {
        float64_t deriv = derivative(x);
        return sqrt(1.0 + deriv * deriv);
    }

    float64_t2 computeTangent(float64_t x) const override
    {
        const float64_t deriv = derivative(x);
        float64_t2 v = float64_t2(1.0, deriv);
        if (isinf(deriv))
            v = float64_t2(0.0, 1.0);
        return v;
    }

    float64_t2 computePosition(float64_t x) const override { return float64_t2(x, y(x)); }
};

struct Parabola final : public ExplicitCurve
{
    float64_t a, b, c;

    Parabola(float64_t a, float64_t b, float64_t c)
        : a(a), b(b), c(c)
    {}

    static Parabola fromThreePoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2)
    {
        glm::dmat3 X = glm::dmat3(
            glm::dvec3(P0.x * P0.x, P0.x, 1.0),
            glm::dvec3(P1.x * P1.x, P1.x, 1.0),
            glm::dvec3(P2.x * P2.x, P2.x, 1.0)
        );
        glm::dvec3 M = inverse(transpose(X)) * glm::dvec3(P0.y, P1.y, P2.y);
        return Parabola(M[0], M[1], M[2]);
    }

    float64_t y(float64_t x) const override
    {
        return ((a * x) + b) * x + c;
    }

    float64_t derivative(float64_t x) const override
    {
        return 2.0 * a * x + b;
    }
};

struct CubicCurve final : public ParametricCurve
{
    float64_t4 X;
    float64_t4 Y;

    CubicCurve(const float64_t4& X, const float64_t4& Y) 
        : X(X), Y(Y)
    {}

    float64_t2 computePosition(float64_t t) const override
    {
        return float64_t2(
            ((X[0] * t + X[1]) * t + X[2]) * t + X[3],
            ((Y[0] * t + Y[1]) * t + Y[2]) * t + Y[3]
            );
    }

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override
    {
        return float64_t2(
            (3.0 * X[0] * t + 2.0 * X[1]) * t + X[2],
            (3.0 * Y[0] * t + 2.0 * Y[1]) * t + Y[2]
        );
    }

    //! compute differential arc length at t
    float64_t differentialArcLen(float64_t t) const override
    {
        float64_t2 tangent = computeTangent(t);
        return length(tangent);
    }

    float64_t computeInflectionPoint(float64_t errorThreshold) const override
    {
        // TODO
        return 0.5;
    }
};

// specialized circular arc for the purpose of mixing it with another curve of the same type later
// (r*cos(t*sweep+start), r*sin(t*sweep+start) + originY)
struct CircularArc final : public ParametricCurve
{
    float64_t r;
    float64_t originY; // originX is 0
    float64_t startAngle;
    float64_t sweepAngle;

    CircularArc(float64_t r, float64_t originY, float64_t startAngle, float64_t sweepAngle)
        : r(r), originY(originY), startAngle(startAngle), sweepAngle(sweepAngle)
    {}

    // from circle center (0, -v.y) to start pos (v.x, 0)
    CircularArc(float64_t2 v, float64_t sweepAngle)
        : originY(-v.y), sweepAngle(sweepAngle)
    {
        r = length(v);
        startAngle = getSign(v.y) * acos(v.x / r);
    }

    // from circle center (0, -v.y) to start pos (v.x, 0)
    CircularArc(float64_t2 v)
        : originY(-v.y)
    {
        r = length(v);
        startAngle = getSign(v.y) * acos(v.x / r);
        sweepAngle = -2.0 * getSign(v.y) * acos(abs(originY)/r);
    }

    float64_t2 computePosition(float64_t t) const override
    {
        const float64_t actualT = t * sweepAngle + startAngle;
        return float64_t2(
            r * cos(actualT),
            r * sin(actualT) + originY
        );
    }

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override
    {
        const float64_t actualT = t * sweepAngle + startAngle;
        return float64_t2(
            -1.0 * r * sweepAngle * sin(actualT),
            +1.0 * r * sweepAngle * cos(actualT)
        );
    }

    float64_t2 computeSecondOrderDifferential(float64_t t) const override
    {
        const float64_t actualT = t * sweepAngle + startAngle;
        return float64_t2(
            -1.0 * r * sweepAngle * sweepAngle * cos(actualT),
            -1.0 * r * sweepAngle * sweepAngle * sin(actualT)
        );
    }

    //! compute differential arc length at t
    float64_t differentialArcLen(float64_t t) const override
    {
        float64_t2 tangent = computeTangent(t);
        return length(tangent);
    }

private:
    static float64_t getSign(float64_t x)
    {
        return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
    }
};

// Mixes/Interpolation of two Parametric Curves t from 0 to 1
struct MixedParametricCurves final : public ParametricCurve
{
    const ParametricCurve* curve1;
    const ParametricCurve* curve2;

    MixedParametricCurves(const ParametricCurve* curve1, const ParametricCurve* curve2)
        : curve1(curve1), curve2(curve2)
    {}

    float64_t2 computePosition(float64_t t) const override
    {
        const float64_t2 curve1Pos = curve1->computePosition(t);
        const float64_t2 curve2Pos = curve2->computePosition(t);
        return t * (curve2Pos - curve1Pos) + curve1Pos;
    }

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override
    {
        const float64_t2 curve1Pos = curve1->computePosition(t);
        const float64_t2 curve2Pos = curve2->computePosition(t);
        const float64_t2 curve1Tan = curve1->computeTangent(t);
        const float64_t2 curve2Tan = curve2->computeTangent(t);
        return (1-t)*curve1Tan - curve1Pos + (t)*curve2Tan + curve2Pos;
    }

    //! compute unnormalized tangent vector at t
    float64_t2 computeSecondOrderDifferential(float64_t t) const override
    {
        const float64_t2 curve1Tan = curve1->computeTangent(t);
        const float64_t2 curve2Tan = curve2->computeTangent(t);
        const float64_t2 curve1SecondDiff = curve1->computeSecondOrderDifferential(t);
        const float64_t2 curve2SecondDiff = curve2->computeSecondOrderDifferential(t);
        return (1-t)*curve1SecondDiff + 2.0*(curve2Tan-curve1Tan) + t * curve2SecondDiff;
    }

    //! compute differential arc length at t
    float64_t differentialArcLen(float64_t t) const override
    {
        float64_t2 tangent = computeTangent(t);
        return length(tangent);
    }

    float64_t computeInflectionPoint(float64_t errorThreshold) const override
    {
        auto signedCurvatureUnnormalized = [&](float64_t t)
            {
                const float64_t2 first = computeTangent(t);
                const float64_t2 second = computeSecondOrderDifferential(t);
                return float64_t(first.x * second.y - second.x * first.y);
            };

        constexpr uint16_t MaxIterations = 32u;
        float64_t low = 0.0;
        float64_t high = 1.0;
        float64_t valLow = signedCurvatureUnnormalized(low);
        float64_t valHigh = signedCurvatureUnnormalized(high);
        if (getSign(valLow) != getSign(valHigh))
        {
            if (valLow > valHigh)
                std::swap(low, high);

            float64_t guess = 0.0;
            for (uint16_t i = 0u; i < MaxIterations; ++i)
            {
                guess = (low + high) / 2.0;
                float64_t valGuess = signedCurvatureUnnormalized(guess);
                if (abs(valGuess) < errorThreshold)
                    return guess;

                if (valGuess < 0.0)
                    low = guess;
                else
                    high = guess;
            }

            return guess;
        }
        else
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

private:
    static float64_t getSign(float64_t x)
    {
        return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
    }
};

// Mix between two parabolas from 0 to len
struct MixedParabola final : public ExplicitCurve
{
    float64_t a, b, c, d;

    MixedParabola(const Parabola& parabola1, const Parabola& parabola2, float64_t chordLen)
    {
        a = (parabola2.a - parabola1.a) / chordLen;
        b = (parabola2.b - parabola1.b) / chordLen + parabola1.a;
        c = (parabola2.c - parabola1.c) / chordLen + parabola1.b;
        d = parabola1.c;
    }

    static MixedParabola fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3)
    {
        assert(P1.x == 0);
        assert(P1.y == 0 && P2.y == 0);
        auto parabola1 = Parabola::fromThreePoints(P0, P1, P2);
        auto parabola2 = Parabola::fromThreePoints(P1, P2, P3);
        return MixedParabola(parabola1, parabola2, abs(P2.x - P1.x));
    }

    float64_t y(float64_t x) const override
    {
        return (((a * x) + b) * x + c) * x + d;
    }

    float64_t derivative(float64_t x) const override
    {
        return ((3.0 * a * x) + 2.0 * b) * x + c;
    }

    float64_t computeInflectionPoint(float64_t errorThreshold) const override
    {
        return -b / (3.0*a);
    }
};

// Centered at (0,0), aligned with x axis
struct ExplicitEllipse final : public ExplicitCurve
{
    float64_t a, b;
    ExplicitEllipse(float64_t a, float64_t b)
        : a(a), b(b)
    {}

    float64_t y(float64_t x) const override
    {
        return a * sqrt(1.0 - pow((x / b), 2.0));
    }

    float64_t derivative(float64_t x) const override
    {
        return (-a * x) / ((b * b) * sqrt(1.0 - pow((x / b), 2.0)));
    }
};

// Centered at (0, 0), P1 and P2 on x axis and P1.x = -P2.x
struct ExplicitMixedCircle final : public ExplicitCurve
{
    struct ExplicitCircle
    {
        float64_t2 origin;
        float64_t radius;

        ExplicitCircle(const float64_t2& origin, float64_t radius)
            : origin(origin), radius(radius)
        {}

        static ExplicitCircle fromThreePoints(float64_t2 P0, float64_t2 P1, float64_t2 P2)
        {
            const float64_t2 Mid0 = (P0 + P1) / 2.0;
            const float64_t2 Normal0 = float64_t2(P1.y - P0.y, P0.x - P1.x);

            const float64_t2 Mid1 = (P1 + P2) / 2.0;
            const float64_t2 Normal1 = float64_t2(P2.y - P1.y, P1.x - P2.x);

            const float64_t2 origin = LineLineIntersection(Mid0, Normal0, Mid1, Normal1);
            const float64_t radius = glm::length(P0 - origin);
            return ExplicitCircle(origin, radius);

        }
    };

    float64_t origin1Y;
    float64_t origin2Y;
    float64_t radius1;
    float64_t radius2;
    float64_t chordLen;

    static ExplicitMixedCircle fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3)
    {
        ExplicitMixedCircle ret = {};
        assert(P1.x == -P2.x);
        assert(P1.y == 0 && P2.y == 0);
        const ExplicitCircle circle1 = ExplicitCircle::fromThreePoints(P0, P1, P2);
        const ExplicitCircle circle2 = ExplicitCircle::fromThreePoints(P1, P2, P3);
        assert(circle1.origin.x == 0 && circle2.origin.x == 0);
        ret.radius1 = circle1.radius;
        ret.radius2 = circle2.radius;
        ret.origin1Y = circle1.origin.y;
        ret.origin2Y = circle2.origin.y;
        ret.chordLen = abs(P2.x - P1.x);

        return ret;
    }

    float64_t y(float64_t x) const override
    {
        // https://herbie.uwplse.org/demo/5966aab2781d6c55c07105f5b900f1506cad301e.10eb16397304ba9e003d59ffd34803200ce7cfa6/graph.html
        const float64_t s1 = -1.0 * getSign(origin1Y);
        const float64_t s2 = -1.0 * getSign(origin2Y);

        const float64_t t1 = s1 * sqrt(radius1 * radius1 - x * x) + origin1Y;
        const float64_t t2 = s2 * sqrt(radius2 * radius2 - x * x) + origin2Y;
        const float64_t ret = (x / chordLen + 0.5) * (t2 - t1) + t1;
        return ret;
    }

    float64_t secondDerivative(float64_t x) const
    {
        // https://www.wolframalpha.com/input?i=second+derivative+of+%28x%2Fl%2B1%2F2%29*%28s2*sqrt%28r_2%5E2-x%5E2%29%2Bo2-s1*sqrt%28r_1%5E2-x%5E2%29-o1%29%2Bs1*sqrt%28r_1%5E2-x%5E2%29%2Bo1
        const float64_t s1 = -1.0 * getSign(origin1Y);
        const float64_t s2 = -1.0 * getSign(origin2Y);

        const float64_t t1 = sqrt(radius1 * radius1 - x * x);
        const float64_t t2 = sqrt(radius2 * radius2 - x * x);

        const float64_t u1 = (s1 * x) / t1;
        const float64_t u2 = (s2 * x) / t2;

        const float64_t q1 = (-1.0 * (x * x) / pow(radius1 * radius1 - x * x, 1.5)) - (1.0 / t1);
        const float64_t q2 = (-1.0 * (x * x) / pow(radius2 * radius2 - x * x, 1.5)) - (1.0 / t2);

        const float64_t ret = ((2.0 * (u1 - u2)) / chordLen) + (x / chordLen + 0.5) * (s2 * q2 - s1 * q1) + s1 * q1;
        return ret;
    }

    float64_t derivative(float64_t x) const override
    {
        // https://www.wolframalpha.com/input?i=derivative+%28x%2Fl%2B1%2F2%29*%28s2*sqrt%28r_2%5E2-x%5E2%29%2Borigin2Y-s1*sqrt%28r_1%5E2-x%5E2%29-origin1Y%29%2Bs1*sqrt%28r_1%5E2-x%5E2%29%2Borigin1Y
        // https ://herbie.uwplse.org/demo/d48eba9a858160f5f8e4283f7e6211d41215d354.10eb16397304ba9e003d59ffd34803200ce7cfa6/graph.html
        const float64_t s1 = -1.0 * getSign(origin1Y);
        const float64_t s2 = -1.0 * getSign(origin2Y);

        const float64_t t0 = sqrt(radius1 * radius1 - x * x);
        const float64_t t1 = (s1 * x) / t0;
        const float64_t t2 = sqrt(radius2 * radius2 - x * x);
        const float64_t ret = ((x / chordLen + 0.5) * (t1 - ((x * s2) / t2)) + (((origin2Y - origin1Y) + (s2 * t2) - (s1 * t0)) / chordLen)) - t1;
        return ret;
    }

    float64_t computeInflectionPoint(float64_t errorThreshold) const override
    {
        // bisection search to find inflection point
        // by seeing the graph of second derivative over wide range of values we have deduced that the inflection point exists iff the secondDerivative has opposite signs at begin and end
        constexpr uint16_t MaxIterations = 64u;
        float64_t low = -chordLen/2.0;
        float64_t high = chordLen/2.0;
        float64_t valLow = secondDerivative(low + errorThreshold / 2.0);
        float64_t valHigh = secondDerivative(high - errorThreshold / 2.0);
        if (getSign(valLow) != getSign(valHigh))
        {
            if (valLow > valHigh)
                std::swap(low, high);

            float64_t guess = 0.0;
            for (uint16_t i = 0u; i < MaxIterations; ++i)
            {
                guess = (low + high) / 2.0;
                float64_t valGuess = secondDerivative(guess);
                if (abs(valGuess) < errorThreshold)
                    return guess;

                if (valGuess < 0.0)
                    low = guess;
                else
                    high = guess;
            }

            return guess;
        }
        else
        {
            return std::numeric_limits<double>::quiet_NaN();
        }
    }

private:
    static float64_t getSign(float64_t x)
    {
        return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
    }

};

// Fix Bezier Hack for when P1 is "outside" P0 -> P2
// We project P1 into P0->P2 line and see whether it lies inside.
// Because our curves shouldn't go back on themselves in the direction of the chord
inline void fixBezierMidPoint(QuadraticBezierInfo& bezier)
{
    const float64_t2 localChord = bezier.p[2] - bezier.p[0];
    const float localX = dot(normalize(localChord), bezier.p[1] - bezier.p[0]);
    const bool outside = localX<0 || localX>length(localChord);
    if (outside)
    {
        // _NBL_DEBUG_BREAK_IF(true); // this shouldn't happen but we fix it just in case anyways
        bezier.p[1] = bezier.p[0] * 0.4 + bezier.p[2] * 0.6;
    }
}

typedef std::function<void(const QuadraticBezierInfo&)> AddBezierFunc;

inline void adaptiveSubdivision_impl(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t depth)
{
    float64_t split = curve.inverseArcLen_BisectionSearch(0.5, min, max);

    // Shouldn't happen but may happen if we use NewtonRaphson for non convergent inverse CDF
    if (split <= min || split >= max)
    {
        _NBL_DEBUG_BREAK_IF(true);
        split = (min + max) / 2.0;
    }

    const float64_t2 P0 = curve.computePosition(min);
    const float64_t2 V0 = curve.computeTangent(min);
    const float64_t2 P2 = curve.computePosition(max);
    const float64_t2 V2 = curve.computeTangent(max);
    QuadraticBezierInfo bezier = constructBezierWithTwoPointsAndTangents(P0, V0, P2, V2);

    bool shouldSubdivide = false;

    // TODO: compare with certain threshold
    if (depth > 0u && normalize(V0) == normalize(V2))
    {
        shouldSubdivide = true;
    }
    else
    {
        fixBezierMidPoint(bezier);
        if (depth > 0u)
        {
            const float64_t2 curvePositionAtSplit = curve.computePosition(split);
            const float64_t bezierYAtSplit = bezierYatX(bezier, curvePositionAtSplit.x);
            if (isnan(bezierYAtSplit) || abs(curvePositionAtSplit.y - bezierYAtSplit) > targetMaxError)
                shouldSubdivide = true;
        }
    }

    if (shouldSubdivide)
    {
        adaptiveSubdivision_impl(curve, min, split, targetMaxError, addBezierFunc, depth - 1u);
        adaptiveSubdivision_impl(curve, split, max, targetMaxError, addBezierFunc, depth - 1u);
    }
    else
    {
        addBezierFunc(bezier);
    }
}

//! this subdivision algorithm works/converges for any x-monotonic curve (only 1 y for each x) over the [min, max] range and will continue until hits the `maxDepth` or `targetMaxError` threshold
//! this function will call the AddBezierFunc when the bezier is finalized, whether to render it directly, write it to file, add it to a vector, etc.. is up to the user.
//! the subdivision samples the points based on arc length and the error is computed by distance in y direction, so pre and post transform may be needed for your curve and the outputted beziers
inline void adaptiveSubdivision(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12)
{
    // The curves we're working with will have at most 1 inflection point.
    const float64_t inflectX = curve.computeInflectionPoint(targetMaxError * 1e-5); // if no inflection point then this will return NaN and the adaptive subdivision will continue as normal (from min to max)
    if (inflectX > min && inflectX < max)
    {
        adaptiveSubdivision_impl(curve, min, inflectX, targetMaxError, addBezierFunc, maxDepth);
        adaptiveSubdivision_impl(curve, inflectX, max, targetMaxError, addBezierFunc, maxDepth);
    }
    else
        adaptiveSubdivision_impl(curve, min, max, targetMaxError, addBezierFunc, maxDepth);
}
} // namespace curves
#endif