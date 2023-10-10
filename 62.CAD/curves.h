#ifndef _CAD_EXAMPLE_CURVES_H_
#define _CAD_EXAMPLE_CURVES_H_

#include <nabla.h>
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
using namespace nbl::hlsl;

#include "common.hlsl"
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>

//TODO: Share hlsl and cpp
float64_t2 LineLineIntersection(const float64_t2& p1, const float64_t2& v1, const float64_t2& p2, const float64_t2& v2)
{
    float64_t denominator = v1.y * v2.x - v1.x * v2.y;
    float64_t2 diff = p1 - p2;
    float64_t numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-5 && abs(numerator) < 1e-5)
    {
        // are parallel and the same
        return (p1 + p2) / 2.0;
    }

    float64_t t = numerator / denominator;
    float64_t2 intersectionPoint = p1 + t * v1;
    return intersectionPoint;
}

struct ExplicitCurve
{
    virtual float64_t y(float64_t x) const = 0;
    virtual float64_t derivative(float64_t x) const = 0;

    float64_t differentialArcLen(float64_t x) const
    {
        float64_t deriv = derivative(x);
        return sqrt(1.0 + deriv * deriv);
    }
    float64_t rcpDifferentialArcLen(float64_t x) const
    {
        return 1.0 / differentialArcLen(x);
    }

    // for Integrand Operator
    inline float64_t operator()(const float64_t x) const
    {
        return differentialArcLen(x);
    }
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

    float64_t inflectionPoint() const
    {
        return b / (3.0*a);
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
struct MixedCircle final : public ExplicitCurve
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

    static MixedCircle fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3)
    {
        MixedCircle ret = {};
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

private:
    float64_t getSign(float64_t x) const
    {
        return static_cast<float64_t>((x >= 0.0)) - static_cast<float64_t>((x < 0.0));
    }

};

typedef std::function<void(const QuadraticBezierInfo&)> AddBezierFunc;

inline float64_t bezierYatT(const QuadraticBezierInfo& bezier, float64_t t)
{
    const float64_t a = bezier.p[0].y - 2.0 * bezier.p[1].y + bezier.p[2].y;
    const float64_t b = 2.0 * (bezier.p[1].y - bezier.p[0].y);
    const float64_t c = bezier.p[0].y;
    return ((a * t) + b) * t + c; // eval at t1
}

// To compute error from actual curve
inline float64_t bezierYatX(const QuadraticBezierInfo& bezier, float64_t x)
{
    const float64_t a = bezier.p[0].x - 2.0 * bezier.p[1].x + bezier.p[2].x;
    const float64_t b = 2.0 * (bezier.p[1].x - bezier.p[0].x);
    const float64_t c = bezier.p[0].x - x;
    const float64_t det = b * b - 4.0 * a * c;
    const float64_t detSqrt = sqrt(det);
    const float64_t rcp = 0.5 / a;
    const float64_t bOver2A = b * rcp;

    float64_t t0 = 0.0, t1 = 0.0;
    if (b >= 0)
    {
        t0 = -detSqrt * rcp - bOver2A;
        t1 = 2 * c / (-b - detSqrt);
    }
    else
    {
        t0 = 2 * c / (-b + detSqrt);
        t1 = +detSqrt * rcp - bOver2A;
    }

    if (t0 >= 0.0 && t0 <= 1.0)
        return bezierYatT(bezier, t0);
    else if (t1 >= 0.0 && t1 <= 1.0)
        return bezierYatT(bezier, t1);
    else
        return std::numeric_limits<double>::quiet_NaN();

}

inline QuadraticBezierInfo constructBezierWithTwoPointsAndTangents(float64_t2 P0, float64_t t0, float64_t2 P2, float64_t t2)
{
    QuadraticBezierInfo out = {};
    out.p[0] = P0;
    out.p[2] = P2;

    float64_t2 v0 = float64_t2(1.0, t0);
    if (isinf(t0))
        v0 = float64_t2(0.0, 1.0);

    float64_t2 v2 = float64_t2(1.0, t2);
    if (isinf(t2))
        v2 = float64_t2(0.0, 1.0);

    out.p[1] = LineLineIntersection(P0, v0, P2, v2);
    return out;
}

inline float64_t cdf(const ExplicitCurve& curve, float64_t min, float64_t max)
{
    constexpr uint16_t IntegrationOrder = 10u;
    return nbl::hlsl::math::quadrature::GaussLegendreIntegration<IntegrationOrder, double, ExplicitCurve>::calculateIntegral(curve, min, max);
}

inline float64_t inverseCDF_Bisection(const ExplicitCurve& curve, float64_t targetCDF, float64_t min, float64_t max)
{
    constexpr float64_t cdfAccuracyThreshold = 1e-4;
    constexpr uint16_t iterationThreshold = 16u;

    float64_t xi = 0.0; // return value

    float64_t low = min;
    float64_t high = max;
    for (uint16_t i = 0; i < iterationThreshold; ++i)
    {
        xi = (low + high) / 2.0;
        float64_t sum = cdf(curve, min, xi);
        float64_t integral = sum + cdf(curve, xi, max);

        // we could've done sum/integral - targetCDF, but this is more robust as it avoids a divsion
        float64_t valueAtParamGuess = sum - targetCDF * integral;

        if (abs(valueAtParamGuess) < cdfAccuracyThreshold * integral)
            return xi; // we found xi value that gives us a cdf of targetCDF within cdfAccuracyThreshold
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

inline void adaptiveSubdivision_impl(const ExplicitCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t depth)
{
    float64_t split = inverseCDF_Bisection(curve, 0.5, min, max);

    // Shouldn't happen but may happen if we don't use bisection for inverse CDF
    if (split <= min || split >= max)
    {
        _NBL_DEBUG_BREAK_IF(true);
        split = (min + max) / 2.0;
    }

    const float64_t2 P0 = float64_t2(min, curve.y(min));
    const float64_t P0Tan = curve.derivative(min);
    const float64_t2 P2 = float64_t2(max, curve.y(max));
    const float64_t P2Tan = curve.derivative(max);
    QuadraticBezierInfo bezier = constructBezierWithTwoPointsAndTangents(P0, P0Tan, P2, P2Tan);

    bool shouldSubdivide = false;
    if (depth > 0u)
    {
        constexpr double inf = std::numeric_limits<double>::infinity();
        if (P0Tan == P2Tan || (abs(P0Tan) == inf && abs(P2Tan) == inf))
        {
            shouldSubdivide = true;
        }
        else
        {
            float64_t curveValueAtSplit = curve.y(split);
            float64_t bezierValueAtSplit = bezierYatX(bezier, split);
            if (abs(curveValueAtSplit - bezierValueAtSplit) > targetMaxError)
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

inline void adaptiveSubdivision(const ExplicitCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12)
{
    adaptiveSubdivision_impl(curve, min, max, targetMaxError, addBezierFunc, maxDepth);
}

#endif