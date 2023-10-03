#ifndef _CAD_EXAMPLE_CURVES_H_
#define _CAD_EXAMPLE_CURVES_H_

#include <nabla.h>
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

using namespace nbl::hlsl;

float64_t2 LineLineIntersection(const float64_t2& p1, const float64_t2& v1, const float64_t2& p2, const float64_t2& v2)
{
    float64_t denominator = v1.y * v2.x - v1.x * v2.y;
    float64_t2 diff = p1 - p2;
    float64_t numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-18 and abs(numerator) < 1e-18)
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
    float64_t y(float64_t x);
    float64_t derivative(float64_t x);
    
    float64_t differentialArcLen(float64_t x)
    {
        float64_t deriv = derivative(x);
        return sqrt(1.0 + deriv * deriv);
    }

    float64_t rcpDifferentialArcLen(float64_t x)
    {
        return 1.0 / differentialArcLen(x);
    }
};

struct Parabola final : public ExplicitCurve
{
    float64_t A, B, C;

    Parabola(float64_t A, float64_t B, float64_t C)
        : A(A), B(B), C(C)
    {}

    static Parabola fromThreePoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2)
    {
        glm::dmat3 X = glm::dmat3(
            glm::dvec3(P0.x*P0.x, P0.x, 1.0),
            glm::dvec3(P1.x*P1.x, P1.x, 1.0),
            glm::dvec3(P2.x*P2.x, P2.x, 1.0)
        );
        glm::dvec3 M = inverse(X) * glm::dvec3(P0.y, P1.y, P2.y);
        return Parabola(M[0], M[1], M[2]);
    }

    float64_t y(float64_t x)
    {
        return ((A * x) + B) * x + C;
    }

    float64_t derivative(float64_t x)
    {
        return 2.0 * A + B;
    }
};

// Mix between two parabolas from 0 to len
struct MixedParabola final : public ExplicitCurve
{
    float64_t A, B, C, D;

    MixedParabola(const Parabola& parabola1, const Parabola& parabola2, float64_t chordLen)
    {
        A = (parabola2.A - parabola1.A) / chordLen;
        B = (parabola2.B - parabola1.B) / chordLen + parabola1.A;
        C = (parabola2.C - parabola1.C) / chordLen + parabola1.B;
        D = parabola1.C;
    }

    static MixedParabola fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3)
    {
        assert(P1.x == 0);
        assert(P1.y == 0 && P2.y == 0);
        auto parabola1 = Parabola::fromThreePoints(P0, P1, P2);
        auto parabola2 = Parabola::fromThreePoints(P1, P2, P3);
        return MixedParabola(parabola1, parabola2, abs(P2.x - P1.x));
    }

    float64_t y(float64_t x)
    {
        return (((A * x) + B) * x + C) * x + D;
    }

    float64_t derivative(float64_t x)
    {
        return ((3.0 * A * x) + 2.0 * B) * x + C;
    }
};

// Centered at (0,0), aligned with x axis
struct ExplicitEllipse final : public ExplicitCurve
{
    float64_t a, b;
    ExplicitEllipse(float64_t a, float64_t b) 
        : a(a), b(b)
    {}

    float64_t y(float64_t x)
    {
        return a * sqrt(1.0 - pow((x / b), 2.0));
    }

    float64_t derivative(float64_t x)
    {
        return (-a * x) / ((b * b) * sqrt(1.0 - pow((x / b), 2.0)));
    }
};

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

    float64_t y(float64_t x)
    {
        // https://herbie.uwplse.org/demo/5966aab2781d6c55c07105f5b900f1506cad301e.10eb16397304ba9e003d59ffd34803200ce7cfa6/graph.html
        const float64_t s1 = -1.0 * getSign(origin1Y);
        const float64_t s2 = -1.0 * getSign(origin2Y);

        const float64_t t1 = s1 * sqrt(radius1 * radius1 - x * x) + origin1Y;
        const float64_t t2 = s2 * sqrt(radius2 * radius2 - x * x) + origin2Y;
        const float64_t ret = (x / chordLen + 0.5) * (t2 - t1) + t1;
        return ret;
    }

    float64_t derivative(float64_t x)
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
    float64_t getSign(float64_t x)
    {
        return static_cast<float64_t>((x >= 0.0)) - static_cast<float64_t>((x < 0.0));
    }

};

#endif