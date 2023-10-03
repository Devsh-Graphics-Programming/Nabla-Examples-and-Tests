#ifndef _CAD_EXAMPLE_CURVES_H_
#define _CAD_EXAMPLE_CURVES_H_

#include <nabla.h>
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>

using namespace nbl::hlsl;

float64_t2 LineLineIntersection(float64_t2 p1, float64_t2 v1, float64_t2 p2, float64_t2 v2)
{
    double denominator = v1.y * v2.x - v1.x * v2.y;
    float64_t2 diff = p1 - p2;
    double numerator = dot(float64_t2(v2.y, -v2.x), float64_t2(diff.x, diff.y));

    if (abs(denominator) < 1e-18 and abs(numerator) < 1e-18)
    {
        // are parallel and the same
        return (p1 + p2) / 2.0;
    }

    double t = numerator / denominator;
    float64_t2 intersectionPoint = p1 + t * v1;
    return intersectionPoint;
}

struct ExplicitCurve
{
    double y(double x);
    double derivative(double x);
    
    double differentialArcLen(double x)
    {
        double deriv = derivative(x);
        return sqrt(1.0 + deriv * deriv);
    }

    double rcpDifferentialArcLen(double x)
    {
        return 1.0 / differentialArcLen(x);
    }
};

struct Parabola : ExplicitCurve
{
    double A, B, C;

    Parabola(double A, double B, double C)
        : A(A), B(B), C(C)
    {}

    static Parabola fromThreePoints(float64_t2 P0, float64_t2 P1, float64_t2 P2)
    {
        glm::dmat3 X = glm::dmat3(
            glm::dvec3(P0.x*P0.x, P0.x, 1.0),
            glm::dvec3(P1.x*P1.x, P1.x, 1.0),
            glm::dvec3(P2.x*P2.x, P2.x, 1.0)
        );
        glm::dvec3 M = inverse(X) * glm::dvec3(P0.y, P1.y, P2.y);
        return Parabola(M[0], M[1], M[2]);
    }

    double y(double x)
    {
        return ((A * x) + B) * x + C;
    }

    double derivative(double x)
    {
        return 2.0 * A + B;
    }
};

// Mix between two parabolas from 0 to len
struct MixedParabola
{
    double A, B, C, D;

    MixedParabola(const Parabola& parabola1, const Parabola& parabola2, double chordLen)
    {
        A = (parabola2.A - parabola1.A) / chordLen;
        B = (parabola2.B - parabola1.B) / chordLen + parabola1.A;
        C = (parabola2.C - parabola1.C) / chordLen + parabola1.B;
        D = parabola1.C;
    }

    static MixedParabola fromFourPoints(float64_t2 P0, float64_t2 P1, float64_t2 P2, float64_t2 P3)
    {
        assert(P1.x == 0);
        assert(P1.y == 0 && P2.y == 0);
        auto parabola1 = Parabola::fromThreePoints(P0, P1, P2);
        auto parabola2 = Parabola::fromThreePoints(P1, P2, P3);
        return MixedParabola(parabola1, parabola2, abs(P2.x - P1.x));
    }

    double y(double x)
    {
        return (((A * x) + B) * x + C) * x + D;
    }

    double derivative(double x)
    {
        return ((3.0 * A * x) + 2.0 * B) * x + C;
    }
};

#endif