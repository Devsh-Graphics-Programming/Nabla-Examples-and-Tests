#ifndef _CAD_EXAMPLE_CURVES_H_
#define _CAD_EXAMPLE_CURVES_H_

#include <nabla.h>
#include "glm/glm/glm.hpp"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
using namespace nbl::hlsl;

#include "common.hlsl"
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>


namespace curves
{
// Base class for all our curves
struct ParametricCurve
{
    //! compute position at t
    virtual float64_t2 computePosition(float64_t t) const = 0;

    //! compute unnormalized tangent vector at t
    virtual float64_t2 computeTangent(float64_t t) const = 0;

    //! compute differential arc length at t
    virtual float64_t differentialArcLen(float64_t t) const;

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
    float64_t arcLen(float64_t t0, float64_t t1) const;

    //! compute inverse arc len using bisection search
    float64_t inverseArcLen_BisectionSearch(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold = 1e-4, const uint16_t iterationThreshold = 16u) const;

    //! compute inverse arc len  
    float64_t inverseArcLen(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold = 1e-4) const;

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

    float64_t differentialArcLen(float64_t x) const override;
    float64_t2 computeTangent(float64_t x) const override;
    inline float64_t2 computePosition(float64_t x) const override { return float64_t2(x, y(x)); }
};

struct Parabola final : public ExplicitCurve
{
    float64_t a, b, c;

    Parabola(float64_t a, float64_t b, float64_t c)
        : a(a), b(b), c(c)
    {}

    static Parabola fromThreePoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2);

    float64_t y(float64_t x) const override;
    float64_t derivative(float64_t x) const override;
};

struct CubicCurve final : public ParametricCurve
{
    float64_t4 X;
    float64_t4 Y;

    CubicCurve(const float64_t4& X, const float64_t4& Y)
        : X(X), Y(Y)
    {}

    float64_t2 computePosition(float64_t t) const override;

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override;

    //! compute second order differential at t
    float64_t2 computeSecondOrderDifferential(float64_t t) const override;

    float64_t computeInflectionPoint(float64_t errorThreshold) const override;
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
        sweepAngle = -2.0 * getSign(v.y) * acos(abs(originY) / r);
    }

    float64_t2 computePosition(float64_t t) const override;

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override;

    float64_t2 computeSecondOrderDifferential(float64_t t) const override;

private:
    static float64_t getSign(float64_t x);
};

// Mixes/Interpolation of two Parametric Curves t from 0 to 1
struct MixedParametricCurves final : public ParametricCurve
{
    const ParametricCurve* curve1;
    const ParametricCurve* curve2;

    MixedParametricCurves(const ParametricCurve* curve1, const ParametricCurve* curve2)
        : curve1(curve1), curve2(curve2)
    {}

    float64_t2 computePosition(float64_t t) const override;

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override;

    //! compute second order differential at t
    float64_t2 computeSecondOrderDifferential(float64_t t) const override;

    float64_t computeInflectionPoint(float64_t errorThreshold) const override;

private:
    static float64_t getSign(float64_t x);
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

    static MixedParabola fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3);

    float64_t y(float64_t x) const override;

    float64_t derivative(float64_t x) const override;

    float64_t computeInflectionPoint(float64_t errorThreshold) const override;
};

// Centered at (0,0), aligned with x axis
struct ExplicitEllipse final : public ExplicitCurve
{
    float64_t a, b;
    ExplicitEllipse(float64_t a, float64_t b)
        : a(a), b(b)
    {}

    float64_t y(float64_t x) const override;

    float64_t derivative(float64_t x) const override;
};

// Centered at (0,0), aligned with x axis
struct AxisAlignedEllipse final : public ParametricCurve
{
    float64_t a, b;
    float64_t start, end;
    AxisAlignedEllipse(float64_t a, float64_t b, float64_t start, float64_t end)
        : a(a), b(b), start(start), end(end)
    {}

    float64_t2 computePosition(float64_t t) const override;

    float64_t2 computeTangent(float64_t t) const override;
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

        static ExplicitCircle fromThreePoints(float64_t2 P0, float64_t2 P1, float64_t2 P2);
    };

    float64_t origin1Y;
    float64_t origin2Y;
    float64_t radius1;
    float64_t radius2;
    float64_t chordLen;

    static ExplicitMixedCircle fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3);

    float64_t y(float64_t x) const override;

    float64_t secondDerivative(float64_t x) const;

    float64_t derivative(float64_t x) const override;

    float64_t computeInflectionPoint(float64_t errorThreshold) const override;

private:
    static float64_t getSign(float64_t x);
};

struct EllipticalArcInfo
{
    float64_t2 majorAxis;
    float64_t2 center;
    float64_t2 angleBounds; // [0, 2Pi)
    double eccentricity; // (0, 1]

    inline bool isValid() const
    {
        if (eccentricity > 1.0 || eccentricity <= 0.0)
            return false;
        if (angleBounds.y == angleBounds.x)
            return false;
        if (abs(angleBounds.y - angleBounds.x) > 2.0 * nbl::core::PI<double>())
            return false;
        return true;
    }
};

struct OffsettedBezier : public ParametricCurve
{
    nbl::hlsl::shapes::Quadratic<float64_t> quadratic;
    float64_t offset;

    OffsettedBezier(const nbl::hlsl::shapes::QuadraticBezier<double>& quadBezier, float64_t offset)
        : offset(offset)
    {
        quadratic = nbl::hlsl::shapes::Quadratic<float64_t>::constructFromBezier(quadBezier.P0, quadBezier.P1, quadBezier.P2);
    }

    float64_t2 computePosition(float64_t t) const override;

    //! compute unnormalized tangent vector at t
    float64_t2 computeTangent(float64_t t) const override;

    //! if offset is more than minimum radius of curvature then we get an unwanted gouging/cusp
    float64_t2 findCusps() const;
};

class Subdivision final
{
public:
    typedef std::function<void(nbl::hlsl::shapes::QuadraticBezier<double>&&)> AddBezierFunc;

    //! this subdivision algorithm works/converges for any x-monotonic curve (only 1 y for each x) over the [min, max] range and will continue until hits the `maxDepth` or `targetMaxError` threshold
    //! this function will call the AddBezierFunc when the bezier is finalized, whether to render it directly, write it to file, add it to a vector, etc.. is up to the user.
    //! the subdivision samples the points based on arc length and the error is computed by distance in y direction, so pre and post transform may be needed for your curve and the outputted beziers
    //! it will first split at inflection point of the curve; curves are assumed to have at most 1 inflection point, and will get the best convergence rates. but it will work for curves with more inflection points as well.
    static void adaptive(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12);

    static void adaptive(const EllipticalArcInfo& ellipse, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12);
        
    static void adaptive(const OffsettedBezier& curve, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth = 12);

private:
    static void adaptive_impl(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t depth);

};
} // namespace curves
#endif