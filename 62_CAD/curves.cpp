
#include "curves.h"
#include <nbl/builtin/hlsl/math/quadrature/gauss_legendre/gauss_legendre.hlsl>
#include <nbl/builtin/hlsl/shapes/util.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>

using namespace nbl::hlsl;

namespace curves
{

float64_t ParametricCurve::arcLen(float64_t t0, float64_t t1) const
{
    constexpr uint16_t IntegrationOrder = 10u;
    return nbl::hlsl::math::quadrature::GaussLegendreIntegration<IntegrationOrder, double, ArcLenIntegrand>::calculateIntegral(ArcLenIntegrand(this), t0, t1);
}

float64_t ParametricCurve::inverseArcLen_BisectionSearch(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold, const uint16_t iterationThreshold) const
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

float64_t ParametricCurve::inverseArcLen(float64_t targetLen, float64_t min, float64_t max, const float64_t cdfAccuracyThreshold) const
{
    return inverseArcLen_BisectionSearch(targetLen, min, max, cdfAccuracyThreshold);
}

float64_t ParametricCurve::differentialArcLen(float64_t t) const
{
    return length(computeTangent(t));
}

float64_t ExplicitCurve::differentialArcLen(float64_t x) const
{
    float64_t deriv = derivative(x);
    return sqrt(1.0 + deriv * deriv);
}

float64_t2 ExplicitCurve::computeTangent(float64_t x) const
{
    const float64_t deriv = derivative(x);
    float64_t2 v = float64_t2(1.0, deriv);
    if (isinf(deriv))
        v = float64_t2(0.0, 1.0);
    return v;
}

Parabola Parabola::fromThreePoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2)
{
    glm::dmat3 X = glm::dmat3(
        glm::dvec3(P0.x * P0.x, P0.x, 1.0),
        glm::dvec3(P1.x * P1.x, P1.x, 1.0),
        glm::dvec3(P2.x * P2.x, P2.x, 1.0)
    );
    glm::dvec3 M = inverse(transpose(X)) * glm::dvec3(P0.y, P1.y, P2.y);
    return Parabola(M[0], M[1], M[2]);
}

float64_t Parabola::y(float64_t x) const
{
    return ((a * x) + b) * x + c;
}

float64_t Parabola::derivative(float64_t x) const
{
    return 2.0 * a * x + b;
}

float64_t2 CubicCurve::computePosition(float64_t t) const
{
    return float64_t2(
        ((X[0] * t + X[1]) * t + X[2]) * t + X[3],
        ((Y[0] * t + Y[1]) * t + Y[2]) * t + Y[3]
    );
}

float64_t2 CubicCurve::computeTangent(float64_t t) const
{
    return float64_t2(
        (3.0 * X[0] * t + 2.0 * X[1]) * t + X[2],
        (3.0 * Y[0] * t + 2.0 * Y[1]) * t + Y[2]
    );
}

float64_t2 CubicCurve::computeSecondOrderDifferential(float64_t t) const
{
    return float64_t2(
        6.0 * X[0] * t + 2.0 * X[1],
        6.0 * Y[0] * t + 2.0 * Y[1]
    );
}

float64_t CubicCurve::computeInflectionPoint(float64_t errorThreshold) const
{
    // solve for signed curvature root 
    // when x'*y''-x''*y' = 0
    // https://www.wolframalpha.com/input?i=cross+product+%283*x0*t%5E2%2B2*x1%2Bx2%2C3*y0*t%5E2%2B2*y1%2By2%29+and+%286*x0*t%2B2*x1%2C6*y0*t%2B2*y1%29
    const float64_t a = 6.0 * (X[0] * Y[1] - X[1] * Y[0]);
    const float64_t b = 6.0 * (2.0 * X[1] * Y[0] - 2.0 * X[0] * Y[1] + X[2] * Y[0] - X[0] * Y[2]);
    const float64_t c = 2.0 * (X[2] * Y[1] - X[1] * Y[2]);

    math::equations::Quadratic<float64_t> quadratic = math::equations::Quadratic<float64_t>::construct(a, b, c);
    const float64_t2 roots = quadratic.computeRoots();
    if (roots[0] <= 1.0 && roots[0] >= 0.0)
        return roots[0];
    if (roots[1] <= 1.0 && roots[1] >= 0.0)
        return roots[1];
    return std::numeric_limits<double>::quiet_NaN();
}

float64_t2 CircularArc::computePosition(float64_t t) const
{
    const float64_t actualT = t * sweepAngle + startAngle;
    return float64_t2(
        r * cos(actualT),
        r * sin(actualT) + originY
    );
}

float64_t2 CircularArc::computeTangent(float64_t t) const
{
    const float64_t actualT = t * sweepAngle + startAngle;
    return float64_t2(
        -1.0 * r * sweepAngle * sin(actualT),
        +1.0 * r * sweepAngle * cos(actualT)
    );
}

float64_t2 CircularArc::computeSecondOrderDifferential(float64_t t) const
{
    const float64_t actualT = t * sweepAngle + startAngle;
    return float64_t2(
        -1.0 * r * sweepAngle * sweepAngle * cos(actualT),
        -1.0 * r * sweepAngle * sweepAngle * sin(actualT)
    );
}

float64_t CircularArc::getSign(float64_t x)
{
    return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
}

float64_t2 MixedParametricCurves::computePosition(float64_t t) const
{
    const float64_t2 curve1Pos = curve1->computePosition(t);
    const float64_t2 curve2Pos = curve2->computePosition(t);
    return t * (curve2Pos - curve1Pos) + curve1Pos;
}

float64_t2 MixedParametricCurves::computeTangent(float64_t t) const
{
    const float64_t2 curve1Pos = curve1->computePosition(t);
    const float64_t2 curve2Pos = curve2->computePosition(t);
    const float64_t2 curve1Tan = curve1->computeTangent(t);
    const float64_t2 curve2Tan = curve2->computeTangent(t);
    return (1 - t) * curve1Tan - curve1Pos + (t)*curve2Tan + curve2Pos;
}

float64_t2 MixedParametricCurves::computeSecondOrderDifferential(float64_t t) const
{
    const float64_t2 curve1Tan = curve1->computeTangent(t);
    const float64_t2 curve2Tan = curve2->computeTangent(t);
    const float64_t2 curve1SecondDiff = curve1->computeSecondOrderDifferential(t);
    const float64_t2 curve2SecondDiff = curve2->computeSecondOrderDifferential(t);
    return (1 - t) * curve1SecondDiff + 2.0 * (curve2Tan - curve1Tan) + t * curve2SecondDiff;
}

float64_t MixedParametricCurves::computeInflectionPoint(float64_t errorThreshold) const
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

float64_t MixedParametricCurves::getSign(float64_t x)
{
    return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
}

MixedParabola MixedParabola::fromFourPoints(const float64_t2& P0, const float64_t2& P1, const float64_t2& P2, const float64_t2& P3)
{
    assert(P1.x == 0);
    assert(P1.y == 0 && P2.y == 0);
    auto parabola1 = Parabola::fromThreePoints(P0, P1, P2);
    auto parabola2 = Parabola::fromThreePoints(P1, P2, P3);
    return MixedParabola(parabola1, parabola2, abs(P2.x - P1.x));
}

float64_t MixedParabola::y(float64_t x) const
{
    return (((a * x) + b) * x + c) * x + d;
}

float64_t MixedParabola::derivative(float64_t x) const
{
    return ((3.0 * a * x) + 2.0 * b) * x + c;
}

float64_t MixedParabola::computeInflectionPoint(float64_t errorThreshold) const
{
    return -b / (3.0 * a);
}

float64_t ExplicitEllipse::y(float64_t x) const
{
    return a * sqrt(1.0 - pow((x / b), 2.0));
}

float64_t ExplicitEllipse::derivative(float64_t x) const
{
    return (-a * x) / ((b * b) * sqrt(1.0 - pow((x / b), 2.0)));
}

float64_t2 AxisAlignedEllipse::computePosition(float64_t t) const
{
    const float64_t theta = start + (end - start) * t;
    return float64_t2(a * cos(theta), b * sin(theta));
}

float64_t2 AxisAlignedEllipse::computeTangent(float64_t t) const
{
    const float64_t theta = start + (end - start) * t;
    const float64_t dThetaDt = end - start;
    return float64_t2(-a * dThetaDt * sin(theta), b * dThetaDt * cos(theta));
}

ExplicitMixedCircle::ExplicitCircle ExplicitMixedCircle::ExplicitCircle::fromThreePoints(float64_t2 P0, float64_t2 P1, float64_t2 P2)
{
    const float64_t2 Mid0 = (P0 + P1) / 2.0;
    const float64_t2 Normal0 = float64_t2(P1.y - P0.y, P0.x - P1.x);

    const float64_t2 Mid1 = (P1 + P2) / 2.0;
    const float64_t2 Normal1 = float64_t2(P2.y - P1.y, P1.x - P2.x);

    const float64_t2 origin = shapes::util::LineLineIntersection(Mid0, Normal0, Mid1, Normal1);
    const float64_t radius = glm::length(P0 - origin);
    return ExplicitCircle(origin, radius);

}
float64_t ExplicitMixedCircle::y(float64_t x) const
{
    // https://herbie.uwplse.org/demo/5966aab2781d6c55c07105f5b900f1506cad301e.10eb16397304ba9e003d59ffd34803200ce7cfa6/graph.html
    const float64_t s1 = -1.0 * getSign(origin1Y);
    const float64_t s2 = -1.0 * getSign(origin2Y);

    const float64_t t1 = s1 * sqrt(radius1 * radius1 - x * x) + origin1Y;
    const float64_t t2 = s2 * sqrt(radius2 * radius2 - x * x) + origin2Y;
    const float64_t ret = (x / chordLen + 0.5) * (t2 - t1) + t1;
    return ret;
}

float64_t ExplicitMixedCircle::secondDerivative(float64_t x) const
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

float64_t ExplicitMixedCircle::derivative(float64_t x) const
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

float64_t ExplicitMixedCircle::computeInflectionPoint(float64_t errorThreshold) const
{
    // bisection search to find inflection point
    // by seeing the graph of second derivative over wide range of values we have deduced that the inflection point exists iff the secondDerivative has opposite signs at begin and end
    constexpr uint16_t MaxIterations = 64u;
    float64_t low = -chordLen / 2.0;
    float64_t high = chordLen / 2.0;
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

float64_t ExplicitMixedCircle::getSign(float64_t x)
{
    return static_cast<float64_t>((x > 0.0)) - static_cast<float64_t>((x <= 0.0));
}

float64_t2 OffsettedBezier::computePosition(float64_t t) const
{
    const float64_t2 deriv = quadratic.derivative(t);
    const float64_t2 normal = normalize(float64_t2(deriv.y, -deriv.x));
    return quadratic.evaluate(t) + offset * normal;
}

float64_t2 OffsettedBezier::computeTangent(float64_t t) const
{
    const float64_t2 ddt = quadratic.derivative(t);
    const float64_t2 d2dt2 = quadratic.secondDerivative(t);
    const float64_t g = offset * (ddt.x * d2dt2.y - ddt.y * d2dt2.x);
    return ddt + (ddt * g) / glm::length(ddt);
}

float64_t2 OffsettedBezier::findCusps() const
{
    // we're basically solving for t in "offset = radiusOfCurvature(t)"
    const float64_t lhs = pow(offset * 2.0 * abs(quadratic.B.x * quadratic.A.y - quadratic.B.y * quadratic.A.x), 2.0 / 3.0);
    const float64_t a = 4.0 * (quadratic.A.x * quadratic.A.x + quadratic.A.y * quadratic.A.y);
    const float64_t b = 4.0 * (quadratic.A.x * quadratic.B.x + quadratic.A.y * quadratic.B.y);
    const float64_t c = quadratic.B.x * quadratic.B.x + quadratic.B.y * quadratic.B.y - lhs;
    math::equations::Quadratic<float64_t> findCuspsQuadratic = math::equations::Quadratic<float64_t>::construct(a, b, c);
    return findCuspsQuadratic.computeRoots();
}

// Fix Bezier Hack for when P1 is "outside" P0 -> P2
// We project P1 into P0->P2 line and see whether it lies inside.
// Because our curves shouldn't go back on themselves in the direction of the chord
static void fixBezierMidPoint(shapes::QuadraticBezier<double>& bezier)
{
    const float64_t2 localChord = bezier.P2 - bezier.P0;
    const float64_t localX = dot(normalize(localChord), bezier.P1 - bezier.P0);
    const bool outside = localX<0 || localX>length(localChord);
    if (outside || isnan(bezier.P1.x) || isnan(bezier.P1.y))
    {
        // _NBL_DEBUG_BREAK_IF(true); // this shouldn't happen but we fix it just in case anyways
        bezier.P1 = bezier.P0 * 0.4 + bezier.P2 * 0.6;
    }
}

void Subdivision::adaptive(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth)
{
    // The curves we're working with will have at most 1 inflection point.
    const float64_t inflectX = curve.computeInflectionPoint(targetMaxError); // if no inflection point then this will return NaN and the adaptive subdivision will continue as normal (from min to max)
    if (inflectX > min && inflectX < max)
    {
        adaptive_impl(curve, min, inflectX, targetMaxError, addBezierFunc, maxDepth);
        adaptive_impl(curve, inflectX, max, targetMaxError, addBezierFunc, maxDepth);
    }
    else
        adaptive_impl(curve, min, max, targetMaxError, addBezierFunc, maxDepth);
}

void Subdivision::adaptive(const EllipticalArcInfo& ellipse, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth)
{
    using namespace nbl::hlsl;

    if (!ellipse.isValid())
    {
        _NBL_DEBUG_BREAK_IF(true);
        return;
    }

    float64_t lenghtMajor = length(ellipse.majorAxis);
    float64_t lenghtMinor = lenghtMajor * ellipse.eccentricity;
    float64_t2 normalizedMajor = ellipse.majorAxis / lenghtMajor;

    float64_t2x2 rotate = float64_t2x2({
        float64_t2(normalizedMajor.x, -normalizedMajor.y),
        float64_t2(normalizedMajor.y, normalizedMajor.x)
        });

    AddBezierFunc addTransformedBezier = [&](shapes::QuadraticBezier<double>&& quadBezier)
        {
            quadBezier.P0 = mul(rotate, quadBezier.P0);
            quadBezier.P1 = mul(rotate, quadBezier.P1);
            quadBezier.P2 = mul(rotate, quadBezier.P2);
            quadBezier.P0 += ellipse.center;
            quadBezier.P1 += ellipse.center;
            quadBezier.P2 += ellipse.center;
            addBezierFunc(std::move(quadBezier));
        };

    if (ellipse.angleBounds.x != ellipse.angleBounds.y)
    {
        AxisAlignedEllipse aaEllipse(lenghtMajor, lenghtMinor, ellipse.angleBounds.x, ellipse.angleBounds.y);
        adaptive(aaEllipse, 0.0, 1.0, targetMaxError, addTransformedBezier, maxDepth);
    }
}

void Subdivision::adaptive(const OffsettedBezier& curve, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t maxDepth)
{
    const float64_t2 cusps = curve.findCusps();

    const float64_t t0 = min(cusps[0], cusps[1]);
    const float64_t t1 = max(cusps[0], cusps[1]);

    const bool firstCusp = t0 > 0.0 && t0 < 1.0;
    const bool secondCusp = t1 > 0.0 && t1 < 1.0;

    // if there are two cusps (offset = radius of curvature) then we have that unwanted gouging and we prefer to seperately subdivide those three sections
    if (firstCusp && secondCusp)
    {
        adaptive_impl(curve, 0.0, t0, targetMaxError, addBezierFunc, maxDepth);
        adaptive_impl(curve, t0, t1, targetMaxError, addBezierFunc, maxDepth);
        adaptive_impl(curve, t1, 1.0, targetMaxError, addBezierFunc, maxDepth);
    }
    // otherwise just subdivide from start/0.0 to end/1.0
    else
    {
        adaptive_impl(curve, 0.0, 1.0, targetMaxError, addBezierFunc, maxDepth);
    }
}

void Subdivision::adaptive_impl(const ParametricCurve& curve, float64_t min, float64_t max, float64_t targetMaxError, AddBezierFunc& addBezierFunc, uint32_t depth)
{
    if (min == max)
        return;
    assert(min < max);

    float64_t split = curve.inverseArcLen_BisectionSearch(0.5, min, max);

    // Shouldn't happen but may happen if we use NewtonRaphson for non convergent inverse CDF
    if (split <= min || split >= max)
    {
        _NBL_DEBUG_BREAK_IF(split < min || split > max);
        split = (min + max) / 2.0;
    }

    const float64_t2 P0 = curve.computePosition(min);
    const float64_t2 V0 = curve.computeTangent(min);
    const float64_t2 P2 = curve.computePosition(max);
    const float64_t2 V2 = curve.computeTangent(max);
    shapes::QuadraticBezier<double> bezier = shapes::QuadraticBezier<double>::constructBezierWithTwoPointsAndTangents(P0, V0, P2, V2);

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
            if (glm::distance(P0, P2) < targetMaxError)
            {
                const float64_t2 posAtSplit = curve.computePosition(split);
                // If it came down to a bezier small that causes P0 P2 and the position at split smaller than targetMaxError then we stop
                if (glm::distance(posAtSplit, P0) < targetMaxError)
                    shouldSubdivide = false;
                // But sometimes when P0 and P2 are close together a split will fix them, like a full circle and needs further subdivision
                else
                    shouldSubdivide = true;
            }
            else
            {
                const float64_t2 curvePositionAtSplit = curve.computePosition(split);
                float64_t bezierYAtSplit = bezier.calcYatX(curvePositionAtSplit.x);
                //_NBL_DEBUG_BREAK_IF(isnan(bezierYAtSplit)); 
                // TODO: maybe a better error comaprison is find the normal at split and intersect with the bezier
                if (isnan(bezierYAtSplit) || abs(curvePositionAtSplit.y - bezierYAtSplit) > targetMaxError)
                    shouldSubdivide = true;
            }
        }
    }

    if (shouldSubdivide)
    {
        adaptive_impl(curve, min, split, targetMaxError, addBezierFunc, depth - 1u);
        adaptive_impl(curve, split, max, targetMaxError, addBezierFunc, depth - 1u);
    }
    else
    {
        const bool degenerate = (bezier.P0 == bezier.P2);
        if (!degenerate)
            addBezierFunc(std::move(bezier));
    }
}

}
