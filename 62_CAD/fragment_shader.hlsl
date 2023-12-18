#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

template<typename float_t>
struct DefaultClipper
{
    using float_t2 = vector<float_t, 2>;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.0;

    static DefaultClipper construct()
    {
        DefaultClipper ret;
        return ret;
    }

    inline float_t2 operator()(const float_t t)
    {
        const float_t ret = clamp(t, 0.0, 1.0);
        return float_t2(ret, ret);
    }
};

template<typename CurveType, typename StyleAccessor>
struct StyleClipper
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.000001;

    static StyleClipper<CurveType, StyleAccessor> construct(StyleAccessor styleAccessor,
        CurveType curve,
        typename CurveType::ArcLengthCalculator arcLenCalc,
        float phaseShift)
    {
        StyleClipper<CurveType, StyleAccessor> ret = { styleAccessor, curve, arcLenCalc, phaseShift };
        return ret;
    }

    float_t2 operator()(float_t t)
    {
        // basicaly 0.0 and 1.0 but with a guardband to discard outside the range
        const float_t minT = 0.0 - 1.0;
        const float_t maxT = 1.0 + 1.0;

        const LineStyle style = lineStyles[styleAccessor.styleIdx];
        const float_t arcLen = arcLenCalc.calcArcLen(t);
        const float_t worldSpaceArcLen = arcLen * float_t(globals.worldToScreenRatio);
        float_t normalizedPlaceInPattern = frac(worldSpaceArcLen * style.reciprocalStipplePatternLen + phaseShift);
        uint32_t patternIdx = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPattern);

        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::infinity; 
        float_t2 ret = float_t2(InvalidT, InvalidT);

        const float_t patternLen = float_t(globals.screenToWorldRatio) / style.reciprocalStipplePatternLen;
        // odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
        const bool notInDrawSection = patternIdx & 0x1;
        
        // TODO[Erfan]: Disable this piece of code after clipping, and comment the reason, that the bezier start and end at 0.0 and 1.0 should be in drawable sections
        float_t minDrawT = 0.0;
        float_t maxDrawT = 1.0;
        {
            float_t normalizedPlaceInPatternBegin = frac(phaseShift);
            uint32_t patternIdxBegin = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternBegin);
            const bool BeginInNonDrawSection = patternIdxBegin & 0x1;

            if (BeginInNonDrawSection)
            {
                float_t diffToRightDrawableSection = (patternIdxBegin == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdxBegin];
                diffToRightDrawableSection -= normalizedPlaceInPatternBegin;
                float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLen;
                const float_t arcLenForT1 = 0.0 + scrSpcOffsetToArcLen1;
                minDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, 0.0);
            }

            const float_t arcLenEnd = arcLenCalc.calcArcLen(1.0);
            const float_t worldSpaceArcLenEnd = arcLenEnd * float_t(globals.worldToScreenRatio);
            float_t normalizedPlaceInPatternEnd = frac(worldSpaceArcLenEnd * style.reciprocalStipplePatternLen + phaseShift);
            uint32_t patternIdxEnd = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternEnd);
            const bool EndInNonDrawSection = patternIdxEnd & 0x1;

            if (EndInNonDrawSection)
            {
                float_t diffToLeftDrawableSection = (patternIdxEnd == 0) ? 0.0 : styleAccessor[patternIdxEnd - 1];
                diffToLeftDrawableSection -= normalizedPlaceInPatternEnd;
                float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLen;
                const float_t arcLenForT0 = arcLenEnd + scrSpcOffsetToArcLen0;
                maxDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, 1.0);
            }
        }

        if (notInDrawSection)
        {
            float_t diffToLeftDrawableSection = (patternIdx == 0) ? 0.0 : styleAccessor[patternIdx - 1];
            diffToLeftDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLen;
            const float_t arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
            float_t t0 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, t);
            t0 = clamp(t0, minDrawT, maxDrawT);

            float_t diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdx];
            diffToRightDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLen;
            const float_t arcLenForT1 = arcLen + scrSpcOffsetToArcLen1;
            float_t t1 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, t);
            t1 = clamp(t1, minDrawT, maxDrawT);

            ret = float_t2(t0, t1);
        }
        else
        {
            t = clamp(t, minDrawT, maxDrawT);
            ret = float_t2(t, t);
        }

        return ret;
    }

    StyleAccessor styleAccessor;
    CurveType curve;
    typename CurveType::ArcLengthCalculator arcLenCalc;
    float phaseShift;
};

template<typename CurveType, typename Clipper = DefaultClipper<typename CurveType::scalar_t> >
struct ClippedSignedDistance
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;

    const static float_t sdf(CurveType curve, float_t2 pos, float_t thickness, bool isRoadStyle, Clipper clipper = DefaultClipper<typename CurveType::scalar_t>::construct())
    {
        typename CurveType::Candidates candidates = curve.getClosestCandidates(pos);

        const float_t InvalidT = 3.402823466e+38F; // TODO: use numeric limits
        const float_t MAX_DISTANCE_SQUARED = (thickness + 1.0f) * (thickness + 1.0f); // TODO: ' + 1' is too much?

        bool clipped = false;
        float_t closestDistanceSquared = MAX_DISTANCE_SQUARED;
        float_t closestT = InvalidT;
        [[unroll(CurveType::MaxCandidates)]]
        for (uint32_t i = 0; i < CurveType::MaxCandidates; i++)
        {
            const float_t candidateDistanceSquared = length(curve.evaluate(candidates[i]) - pos);
            if (candidateDistanceSquared < closestDistanceSquared)
            {
                float_t2 snappedTs = clipper(candidates[i]);

                if (snappedTs[0] == InvalidT)
                {
                    continue;
                }

                if (snappedTs[0] != candidates[i])
                {
                    // left snapped or clamped
                    const float_t leftSnappedCandidateDistanceSquared = length(curve.evaluate(snappedTs[0]) - pos);
                    if (leftSnappedCandidateDistanceSquared < closestDistanceSquared)
                    {
                        clipped = true;
                        closestT = snappedTs[0];
                        closestDistanceSquared = leftSnappedCandidateDistanceSquared;
                    }

                    if (snappedTs[0] != snappedTs[1])
                    {
                        // right snapped or clamped
                        const float_t rightSnappedCandidateDistanceSquared = length(curve.evaluate(snappedTs[1]) - pos);
                        if (rightSnappedCandidateDistanceSquared < closestDistanceSquared)
                        {
                            clipped = true;
                            closestT = snappedTs[1];
                            closestDistanceSquared = rightSnappedCandidateDistanceSquared;
                        }
                    }
                }
                else
                {
                    // no snapping
                    if (candidateDistanceSquared < closestDistanceSquared)
                    {
                        clipped = false;
                        closestT = candidates[i];
                        closestDistanceSquared = candidateDistanceSquared;
                    }
                }
            }
        }


        float_t roundedDistance = closestDistanceSquared - thickness;
        if(!isRoadStyle)
        {
            return roundedDistance;
        }
        else
        {
            const float_t aaWidth = globals.antiAliasingFactor;
            float_t rectCappedDistance = roundedDistance;

            if (clipped)
            {
                float_t2 q = mul(curve.getLocalCoordinateSpace(closestT), pos - curve.evaluate(closestT));
                rectCappedDistance = capSquare(q, thickness, aaWidth);
            }

            return rectCappedDistance;
        }
    }

    static float capSquare(float_t2 q, float_t th, float_t aaWidth)
    {
        float_t2 d = abs(q) - float_t2(aaWidth, th);
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }
};

// sdf of Isosceles Trapezoid y-aligned by https://iquilezles.org/articles/distfunctions2d/
float sdTrapezoid(float2 p, float r1, float r2, float he)
{
    float2 k1 = float2(r2, he);
    float2 k2 = float2(r2 - r1, 2.0 * he);

    p.x = abs(p.x);
    float2 ca = float2(max(0.0, p.x - ((p.y < 0.0) ? r1 : r2)), abs(p.y) - he);
    float2 cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2,k2), 0.0, 1.0);

    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;

    return s * sqrt(min(dot(ca,ca), dot(cb,cb)));
}

float miterSDF(float2 p, float thickness, float2 a, float2 b, float ra, float rb)
{
    float h = length(b - a) / 2.0;
    float2 d = normalize(b - a);
    float2x2 rot = float2x2(d.y, -d.x, d.x, d.y);
    p = mul(rot, p);
    p.y -= h - thickness;
    return sdTrapezoid(p, ra, rb, h);
}

typedef StyleClipper<nbl::hlsl::shapes::Quadratic<float>, StyleAccessor> BezierStyleClipper;
typedef StyleClipper<nbl::hlsl::shapes::Line<float>, StyleAccessor> LineStyleClipper;

float4 main(PSInput input) : SV_TARGET
{
#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
    [[vk::ext_capability(/*FragmentShaderPixelInterlockEXT*/ 5378)]]
    [[vk::ext_extension("SPV_EXT_fragment_shader_interlock")]]
    vk::ext_execution_mode(/*PixelInterlockOrderedEXT*/ 5366);
#endif
	
    ObjectType objType = input.getObjType();
    float localAlpha = 0.0f;
    uint32_t currentMainObjectIdx = input.getMainObjectIdx();

    if (objType == ObjectType::LINE)
    {
        const float2 start = input.getLineStart();
        const float2 end = input.getLineEnd();
        const uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
        const float thickness = input.getLineThickness();
        const bool isRoadStyle = lineStyles[styleIdx].isRoadStyleFlag;

        nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(start, end);
        nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);

        float distance;
        if (!lineStyles[styleIdx].hasStipples())
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, input.position.xy, thickness, isRoadStyle);
        }
        else
        {
            StyleAccessor styleAccessor = { styleIdx };
            LineStyleClipper clipper = LineStyleClipper::construct(styleAccessor, lineSegment, arcLenCalc, input.getCurrentPhaseShift());
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, input.position.xy, thickness, isRoadStyle, clipper);
        }

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::QUAD_BEZIER)
    {
        nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
        nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator arcLenCalc = input.getQuadraticArcLengthCalculator();

        const uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
        const float thickness = input.getLineThickness();
        const bool isRoadStyle = lineStyles[styleIdx].isRoadStyleFlag;
        float distance;

        if (!lineStyles[styleIdx].hasStipples())
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Quadratic<float> >::sdf(quadratic, input.position.xy, thickness, isRoadStyle);
        }
        else
        {
            StyleAccessor styleAccessor = { styleIdx };
            BezierStyleClipper clipper = BezierStyleClipper::construct(styleAccessor, quadratic, arcLenCalc, input.getCurrentPhaseShift());
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Quadratic<float>, BezierStyleClipper>::sdf(quadratic, input.position.xy, thickness, isRoadStyle, clipper);
        }

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::CURVE_BOX) 
    {
        const float minorBBoxUv = input.getMinorBBoxUv();
        const float majorBBoxUv = input.getMajorBBoxUv();

        nbl::hlsl::math::equations::Quadratic<float> curveMinMinor = input.getCurveMinMinor();
        nbl::hlsl::math::equations::Quadratic<float> curveMinMajor = input.getCurveMinMajor();
        nbl::hlsl::math::equations::Quadratic<float> curveMaxMinor = input.getCurveMaxMinor();
        nbl::hlsl::math::equations::Quadratic<float> curveMaxMajor = input.getCurveMaxMajor();

        nbl::hlsl::math::equations::Quadratic<float> minCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMinMajor.a, curveMinMajor.b, curveMinMajor.c - clamp(majorBBoxUv,0.001,0.999));
        nbl::hlsl::math::equations::Quadratic<float> maxCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMaxMajor.a, curveMaxMajor.b, curveMaxMajor.c - clamp(majorBBoxUv,0.001,0.999));

        const float minT = clamp(PrecomputedRootFinder<float>::construct(minCurveEquation).computeRoots(), 0.0, 1.0);
        const float minEv = curveMinMinor.evaluate(minT);
        
        const float maxT = clamp(PrecomputedRootFinder<float>::construct(maxCurveEquation).computeRoots(), 0.0, 1.0);
        const float maxEv = curveMaxMinor.evaluate(maxT);
        
        const float minorDirectionOverScreenSpaceChange = length(float2(ddx(minorBBoxUv),ddy(minorBBoxUv))); // we decided to do this instead of fwidth for dMinor/dScreen
        const float majorDirectionOverScreenSpaceChange = length(float2(ddx(majorBBoxUv),ddy(majorBBoxUv))); // we decided to do this instead of fwidth

        float2 normalMinCurve = float2(
            2.0 * curveMinMajor.a * minT + curveMinMajor.b,
            2.0 * curveMinMinor.a * minT + curveMinMinor.b);
        normalMinCurve = normalize(normalMinCurve * float2(minorDirectionOverScreenSpaceChange, -majorDirectionOverScreenSpaceChange));

        float2 normalMaxCurve = float2(
            2.0 * curveMaxMajor.a * maxT + curveMaxMajor.b,
            2.0 * curveMaxMinor.a * maxT + curveMaxMinor.b);
        normalMaxCurve = normalize(normalMaxCurve * float2(minorDirectionOverScreenSpaceChange, -majorDirectionOverScreenSpaceChange));

        float curveMinorDistance = min(
            normalMinCurve.x * (minorBBoxUv - minEv),
            normalMaxCurve.x * (maxEv - minorBBoxUv));

        const float aabbMajorDistance = min(majorBBoxUv, 1.0 - majorBBoxUv);

        const float antiAliasingFactorMinor = globals.antiAliasingFactor * fwidth(minorBBoxUv);
        const float antiAliasingFactorMajor = globals.antiAliasingFactor * fwidth(majorBBoxUv);

        localAlpha = 1.0;
        localAlpha *= smoothstep(-antiAliasingFactorMajor, 0.0, aabbMajorDistance);
        localAlpha *= smoothstep(-antiAliasingFactorMinor, antiAliasingFactorMinor, curveMinorDistance);
    }
    else if (objType == ObjectType::POLYLINE_CONNECTOR)
    {
        const float2 P = input.position.xy - input.getPolylineConnectorCircleCenter();
        const float distance = miterSDF(
            P,
            input.getLineThickness(),
            input.getPolylineConnectorTrapezoidStart(),
            input.getPolylineConnectorTrapezoidEnd(),
            input.getPolylineConnectorTrapezoidLongBase(),
            input.getPolylineConnectorTrapezoidShortBase());

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }

    uint2 fragCoord = uint2(input.position.xy);
    float4 col;

    if (localAlpha <= 0)
        discard;


#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
        beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];

    const uint32_t localQuantizedAlpha = (uint32_t)(localAlpha * 255.f);
    const uint32_t quantizedAlpha = bitfieldExtract(packedData,0,AlphaBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const uint32_t mainObjectIdx = bitfieldExtract(packedData,AlphaBits,MainObjectIdxBits);
    const bool resolve = currentMainObjectIdx != mainObjectIdx;
    if (resolve || localQuantizedAlpha > quantizedAlpha)
        pseudoStencil[fragCoord] = bitfieldInsert(localQuantizedAlpha,currentMainObjectIdx,AlphaBits,MainObjectIdxBits);

    endInvocationInterlockEXT();

    if (!resolve)
        discard;

    // draw with previous geometry's style's color :kek:
    col = lineStyles[mainObjects[mainObjectIdx].styleIdx].color;
    col.w *= float(quantizedAlpha) / 255.f;
#else
    col = input.getColor();
    col.w *= localAlpha;
#endif
    return float4(col);
}
