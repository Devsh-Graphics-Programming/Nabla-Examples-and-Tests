#pragma shader_stage(fragment)

#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

#if defined(NBL_FEATURE_FRAGMENT_SHADER_PIXEL_INTERLOCK)
[[vk::ext_instruction(/* OpBeginInvocationInterlockEXT */ 5364)]]
void beginInvocationInterlockEXT();
[[vk::ext_instruction(/* OpEndInvocationInterlockEXT */ 5365)]]
void endInvocationInterlockEXT();
#endif

// TODO[Lucas]: have a function for quadratic equation solving
// Write a general one, and maybe another one that uses precomputed values, and move these to somewhere nice in our builtin hlsl shaders if we don't have one
// See: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c

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

struct StyleAccessor
{
    uint32_t styleIdx;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return lineStyles[styleIdx].stipplePattern[ix];
    }
};

template<typename CurveType, typename StyleAccessor>
struct StyleClipper
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.000001;

    // TODO[Przemek]: this should now also include a float phaseShift which is in style's normalized space
    static StyleClipper<CurveType, StyleAccessor> construct(StyleAccessor styleAccessor,
        CurveType curve,
        typename CurveType::ArcLengthCalculator arcLenCalc)
    {
        StyleClipper<CurveType, StyleAccessor> ret = { styleAccessor, curve, arcLenCalc };
        return ret;
    }

    float_t2 operator()(float_t t)
    {
        // basicaly 0.0 and 1.0 but with a guardband to discard outside the range
        const float_t minT = 0.0 - 2.0;
        const float_t maxT = 1.0 + 2.0;

        const LineStyle style = lineStyles[styleAccessor.styleIdx];
        const float_t arcLen = arcLenCalc.calcArcLen(t);
        t = clamp(t, minT, maxT);
        const float_t worldSpaceArcLen = arcLen * float_t(globals.worldToScreenRatio);
        // TODO[Przemek]: apply the phase shift of the curve here as well
        float_t normalizedPlaceInPattern = frac(worldSpaceArcLen * style.reciprocalStipplePatternLen + style.phaseShift);
        uint32_t patternIdx = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPattern);


        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::infinity; 
        float_t2 ret = float_t2(InvalidT, InvalidT);

        // odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
        if (patternIdx & 0x1)
        {
            float_t diffToLeftDrawableSection = (patternIdx == 0) ? 0.0 : style.stipplePattern[patternIdx - 1];
            float_t diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0 : style.stipplePattern[patternIdx];
            diffToLeftDrawableSection -= normalizedPlaceInPattern;
            diffToRightDrawableSection -= normalizedPlaceInPattern;

            const float_t patternLen = float_t(globals.screenToWorldRatio) / style.reciprocalStipplePatternLen;
            float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLen;
            float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLen;

            const float_t arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
            const float_t arcLenForT1 = arcLen + scrSpcOffsetToArcLen1;

            float_t t0 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, t);
            float_t t1 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, t);

            ret = float_t2(t0, t1);

            if (ret[1] > 1.0 - AccuracyThresholdT)
            {
                if (ret[0] > 1.0 - AccuracyThresholdT)
                {
                    // stipplePatternSize is odd by construction (pattern starts with + and ends with -)
                    const bool leftIsDot = 
                        (patternIdx > 1 && style.stipplePattern[patternIdx - 1] == style.stipplePattern[patternIdx - 2]) ||
                        (patternIdx == 1 && style.stipplePattern[0] == 0.0);
                    if (leftIsDot)
                    {
                        ret[0] = InvalidT;
                        ret[1] = InvalidT;
                    }
                    else
                    {
                        ret[0] = 1.0;
                        ret[1] = 1.0;
                    }
                }
                else
                {
                    ret[1] = ret[0];
                }
            }

            if (ret[0] < 0.0 + AccuracyThresholdT)
            {
                if (ret[1] < 0.0 + AccuracyThresholdT)
                {
                    // stipplePatternSize is odd by construction (pattern starts with + and ends with -)
                    const bool rightIsDot =
                        (patternIdx == style.stipplePatternSize && style.stipplePattern[0] == 0.0) ||
                        (patternIdx + 2 <= style.stipplePatternSize && style.stipplePattern[patternIdx] == style.stipplePattern[patternIdx + 1]);

                    if (rightIsDot)
                    {
                        ret[0] = InvalidT;
                        ret[1] = InvalidT;
                    }
                    else
                    {
                        ret[0] = 0.0;
                        ret[1] = 0.0;
                    }
                }
                else
                {
                    ret[0] = ret[1];
                }
            }
        }
        else
        {
            const float_t patternLen = float_t(globals.screenToWorldRatio) / style.reciprocalStipplePatternLen;

            if (t > 1.0 - AccuracyThresholdT)
            {
                float_t diffToLeftDrawableSection = (patternIdx == 0) ? 0.0 : style.stipplePattern[patternIdx - 1];
                diffToLeftDrawableSection -= normalizedPlaceInPattern;
                float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLen;
                const float_t arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
                float_t t0 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, t);
                if (t0 > 1.0 - AccuracyThresholdT)
                {
                    t = InvalidT;
                }
                else
                {
                    t = 1.0;
                }
            }

            if (t < 0.0 + AccuracyThresholdT)
            {
                float_t diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0 : style.stipplePattern[patternIdx];
                diffToRightDrawableSection -= normalizedPlaceInPattern;
                float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLen;
                const float_t arcLenForT1 = arcLen + scrSpcOffsetToArcLen1;
                float_t t1 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, t);
                if (t1 < 0.0 + AccuracyThresholdT)
                {
                    t = InvalidT;
                }
                else
                {
                    t = 0.0;
                }
            }

            ret = float_t2(t, t);
        }

        return ret;
    }

    StyleAccessor styleAccessor;
    CurveType curve;
    typename CurveType::ArcLengthCalculator arcLenCalc;
};

template<typename CurveType, typename Clipper = DefaultClipper<typename CurveType::scalar_t> >
struct ClippedSignedDistance
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;

    const static float_t sdf(CurveType curve, float_t2 pos, float_t thickness, Clipper clipper = DefaultClipper<typename CurveType::scalar_t>::construct())
    {
        typename CurveType::Candidates candidates = curve.getClosestCandidates(pos);

        const float_t InvalidT = 3.402823466e+38F; // TODO: use numeric limits
        const float_t MAX_DISTANCE_SQUARED = (thickness + 1.0f) * (thickness + 1.0f);

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

        // TODO[Przemek]: if style `isRoadStyle` is true use rectCapped, else use normal roundedDistance and remove this #ifdef
#define ROUNDED
#ifdef ROUNDED
        return roundedDistance;
#else
        const float_t aaWidth = globals.antiAliasingFactor;
        float_t rectCappedDistance = roundedDistance;

        if (clipped)
        {
            float_t2 q = mul(curve.getLocalCoordinateSpace(closestT), pos - curve.evaluate(closestT));
            rectCappedDistance = capSquare(q, thickness, aaWidth);
        }
        else
            rectCappedDistance = rectCappedDistance;

        return rectCappedDistance;
#endif
    }

    static float capSquare(float_t2 q, float_t th, float_t aaWidth)
    {
        float_t2 d = abs(q) - float_t2(aaWidth, th);
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }
};

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


    // TODO:[Przemek]: handle another object type POLYLINE_CONNECTOR which is our miters eventually and is and sdf of intersection of 2 or more half-planes

    if (objType == ObjectType::LINE)
    {
        const float2 start = input.getLineStart();
        const float2 end = input.getLineEnd();
        const uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
        const float thickness = input.getLineThickness();

        nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(start, end);
        nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);

        float distance;
        if (!lineStyles[styleIdx].hasStipples())
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, input.position.xy, thickness);
        }
        else
        {
            StyleAccessor styleAccessor = { styleIdx };
            LineStyleClipper clipper = LineStyleClipper::construct(styleAccessor, lineSegment, arcLenCalc);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, input.position.xy, thickness, clipper);
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
        float distance;

        if (!lineStyles[styleIdx].hasStipples())
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Quadratic<float> >::sdf(quadratic, input.position.xy, thickness);
        }
        else
        {
            StyleAccessor styleAccessor = { styleIdx };
            BezierStyleClipper clipper = BezierStyleClipper::construct(styleAccessor, quadratic, arcLenCalc);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Quadratic<float>, BezierStyleClipper>::sdf(quadratic, input.position.xy, thickness, clipper);
        }

        const float antiAliasingFactor = globals.antiAliasingFactor;
        localAlpha = 1.0f - smoothstep(-antiAliasingFactor, +antiAliasingFactor, distance);
    }
    /*
    TODO[Lucas]:
        Another else case for CurveBox where you simply do what I said in the notes of common.hlsl PSInput
        and solve two quadratic equations, you could check for it being a "line" for the mid point being nan
        you will use input.getXXX() to get values needed for this computation
    */

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