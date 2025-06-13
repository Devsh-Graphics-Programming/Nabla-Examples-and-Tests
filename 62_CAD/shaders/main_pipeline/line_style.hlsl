#ifndef _CAD_EXAMPLE_LINE_STYLE_HLSL_INCLUDED_
#define _CAD_EXAMPLE_LINE_STYLE_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

// for usage in upper_bound function
struct StyleAccessor
{
    LineStyle style;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return style.getStippleValue(ix);
    }
};

template<typename CurveType>
struct StyleClipper
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.000001;

    static StyleClipper<CurveType> construct(
        LineStyle style,
        CurveType curve,
        typename CurveType::ArcLengthCalculator arcLenCalc,
        float phaseShift,
        float stretch,
        float worldToScreenRatio)
    {
        StyleClipper<CurveType> ret = { style, curve, arcLenCalc, phaseShift, stretch, worldToScreenRatio, 0.0f, 0.0f, 0.0f, 0.0f };

        // values for non-uniform stretching with a rigid segment
        if (style.rigidSegmentIdx != InvalidRigidSegmentIndex && stretch != 1.0f)
        {
            // rigidSegment info in old non stretched pattern
            ret.rigidSegmentStart = (style.rigidSegmentIdx >= 1u) ? style.getStippleValue(style.rigidSegmentIdx - 1u) : 0.0f;
            ret.rigidSegmentEnd = (style.rigidSegmentIdx < style.stipplePatternSize) ? style.getStippleValue(style.rigidSegmentIdx) : 1.0f;
            ret.rigidSegmentLen = ret.rigidSegmentEnd - ret.rigidSegmentStart;
            // stretch value for non rigid segments
            ret.nonRigidSegmentStretchValue = (stretch - ret.rigidSegmentLen) / (1.0f - ret.rigidSegmentLen);
            // rigidSegment info to new stretched pattern
            ret.rigidSegmentStart *= ret.nonRigidSegmentStretchValue / stretch; // get the new normalized rigid segment start
            ret.rigidSegmentLen /= stretch; // get the new rigid segment normalized len
            ret.rigidSegmentEnd = ret.rigidSegmentStart + ret.rigidSegmentLen; // get the new normalized rigid segment end 
        }
        else
        {
            ret.nonRigidSegmentStretchValue = stretch;
        }

        return ret;
    }

    // For non-uniform stretching with a rigid segment (the one segement that shouldn't stretch) the whole pattern changes
    // instead of transforming each of the style.stipplePattern values (max 14 of them), we transform the normalized place in pattern
    float getRealNormalizedPlaceInPattern(float normalizedPlaceInPattern)
    {
        if (style.rigidSegmentIdx != InvalidRigidSegmentIndex && stretch != 1.0f)
        {
            float ret = min(normalizedPlaceInPattern, rigidSegmentStart) / nonRigidSegmentStretchValue; // unstretch parts before rigid segment
            ret += max(normalizedPlaceInPattern - rigidSegmentEnd, 0.0f) / nonRigidSegmentStretchValue; // unstretch parts after rigid segment
            ret += max(min(rigidSegmentLen, normalizedPlaceInPattern - rigidSegmentStart), 0.0f); // unstretch parts inside rigid segment
            ret *= stretch;
            return ret;
        }
        else
        {
            return normalizedPlaceInPattern;
        }
    }

    float_t2 operator()(float_t t)
    {
        // basicaly 0.0 and 1.0 but with a guardband to discard outside the range
        const float_t minT = 0.0 - 1.0;
        const float_t maxT = 1.0 + 1.0;

        StyleAccessor styleAccessor = { style };
        const float_t reciprocalStretchedStipplePatternLen = style.reciprocalStipplePatternLen / stretch;
        const float_t patternLenInScreenSpace = 1.0 / (worldToScreenRatio * style.reciprocalStipplePatternLen);

        const float_t arcLen = arcLenCalc.calcArcLen(t);
        const float_t worldSpaceArcLen = arcLen * float_t(worldToScreenRatio);
        float_t normalizedPlaceInPattern = frac(worldSpaceArcLen * reciprocalStretchedStipplePatternLen + phaseShift);
        normalizedPlaceInPattern = getRealNormalizedPlaceInPattern(normalizedPlaceInPattern);
        uint32_t patternIdx = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPattern);

        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::infinity;
        float_t2 ret = float_t2(InvalidT, InvalidT);

        // odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
        const bool notInDrawSection = patternIdx & 0x1;

        // TODO[Erfan]: Disable this piece of code after clipping, and comment the reason, that the bezier start and end at 0.0 and 1.0 should be in drawable sections
        float_t minDrawT = 0.0;
        float_t maxDrawT = 1.0;
        {
            float_t normalizedPlaceInPatternBegin = frac(phaseShift);
            normalizedPlaceInPatternBegin = getRealNormalizedPlaceInPattern(normalizedPlaceInPatternBegin);
            uint32_t patternIdxBegin = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternBegin);
            const bool BeginInNonDrawSection = patternIdxBegin & 0x1;

            if (BeginInNonDrawSection)
            {
                float_t diffToRightDrawableSection = (patternIdxBegin == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdxBegin];
                diffToRightDrawableSection -= normalizedPlaceInPatternBegin;
                float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLenInScreenSpace * ((patternIdxBegin != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);
                const float_t arcLenForT1 = 0.0 + scrSpcOffsetToArcLen1;
                minDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, 0.0);
            }

            // Completely in non-draw section -> clip away:
            if (minDrawT >= 1.0)
                return ret;

            const float_t arcLenEnd = arcLenCalc.calcArcLen(1.0);
            const float_t worldSpaceArcLenEnd = arcLenEnd * float_t(worldToScreenRatio);
            float_t normalizedPlaceInPatternEnd = frac(worldSpaceArcLenEnd * reciprocalStretchedStipplePatternLen + phaseShift);
            normalizedPlaceInPatternEnd = getRealNormalizedPlaceInPattern(normalizedPlaceInPatternEnd);
            uint32_t patternIdxEnd = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternEnd);
            const bool EndInNonDrawSection = patternIdxEnd & 0x1;

            if (EndInNonDrawSection)
            {
                float_t diffToLeftDrawableSection = (patternIdxEnd == 0) ? 0.0 : styleAccessor[patternIdxEnd - 1];
                diffToLeftDrawableSection -= normalizedPlaceInPatternEnd;
                float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLenInScreenSpace * ((patternIdxEnd != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);
                const float_t arcLenForT0 = arcLenEnd + scrSpcOffsetToArcLen0;
                maxDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, 1.0);
            }
        }

        if (notInDrawSection)
        {
            float toScreenSpaceLen = patternLenInScreenSpace * ((patternIdx != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);

            float_t diffToLeftDrawableSection = (patternIdx == 0) ? 0.0 : styleAccessor[patternIdx - 1];
            diffToLeftDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * toScreenSpaceLen;
            const float_t arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
            float_t t0 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, t);
            t0 = clamp(t0, minDrawT, maxDrawT);

            float_t diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdx];
            diffToRightDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * toScreenSpaceLen;
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

    LineStyle style;
    CurveType curve;
    typename CurveType::ArcLengthCalculator arcLenCalc;
    float phaseShift;
    float stretch;
    float worldToScreenRatio;
    // precomp value for non uniform stretching
    float rigidSegmentStart;
    float rigidSegmentEnd;
    float rigidSegmentLen;
    float nonRigidSegmentStretchValue;
};

typedef StyleClipper< nbl::hlsl::shapes::Quadratic<float> > BezierStyleClipper;
typedef StyleClipper< nbl::hlsl::shapes::Line<float> > LineStyleClipper;

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

template<typename CurveType, typename Clipper = DefaultClipper<typename CurveType::scalar_t> >
struct ClippedSignedDistance
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;

    const static float_t sdf(CurveType curve, float_t2 pos, float_t thickness, bool isRoadStyle, Clipper clipper = DefaultClipper<typename CurveType::scalar_t>::construct())
    {
        typename CurveType::Candidates candidates = curve.getClosestCandidates(pos);

        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::max;
        // TODO: Fix and test, we're not working with squared distance anymore
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
        if (!isRoadStyle)
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

#endif