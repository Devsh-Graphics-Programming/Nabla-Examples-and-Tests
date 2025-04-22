#ifndef _CAD_EXAMPLE_DTM_HLSL_INCLUDED_
#define _CAD_EXAMPLE_DTM_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>

// TODO: functions outside of the "dtm" namespace need to be moved to another file

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

namespace dtm
{

// for usage in upper_bound function
struct DTMSettingsHeightsAccessor
{
    DTMHeightShadingSettings settings;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return settings.heightColorMapHeights[ix];
    }
};

float dot2(in float2 vec)
{
    return dot(vec, vec);
}

// TODO: Later move these functions and structs to dtmSettings.hlsl and a namespace like dtmSettings::height_shading or dtmSettings::contours, etc..
struct HeightSegmentTransitionData
{
    float currentHeight;
    float4 currentSegmentColor;
    float boundaryHeight;
    float4 otherSegmentColor;
};

// This function interpolates between the current and nearest segment colors based on the
// screen-space distance to the segment boundary. The result is a smoothly blended color
// useful for visualizing discrete height levels without harsh edges.
float4 smoothHeightSegmentTransition(in HeightSegmentTransitionData transitionInfo, in float heightDeriv)
{
    float pxDistanceToNearestSegment = abs((transitionInfo.currentHeight - transitionInfo.boundaryHeight) / heightDeriv);
    float nearestSegmentColorCoverage = smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, pxDistanceToNearestSegment);
    float4 localHeightColor = lerp(transitionInfo.otherSegmentColor, transitionInfo.currentSegmentColor, nearestSegmentColorCoverage);
    return localHeightColor;
}

// Computes the continuous position of a height value within uniform intervals.
// flooring this value will give the interval index
//
// If `isCenteredShading` is true, the intervals are centered around `minHeight`, meaning the
// first interval spans [minHeight - intervalLength / 2.0, minHeight + intervalLength / 2.0].
// Otherwise, intervals are aligned from `minHeight` upward, so the first interval spans
// [minHeight, minHeight + intervalLength].
//
// Parameters:
// - height: The height value to classify.
// - minHeight: The reference starting height for interval calculation.
// - intervalLength: The length of each interval segment.
// - isCenteredShading: Whether to center the shading intervals around minHeight.
//
// Returns:
// - A float representing the continuous position within the interval grid.
float getIntervalPosition(in float height, in float minHeight, in float intervalLength, in bool isCenteredShading)
{
    if (isCenteredShading)
        return ((height - minHeight) / intervalLength + 0.5f);
    else
        return ((height - minHeight) / intervalLength);
}

void getIntervalHeightAndColor(in int intervalIndex, in DTMHeightShadingSettings settings, out float4 outIntervalColor, out float outIntervalHeight)
{
    float minShadingHeight = settings.heightColorMapHeights[0];
    float heightForColor = minShadingHeight + float(intervalIndex) * settings.intervalIndexToHeightMultiplier;

    if (settings.isCenteredShading)
        outIntervalHeight = minShadingHeight + (float(intervalIndex) - 0.5) * settings.intervalLength;
    else
        outIntervalHeight = minShadingHeight + (float(intervalIndex)) * settings.intervalLength;

    DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
    uint32_t upperBoundHeightIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, settings.heightColorEntryCount, heightForColor), settings.heightColorEntryCount - 1u);
    uint32_t lowerBoundHeightIndex = max(upperBoundHeightIndex - 1, 0);

    float upperBoundHeight = settings.heightColorMapHeights[upperBoundHeightIndex];
    float lowerBoundHeight = settings.heightColorMapHeights[lowerBoundHeightIndex];

    float4 upperBoundColor = settings.heightColorMapColors[upperBoundHeightIndex];
    float4 lowerBoundColor = settings.heightColorMapColors[lowerBoundHeightIndex];

    if (upperBoundHeight == lowerBoundHeight)
    {
        outIntervalColor = upperBoundColor;
    }
    else
    {
        float interpolationVal = (heightForColor - lowerBoundHeight) / (upperBoundHeight - lowerBoundHeight);
        outIntervalColor = lerp(lowerBoundColor, upperBoundColor, interpolationVal);
    }
}

float3 calculateDTMTriangleBarycentrics(in float2 v1, in float2 v2, in float2 v3, in float2 p)
{
    float denom = (v2.x - v1.x) * (v3.y - v1.y) - (v3.x - v1.x) * (v2.y - v1.y);
    float u = ((v2.y - v3.y) * (p.x - v3.x) + (v3.x - v2.x) * (p.y - v3.y)) / denom;
    float v = ((v3.y - v1.y) * (p.x - v3.x) + (v1.x - v3.x) * (p.y - v3.y)) / denom;
    float w = 1.0 - u - v;
    return float3(u, v, w);
}

float4 calculateDTMHeightColor(in DTMHeightShadingSettings settings, in float3 v[3], in float heightDeriv, in float2 fragPos, in float height)
{
    float4 outputColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // HEIGHT SHADING
    const uint32_t heightMapSize = settings.heightColorEntryCount;
    float minShadingHeight = settings.heightColorMapHeights[0];
    float maxShadingHeight = settings.heightColorMapHeights[heightMapSize - 1];

    if (heightMapSize > 0)
    {
        // partially based on https://www.shadertoy.com/view/XsXSz4 by Inigo Quilez
        float2 e0 = v[1] - v[0];
        float2 e1 = v[2] - v[1];
        float2 e2 = v[0] - v[2];

        float triangleAreaSign = -sign(e0.x * e2.y - e0.y * e2.x);
        float2 v0 = fragPos - v[0];
        float2 v1 = fragPos - v[1];
        float2 v2 = fragPos - v[2];

        float distanceToLine0 = sqrt(dot2(v0 - e0 * dot(v0, e0) / dot(e0, e0)));
        float distanceToLine1 = sqrt(dot2(v1 - e1 * dot(v1, e1) / dot(e1, e1)));
        float distanceToLine2 = sqrt(dot2(v2 - e2 * dot(v2, e2) / dot(e2, e2)));

        float line0Sdf = distanceToLine0 * triangleAreaSign * sign(v0.x * e0.y - v0.y * e0.x);
        float line1Sdf = distanceToLine1 * triangleAreaSign * sign(v1.x * e1.y - v1.y * e1.x);
        float line2Sdf = distanceToLine2 * triangleAreaSign * sign(v2.x * e2.y - v2.y * e2.x);
        float line3Sdf = (minShadingHeight - height) / heightDeriv;
        float line4Sdf = (height - maxShadingHeight) / heightDeriv;

        float convexPolygonSdf = max(line0Sdf, line1Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line2Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line3Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line4Sdf);

        outputColor.a = 1.0f - smoothstep(0.0f, globals.antiAliasingFactor + globals.antiAliasingFactor, convexPolygonSdf);

        // calculate height color
        E_HEIGHT_SHADING_MODE mode = settings.determineHeightShadingMode();
        if (mode == E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS)
        {
            DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
            int upperBoundIndex = nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, heightMapSize, height);
            int mapIndex = max(upperBoundIndex - 1, 0);
            int mapIndexPrev = max(mapIndex - 1, 0);
            int mapIndexNext = min(mapIndex + 1, heightMapSize - 1);

            // logic explainer: if colorIdx is 0.0 then it means blend with next
            // if color idx is >= length of the colours array then it means it's also > 0.0 and this blend with prev is true
            // if color idx is > 0 and < len - 1, then it depends on the current pixel's height value and two closest height values
            bool blendWithPrev = (mapIndex > 0)
                && (mapIndex >= heightMapSize - 1 || (height * 2.0 < settings.heightColorMapHeights[upperBoundIndex] + settings.heightColorMapHeights[mapIndex]));

            HeightSegmentTransitionData transitionInfo;
            transitionInfo.currentHeight = height;
            transitionInfo.currentSegmentColor = settings.heightColorMapColors[mapIndex];
            transitionInfo.boundaryHeight = blendWithPrev ? settings.heightColorMapHeights[mapIndex] : settings.heightColorMapHeights[mapIndexNext];
            transitionInfo.otherSegmentColor = blendWithPrev ? settings.heightColorMapColors[mapIndexPrev] : settings.heightColorMapColors[mapIndexNext];

            float4 localHeightColor = smoothHeightSegmentTransition(transitionInfo, heightDeriv);
            outputColor.rgb = localHeightColor.rgb;
            outputColor.a *= localHeightColor.a;
        }
        else if (mode == E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS)
        {
            float intervalPosition = getIntervalPosition(height, minShadingHeight, settings.intervalLength, settings.isCenteredShading);
            float positionWithinInterval = frac(intervalPosition);
            int intervalIndex = nbl::hlsl::_static_cast<int>(intervalPosition);

            float4 currentIntervalColor;
            float currentIntervalHeight;
            getIntervalHeightAndColor(intervalIndex, settings, currentIntervalColor, currentIntervalHeight);

            bool blendWithPrev = (positionWithinInterval < 0.5f);

            HeightSegmentTransitionData transitionInfo;
            transitionInfo.currentHeight = height;
            transitionInfo.currentSegmentColor = currentIntervalColor;
            if (blendWithPrev)
            {
                int prevIntervalIdx = max(intervalIndex - 1, 0);
                float prevIntervalHeight; // unused, the currentIntervalHeight is the boundary height between current and prev
                getIntervalHeightAndColor(prevIntervalIdx, settings, transitionInfo.otherSegmentColor, prevIntervalHeight);
                transitionInfo.boundaryHeight = currentIntervalHeight;
            }
            else
            {
                int nextIntervalIdx = intervalIndex + 1;
                getIntervalHeightAndColor(nextIntervalIdx, settings, transitionInfo.otherSegmentColor, transitionInfo.boundaryHeight);
            }

            float4 localHeightColor = smoothHeightSegmentTransition(transitionInfo, heightDeriv);
            outputColor.rgb = localHeightColor.rgb;
            outputColor.a *= localHeightColor.a;
        }
        else if (mode == E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS)
        {
            DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
            uint32_t upperBoundHeightIndex = nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, heightMapSize, height);
            uint32_t lowerBoundHeightIndex = upperBoundHeightIndex == 0 ? upperBoundHeightIndex : upperBoundHeightIndex - 1;

            float upperBoundHeight = settings.heightColorMapHeights[upperBoundHeightIndex];
            float lowerBoundHeight = settings.heightColorMapHeights[lowerBoundHeightIndex];

            float4 upperBoundColor = settings.heightColorMapColors[upperBoundHeightIndex];
            float4 lowerBoundColor = settings.heightColorMapColors[lowerBoundHeightIndex];

            float interpolationVal;
            if (upperBoundHeightIndex == 0)
                interpolationVal = 1.0f;
            else
                interpolationVal = (height - lowerBoundHeight) / (upperBoundHeight - lowerBoundHeight);

            float4 localHeightColor = lerp(lowerBoundColor, upperBoundColor, interpolationVal);

            outputColor.a *= localHeightColor.a;
            outputColor.rgb = localHeightColor.rgb * outputColor.a + outputColor.rgb * (1.0f - outputColor.a);
        }
    }

    return outputColor;
}

float4 calculateDTMContourColor(in DTMContourSettings contourSettings, in float3 v[3], in float2 fragPos, in float height)
{
    float4 outputColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

    LineStyle contourStyle = loadLineStyle(contourSettings.contourLineStyleIdx);
    const float contourThickness = (contourStyle.screenSpaceLineWidth + contourStyle.worldSpaceLineWidth * globals.screenToWorldRatio) * 0.5f;
    float stretch = 1.0f;
    float phaseShift = 0.0f;

    // TODO: move to ubo or push constants
    const float startHeight = contourSettings.contourLinesStartHeight;
    const float endHeight = contourSettings.contourLinesEndHeight;
    const float interval = contourSettings.contourLinesHeightInterval;

    // TODO: can be precomputed
    const int maxContourLineIdx = (endHeight - startHeight) / interval;

    // TODO: it actually can output a negative number, fix
    int contourLineIdx = nbl::hlsl::_static_cast<int>((height - startHeight) / interval + 0.5f);
    contourLineIdx = clamp(contourLineIdx, 0, maxContourLineIdx);
    float contourLineHeight = startHeight + interval * contourLineIdx;

    int contourLinePointsIdx = 0;
    float2 contourLinePoints[2];
    // TODO: case where heights we are looking for are on all three vertices
    for (int i = 0; i < 3; ++i)
    {
        if (contourLinePointsIdx == 2)
            break;

        float3 p0 = v[i];
        float3 p1 = v[(i + 1) % 3];

        if (p1.z < p0.z)
            nbl::hlsl::swap(p0, p1);

        float minHeight = p0.z;
        float maxHeight = p1.z;

        if (height >= minHeight && height <= maxHeight)
        {
            float2 edge = float2(p1.x, p1.y) - float2(p0.x, p0.y);
            float scale = (contourLineHeight - minHeight) / (maxHeight - minHeight);

            contourLinePoints[contourLinePointsIdx] = scale * edge + float2(p0.x, p0.y);
            ++contourLinePointsIdx;
        }
    }

    if (contourLinePointsIdx == 2)
    {
        nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(contourLinePoints[0], contourLinePoints[1]);

        float distance = nbl::hlsl::numeric_limits<float>::max;
        if (!contourStyle.hasStipples() || stretch == InvalidStyleStretchValue)
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, fragPos, contourThickness, contourStyle.isRoadStyleFlag);
        }
        else
        {
            // TODO:
            // It might be beneficial to calculate distance between pixel and contour line to early out some pixels and save yourself from stipple sdf computations!
            // where you only compute the complex sdf if abs((height - contourVal) / heightDeriv) <= aaFactor
            nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
            LineStyleClipper clipper = LineStyleClipper::construct(contourStyle, lineSegment, arcLenCalc, phaseShift, stretch, globals.worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, fragPos, contourThickness, contourStyle.isRoadStyleFlag, clipper);
        }

        outputColor.a = 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, distance);
        outputColor.a *= contourStyle.color.a;
        outputColor.rgb = contourStyle.color.rgb;

        return outputColor;
    }

    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

float4 calculateDTMOutlineColor(in uint outlineLineStyleIdx, in float3 v[3], in float2 fragPos, in float3 baryCoord, in float height)
{
    float4 outputColor;

    LineStyle outlineStyle = loadLineStyle(outlineLineStyleIdx);
    const float outlineThickness = (outlineStyle.screenSpaceLineWidth + outlineStyle.worldSpaceLineWidth * globals.screenToWorldRatio) * 0.5f;
    const float phaseShift = 0.0f; // input.getCurrentPhaseShift();
    const float stretch = 1.0f;

    // index of vertex opposing an edge, needed for calculation of triangle heights
    uint opposingVertexIdx[3];
    opposingVertexIdx[0] = 2;
    opposingVertexIdx[1] = 0;
    opposingVertexIdx[2] = 1;

    float minDistance = nbl::hlsl::numeric_limits<float>::max;
    if (!outlineStyle.hasStipples() || stretch == InvalidStyleStretchValue)
    {
        for (int i = 0; i < 3; ++i)
        {
            float3 p0 = v[i];
            float3 p1 = v[(i + 1) % 3];

            float distance = nbl::hlsl::numeric_limits<float>::max;
            nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(float2(p0.x, p0.y), float2(p1.x, p1.y));
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, fragPos, outlineThickness, outlineStyle.isRoadStyleFlag);

            minDistance = min(minDistance, distance);
        }
    }
    else
    {
        for (int i = 0; i < 3; ++i)
        {
            float3 p0 = v[i];
            float3 p1 = v[(i + 1) % 3];

            // long story short, in order for stipple patterns to be consistent:
            // - point with lesser x coord should be starting point
            // - if x coord of both points are equal then point with lesser y value should be starting point
            if (p1.x < p0.x)
                nbl::hlsl::swap(p0, p1);
            else if (p1.x == p0.x && p1.y < p0.y)
                nbl::hlsl::swap(p0, p1);

            nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(float2(p0.x, p0.y), float2(p1.x, p1.y));

            float distance = nbl::hlsl::numeric_limits<float>::max;
            nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
            LineStyleClipper clipper = LineStyleClipper::construct(outlineStyle, lineSegment, arcLenCalc, phaseShift, stretch, globals.worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, fragPos, outlineThickness, outlineStyle.isRoadStyleFlag, clipper);

            minDistance = min(minDistance, distance);
        }

    }

    outputColor.a = 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, minDistance);
    outputColor.a *= outlineStyle.color.a;
    outputColor.rgb = outlineStyle.color.rgb;

    return outputColor;
}

float4 blendUnder(in float4 srcColor, in float4 dstColor)
{
    dstColor.rgb = dstColor.rgb * dstColor.a + (1 - dstColor.a) * srcColor.a * srcColor.rgb;
    dstColor.a = (1.0f - srcColor.a) * dstColor.a + srcColor.a;

    return dstColor;
}
}

#endif