#define FRAGMENT_SHADER_INPUT
#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>
#include <nbl/builtin/hlsl/text_rendering/msdf.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_barycentric.hlsl>

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

// line segment sdf which returns the distance vector specialized for usage in hatch box line boundaries
float2 sdLineDstVec(float2 P, float2 A, float2 B)
{
    const float2 PA = P - A;
    const float2 BA = B - A;
    float h = clamp(dot(PA, BA) / dot(BA, BA), 0.0, 1.0);
    return PA - BA * h;
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

typedef StyleClipper< nbl::hlsl::shapes::Quadratic<float> > BezierStyleClipper;
typedef StyleClipper< nbl::hlsl::shapes::Line<float> > LineStyleClipper;

// for usage in upper_bound function
struct DTMSettingsHeightsAccessor
{
    DTMSettings dtmSettings;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return dtmSettings.heightColorMapHeights[ix];
    }
};

// We need to specialize color calculation based on FragmentShaderInterlock feature availability for our transparency algorithm
// because there is no `if constexpr` in hlsl
// @params
// textureColor: color sampled from a texture
// useStyleColor: instead of writing and reading from colorStorage, use main object Idx to find the style color for the object.
template<bool FragmentShaderPixelInterlock>
float32_t4 calculateFinalColor(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 textureColor, bool colorFromTexture);

template<>
float32_t4 calculateFinalColor<false>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor, bool colorFromTexture)
{
    uint32_t styleIdx = loadMainObject(currentMainObjectIdx).styleIdx;
    if (!colorFromTexture)
    {
        float32_t4 col = loadLineStyle(styleIdx).color;
        col.w *= localAlpha;
        return float4(col);
    }
    else
        return float4(localTextureColor, localAlpha);
}
template<>
float32_t4 calculateFinalColor<true>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor, bool colorFromTexture)
{
    float32_t4 color;
    nbl::hlsl::spirv::beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];

    const uint32_t localQuantizedAlpha = (uint32_t)(localAlpha * 255.f);
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const bool differentMainObject = currentMainObjectIdx != storedMainObjectIdx; // meaning current pixel's main object is different than what is already stored
    const bool resolve = differentMainObject && storedMainObjectIdx != InvalidMainObjectIdx;
    uint32_t toResolveStyleIdx = InvalidStyleIdx;
    
    // load from colorStorage only if we want to resolve color from texture instead of style
    // sampling from colorStorage needs to happen in critical section because another fragment may also want to store into it at the same time + need to happen before store
    if (resolve)
    {
        toResolveStyleIdx = loadMainObject(storedMainObjectIdx).styleIdx;
        if (toResolveStyleIdx == InvalidStyleIdx) // if style idx to resolve is invalid, then it means we should resolve from color
            color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);
    }
    
    // If current localAlpha is higher than what is already stored in pseudoStencil we will update the value in pseudoStencil or the color in colorStorage, this is equivalent to programmable blending MAX operation.
    // OR If previous pixel has a different ID than current's  (i.e. previous either empty/invalid or a differnet mainObject), we should update our alpha and color storages.
    if (differentMainObject || localQuantizedAlpha > storedQuantizedAlpha)
    {
        pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(localQuantizedAlpha,currentMainObjectIdx,AlphaBits,MainObjectIdxBits);
        if (colorFromTexture) // writing color from texture
            colorStorage[fragCoord] = packR11G11B10_UNORM(localTextureColor);
    }
    
    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (!resolve)
        discard;

    // draw with previous geometry's style's color or stored in texture buffer :kek:
    // we don't need to load the style's color in critical section because we've already retrieved the style index from the stored main obj
    if (toResolveStyleIdx != InvalidStyleIdx) // if toResolveStyleIdx is valid then that means our resolved color should come from line style
        color = loadLineStyle(toResolveStyleIdx).color;
    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}

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

// NOTE[Erfan to Przemek][REMOVE WHEN READ]: I renamed to `smoothHeightSegmentTransition` and made it return value instead of take `out` param + removed applying it to final output color (it's responsibility of the caller now)
// Now the resposibility of this  function is just to "Figure out what the interpolated color at the transition is." and doesn't assume how it's gonna be applied to the final color
// that's more predictible and atomic. Additionally I think `out` functions make the code a little bit more unreadable as well

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
        return ( (height - minHeight) / intervalLength + 0.5f);
    else
        return ( (height - minHeight) / intervalLength );
}

void getIntervalHeightAndColor(in int intervalIndex, in DTMSettings dtmSettings, out float4 outIntervalColor, out float outIntervalHeight)
{
    float minShadingHeight = dtmSettings.heightColorMapHeights[0];
    outIntervalHeight = minShadingHeight + float(intervalIndex) * dtmSettings.intervalIndexToHeightMultiplier;

    DTMSettingsHeightsAccessor dtmHeightsAccessor = { dtmSettings };
    uint32_t upperBoundHeightIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, dtmSettings.heightColorEntryCount, outIntervalHeight), dtmSettings.heightColorEntryCount-1u);
    uint32_t lowerBoundHeightIndex = max(upperBoundHeightIndex - 1, 0);

    float upperBoundHeight = dtmSettings.heightColorMapHeights[upperBoundHeightIndex];
    float lowerBoundHeight = dtmSettings.heightColorMapHeights[lowerBoundHeightIndex];

    float4 upperBoundColor = dtmSettings.heightColorMapColors[upperBoundHeightIndex];
    float4 lowerBoundColor = dtmSettings.heightColorMapColors[lowerBoundHeightIndex];
    
    if (upperBoundHeight == lowerBoundHeight)
    {
        outIntervalColor = upperBoundColor;
    }
    else
    {
        float interpolationVal = (outIntervalHeight - lowerBoundHeight) / (upperBoundHeight - lowerBoundHeight);
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

float4 calculateDTMHeightColor(in DTMSettings dtmSettings, in float3 v[3], in float heightDeriv, in float2 fragPos, in float height)
{
    float4 outputColor = float4(0.0f, 0.0f, 0.0f, 1.0f);

    // HEIGHT SHADING
    const uint32_t heightMapSize = dtmSettings.heightColorEntryCount;
    float minShadingHeight = dtmSettings.heightColorMapHeights[0];
    float maxShadingHeight = dtmSettings.heightColorMapHeights[heightMapSize - 1];

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

        float line0Sdf = distanceToLine0 * triangleAreaSign * (v0.x * e0.y - v0.y * e0.x);
        float line1Sdf = distanceToLine1 * triangleAreaSign * (v1.x * e1.y - v1.y * e1.x);
        float line2Sdf = distanceToLine2 * triangleAreaSign * (v2.x * e2.y - v2.y * e2.x);
        float line3Sdf = (minShadingHeight - height) / heightDeriv;
        float line4Sdf = (height - maxShadingHeight) / heightDeriv;

        float convexPolygonSdf = max(line0Sdf, line1Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line2Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line3Sdf);
        convexPolygonSdf = max(convexPolygonSdf, line4Sdf);

        // TODO: separate
        outputColor.a = 1.0f - smoothstep(0.0f, globals.antiAliasingFactor * 2.0f, convexPolygonSdf);

        // calculate height color
        E_HEIGHT_SHADING_MODE mode = dtmSettings.determineHeightShadingMode();

        if (mode == E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS)
        {
            DTMSettingsHeightsAccessor dtmHeightsAccessor = { dtmSettings };
            int upperBoundIndex = nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, heightMapSize, height);
            int mapIndex = max(upperBoundIndex - 1, 0);
            int mapIndexPrev = max(mapIndex - 1, 0);
            int mapIndexNext = min(mapIndex + 1, heightMapSize - 1);

            // logic explainer: if colorIdx is 0.0 then it means blend with next
            // if color idx is >= length of the colours array then it means it's also > 0.0 and this blend with prev is true
            // if color idx is > 0 and < len - 1, then it depends on the current pixel's height value and two closest height values
            bool blendWithPrev = (mapIndex > 0)
                && (mapIndex >= heightMapSize - 1 || (height * 2.0 < dtmSettings.heightColorMapHeights[upperBoundIndex] + dtmSettings.heightColorMapHeights[mapIndex]));

            HeightSegmentTransitionData transitionInfo;
            transitionInfo.currentHeight = height;
            transitionInfo.currentSegmentColor = dtmSettings.heightColorMapColors[mapIndex];
            transitionInfo.boundaryHeight = blendWithPrev ? dtmSettings.heightColorMapHeights[mapIndex] : dtmSettings.heightColorMapHeights[mapIndexNext];
            transitionInfo.otherSegmentColor = blendWithPrev ? dtmSettings.heightColorMapColors[mapIndexPrev] : dtmSettings.heightColorMapColors[mapIndexNext];

            float4 localHeightColor = smoothHeightSegmentTransition(transitionInfo, heightDeriv);
            outputColor.rgb = localHeightColor.rgb;
            outputColor.a *= localHeightColor.a;
        }
        else if (mode == E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS)
        {
            float intervalPosition = getIntervalPosition(height, minShadingHeight, dtmSettings.intervalLength, dtmSettings.isCenteredShading);
            float positionWithinInterval = frac(intervalPosition);
            int intervalIndex = nbl::hlsl::_static_cast<int>(intervalPosition);

            float4 currentIntervalColor;
            float currentIntervalHeight;
            getIntervalHeightAndColor(intervalIndex, dtmSettings, currentIntervalColor, currentIntervalHeight);
            
            bool blendWithPrev = (positionWithinInterval < 0.5f);
            
            HeightSegmentTransitionData transitionInfo;
            transitionInfo.currentHeight = height;
            transitionInfo.currentSegmentColor = currentIntervalColor;
            if (blendWithPrev)
            {
                int prevIntervalIdx = max(intervalIndex - 1, 0);
                float prevIntervalHeight; // unused, the currentIntervalHeight is the boundary height between current and prev
                getIntervalHeightAndColor(prevIntervalIdx, dtmSettings, transitionInfo.otherSegmentColor, prevIntervalHeight);
                transitionInfo.boundaryHeight = currentIntervalHeight;
            }
            else
            {
                int nextIntervalIdx = intervalIndex + 1;
                getIntervalHeightAndColor(nextIntervalIdx, dtmSettings, transitionInfo.otherSegmentColor, transitionInfo.boundaryHeight);
            }
            
            float4 localHeightColor = smoothHeightSegmentTransition(transitionInfo, heightDeriv);
            outputColor.rgb = localHeightColor.rgb;
            outputColor.a *= localHeightColor.a;
        }
        else if (mode == E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS)
        {
            DTMSettingsHeightsAccessor dtmHeightsAccessor = { dtmSettings };
            uint32_t upperBoundHeightIndex = nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, heightMapSize, height);
            uint32_t lowerBoundHeightIndex = upperBoundHeightIndex == 0 ? upperBoundHeightIndex : upperBoundHeightIndex - 1;

            float upperBoundHeight = dtmSettings.heightColorMapHeights[upperBoundHeightIndex];
            float lowerBoundHeight = dtmSettings.heightColorMapHeights[lowerBoundHeightIndex];

            float4 upperBoundColor = dtmSettings.heightColorMapColors[upperBoundHeightIndex];
            float4 lowerBoundColor = dtmSettings.heightColorMapColors[lowerBoundHeightIndex];

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

float4 calculateDTMContourColor(in DTMSettings dtmSettings, in float3 v[3], in uint2 edgePoints[3], in PSInput psInput, in float height)
{
    float4 outputColor;

    LineStyle contourStyle = loadLineStyle(dtmSettings.contourLineStyleIdx);
    const float contourThickness = psInput.getContourLineThickness();
    float stretch = 1.0f;
    float phaseShift = 0.0f;
    const float worldToScreenRatio = psInput.getCurrentWorldToScreenRatio();

    // TODO: move to ubo or push constants
    const float startHeight = dtmSettings.contourLinesStartHeight;
    const float endHeight = dtmSettings.contourLinesEndHeight;
    const float interval = dtmSettings.contourLinesHeightInterval;

    // TODO: can be precomputed
    const int maxContourLineIdx = (endHeight - startHeight + 1) / interval;

    // TODO: it actually can output a negative number, fix
    int contourLineIdx = nbl::hlsl::_static_cast<int>((height - startHeight + (interval * 0.5f)) / interval);
    contourLineIdx = clamp(contourLineIdx, 0, maxContourLineIdx);
    float contourLineHeight = startHeight + interval * contourLineIdx;

    int contourLinePointsIdx = 0;
    float2 contourLinePoints[2];
    // TODO: case where heights we are looking for are on all three vertices
    for (int i = 0; i < 3; ++i)
    {
        if (contourLinePointsIdx == 2)
            break;

        const uint2 currentEdgePoints = edgePoints[i];
        float3 p0 = v[currentEdgePoints[0]];
        float3 p1 = v[currentEdgePoints[1]];

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

    {
        nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(contourLinePoints[0], contourLinePoints[1]);

        float distance = nbl::hlsl::numeric_limits<float>::max;
        if (!contourStyle.hasStipples() || stretch == InvalidStyleStretchValue)
        {
            distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, psInput.position.xy, contourThickness, contourStyle.isRoadStyleFlag);
        }
        else
        {
            // TODO:
            // It might be beneficial to calculate distance between pixel and contour line to early out some pixels and save yourself from stipple sdf computations!
            // where you only compute the complex sdf if abs((height - contourVal) / heightDeriv) <= aaFactor
            nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
            LineStyleClipper clipper = LineStyleClipper::construct(contourStyle, lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, psInput.position.xy, contourThickness, contourStyle.isRoadStyleFlag, clipper);
        }
        
        outputColor.a = smoothstep(+globals.antiAliasingFactor, -globals.antiAliasingFactor, distance) * contourStyle.color.a;
        outputColor.rgb = contourStyle.color.rgb;
    }

    return outputColor;
}

float4 calculateDTMOutlineColor(in DTMSettings dtmSettings, in float3 v[3], in uint2 edgePoints[3], in PSInput psInput, in float3 baryCoord, in float height)
{
    float4 outputColor;

    LineStyle outlineStyle = loadLineStyle(dtmSettings.outlineLineStyleIdx);
    const float outlineThickness = psInput.getOutlineThickness();
    const float phaseShift = 0.0f; // input.getCurrentPhaseShift();
    const float worldToScreenRatio = psInput.getCurrentWorldToScreenRatio();
    const float stretch = 1.0f;

    // index of vertex opposing an edge, needed for calculation of triangle heights
    uint opposingVertexIdx[3];
    opposingVertexIdx[0] = 2;
    opposingVertexIdx[1] = 0;
    opposingVertexIdx[2] = 1;

    // find sdf of every edge
    float triangleAreaTimesTwo;
    {
        float3 AB = v[0] - v[1];
        float3 AC = v[0] - v[2];
        AB.z = 0.0f;
        AC.z = 0.0f;

        // TODO: figure out if there is a faster solution
        triangleAreaTimesTwo = length(cross(AB, AC));
    }

    // calculate sdf of every edge as it wasn't stippled
    float distances[3];
    for (int i = 0; i < 3; ++i)
    {
        const uint2 currentEdgePoints = edgePoints[i];
        float3 A = v[currentEdgePoints[0]];
        float3 B = v[currentEdgePoints[1]];
        float3 AB = B - A;
        float ABLen = length(AB);
        float triangleHeightToOpositeVertex = triangleAreaTimesTwo / ABLen;

        distances[i] = triangleHeightToOpositeVertex * baryCoord[opposingVertexIdx[i]];
    }

    float minDistance = nbl::hlsl::numeric_limits<float>::max;
    if (!outlineStyle.hasStipples() || stretch == InvalidStyleStretchValue)
    {
        for (uint i = 0; i < 3; ++i)
            distances[i] -= outlineThickness;

        minDistance = min(distances[0], min(distances[1], distances[2]));
    }
    else
    {
        for (int i = 0; i < 3; ++i)
        {
            if (distances[i] > outlineThickness)
                continue;

            const uint2 currentEdgePoints = edgePoints[i];
            float3 p0 = v[currentEdgePoints[0]];
            float3 p1 = v[currentEdgePoints[1]];

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
            LineStyleClipper clipper = LineStyleClipper::construct(outlineStyle, lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, psInput.position.xy, outlineThickness, outlineStyle.isRoadStyleFlag, clipper);

            minDistance = min(minDistance, distance);
        }

    }

    outputColor.a = smoothstep(+globals.antiAliasingFactor, -globals.antiAliasingFactor, minDistance) * outlineStyle.color.a;
    outputColor.rgb = outlineStyle.color.rgb;

    return outputColor;
}

struct DTMColorBlender
{
    void init()
    {
        colorCount = 0;
    }

    void addColorOnTop(in float4 color)
    {
        colors[colorCount] = color;
        colorCount++;
    }

    float4 blend()
    {
        if (colorCount == 0)
            return float4(0.0f, 0.0f, 0.0f, 1.0f);

        float4 outputColor = colors[0];
        for (int i = 1; i < colorCount; ++i)
        {
            outputColor.rgb = colors[i].rgb * colors[i].a + outputColor.rgb * outputColor.a * (1.0f - colors[i].a);
            outputColor.a = colors[i].a + outputColor.a * (1.0f - colors[i].a);
        }

        return outputColor;
    }

    int colorCount;
    float4 colors[3];
};

[[vk::spvexecutionmode(spv::ExecutionModePixelInterlockOrderedEXT)]]
[shader("pixel")]
float4 fragMain(PSInput input) : SV_TARGET
{
    float localAlpha = 0.0f;
    float3 textureColor = float3(0, 0, 0); // color sampled from a texture

    ObjectType objType = input.getObjType();
    const uint32_t currentMainObjectIdx = input.getMainObjectIdx();
    const MainObject mainObj = loadMainObject(currentMainObjectIdx);
    
    if (pc.isDTMRendering)
    {   
        DTMSettings dtmSettings = loadDTMSettings(mainObj.dtmSettingsIdx);

        float3 v[3];
        v[0] = input.getScreenSpaceVertexAttribs(0);
        v[1] = input.getScreenSpaceVertexAttribs(1);
        v[2] = input.getScreenSpaceVertexAttribs(2);

        // indices of points constructing every edge
        uint2 edgePoints[3];
        edgePoints[0] = uint2(0, 1);
        edgePoints[1] = uint2(1, 2);
        edgePoints[2] = uint2(2, 0);

        const float3 baryCoord = calculateDTMTriangleBarycentrics(v[0], v[1], v[2], input.position.xy);
        float height = baryCoord.x * v[0].z + baryCoord.y * v[1].z + baryCoord.z * v[2].z;
        float heightDeriv = fwidth(height);

        DTMColorBlender blender;
        blender.init();
        if(dtmSettings.drawHeightShadingEnabled())
            blender.addColorOnTop(calculateDTMHeightColor(dtmSettings, v, heightDeriv, input.position.xy, height));
        if (dtmSettings.drawContourEnabled())
            blender.addColorOnTop(calculateDTMContourColor(dtmSettings, v, edgePoints, input, height));
        if (dtmSettings.drawOutlineEnabled())
            blender.addColorOnTop(calculateDTMOutlineColor(dtmSettings, v, edgePoints, input, baryCoord, height));
        float4 dtmColor = blender.blend();

        textureColor = dtmColor.rgb;
        localAlpha = dtmColor.a;

        return calculateFinalColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(uint2(input.position.xy), localAlpha, currentMainObjectIdx, textureColor, true);
    }
    else
    {
        // figure out local alpha with sdf
        if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
        {
            float distance = nbl::hlsl::numeric_limits<float>::max;
            if (objType == ObjectType::LINE)
            {
                const float2 start = input.getLineStart();
                const float2 end = input.getLineEnd();
                const uint32_t styleIdx = mainObj.styleIdx;
                const float thickness = input.getLineThickness();
                const float phaseShift = input.getCurrentPhaseShift();
                const float stretch = input.getPatternStretch();
                const float worldToScreenRatio = input.getCurrentWorldToScreenRatio();

                nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(start, end);

                LineStyle style = loadLineStyle(styleIdx);

                if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
                {
                    distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag);
                }
                else
                {
                    nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
                    LineStyleClipper clipper = LineStyleClipper::construct(loadLineStyle(styleIdx), lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
                    distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
                }
            }
            else if (objType == ObjectType::QUAD_BEZIER)
            {
                nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
                nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator arcLenCalc = input.getQuadraticArcLengthCalculator();

                const uint32_t styleIdx = mainObj.styleIdx;
                const float thickness = input.getLineThickness();
                const float phaseShift = input.getCurrentPhaseShift();
                const float stretch = input.getPatternStretch();
                const float worldToScreenRatio = input.getCurrentWorldToScreenRatio();

                LineStyle style = loadLineStyle(styleIdx);
                if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
                {
                    distance = ClippedSignedDistance< nbl::hlsl::shapes::Quadratic<float> >::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag);
                }
                else
                {
                    BezierStyleClipper clipper = BezierStyleClipper::construct(loadLineStyle(styleIdx), quadratic, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
                    distance = ClippedSignedDistance<nbl::hlsl::shapes::Quadratic<float>, BezierStyleClipper>::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
                }
            }
            else if (objType == ObjectType::POLYLINE_CONNECTOR)
            {
                const float2 P = input.position.xy - input.getPolylineConnectorCircleCenter();
                distance = miterSDF(
                    P,
                    input.getLineThickness(),
                    input.getPolylineConnectorTrapezoidStart(),
                    input.getPolylineConnectorTrapezoidEnd(),
                    input.getPolylineConnectorTrapezoidLongBase(),
                    input.getPolylineConnectorTrapezoidShortBase());

            }
            localAlpha = smoothstep(+globals.antiAliasingFactor, -globals.antiAliasingFactor, distance);
        }
        else if (objType == ObjectType::CURVE_BOX) 
        {
            const float minorBBoxUV = input.getMinorBBoxUV();
            const float majorBBoxUV = input.getMajorBBoxUV();

            nbl::hlsl::math::equations::Quadratic<float> curveMinMinor = input.getCurveMinMinor();
            nbl::hlsl::math::equations::Quadratic<float> curveMinMajor = input.getCurveMinMajor();
            nbl::hlsl::math::equations::Quadratic<float> curveMaxMinor = input.getCurveMaxMinor();
            nbl::hlsl::math::equations::Quadratic<float> curveMaxMajor = input.getCurveMaxMajor();

            //  TODO(Optimization): Can we ignore this majorBBoxUV clamp and rely on the t clamp that happens next? then we can pass `PrecomputedRootFinder`s instead of computing the values per pixel.
            nbl::hlsl::math::equations::Quadratic<float> minCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMinMajor.a, curveMinMajor.b, curveMinMajor.c - clamp(majorBBoxUV, 0.0, 1.0));
            nbl::hlsl::math::equations::Quadratic<float> maxCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMaxMajor.a, curveMaxMajor.b, curveMaxMajor.c - clamp(majorBBoxUV, 0.0, 1.0));

            const float minT = clamp(PrecomputedRootFinder<float>::construct(minCurveEquation).computeRoots(), 0.0, 1.0);
            const float minEv = curveMinMinor.evaluate(minT);

            const float maxT = clamp(PrecomputedRootFinder<float>::construct(maxCurveEquation).computeRoots(), 0.0, 1.0);
            const float maxEv = curveMaxMinor.evaluate(maxT);

            const bool insideMajor = majorBBoxUV >= 0.0 && majorBBoxUV <= 1.0;
            const bool insideMinor = minorBBoxUV >= minEv && minorBBoxUV <= maxEv;

            if (insideMinor && insideMajor)
            {
                localAlpha = 1.0;
            }
            else
            {
                // Find the true SDF of a hatch box boundary which is bounded by two curves, It requires knowing the distance from the current UV to the closest point on bounding curves and the limiting lines (in major direction)
                // We also keep track of distance vector (minor, major) to convert to screenspace distance for anti-aliasing with screenspace aaFactor
                const float InvalidT = nbl::hlsl::numeric_limits<float32_t>::max;
                const float MAX_DISTANCE_SQUARED = nbl::hlsl::numeric_limits<float32_t>::max;

                const float2 boxScreenSpaceSize = input.getCurveBoxScreenSpaceSize();


                float closestDistanceSquared = MAX_DISTANCE_SQUARED;
                const float2 pos = float2(minorBBoxUV, majorBBoxUV) * boxScreenSpaceSize;

                if (minorBBoxUV < minEv)
                {
                    // DO SDF of Min Curve
                    nbl::hlsl::shapes::Quadratic<float> minCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                        float2(curveMinMinor.a, curveMinMajor.a) * boxScreenSpaceSize,
                        float2(curveMinMinor.b, curveMinMajor.b) * boxScreenSpaceSize,
                        float2(curveMinMinor.c, curveMinMajor.c) * boxScreenSpaceSize);

                    nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = minCurve.getClosestCandidates(pos);
                    [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                    for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                    {
                        candidates[i] = clamp(candidates[i], 0.0, 1.0);
                        const float2 distVector = minCurve.evaluate(candidates[i]) - pos;
                        const float candidateDistanceSquared = dot(distVector, distVector);
                        if (candidateDistanceSquared < closestDistanceSquared)
                            closestDistanceSquared = candidateDistanceSquared;
                    }
                }
                else if (minorBBoxUV > maxEv)
                {
                    // Do SDF of Max Curve
                    nbl::hlsl::shapes::Quadratic<float> maxCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                        float2(curveMaxMinor.a, curveMaxMajor.a) * boxScreenSpaceSize,
                        float2(curveMaxMinor.b, curveMaxMajor.b) * boxScreenSpaceSize,
                        float2(curveMaxMinor.c, curveMaxMajor.c) * boxScreenSpaceSize);
                    nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = maxCurve.getClosestCandidates(pos);
                    [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                    for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                    {
                        candidates[i] = clamp(candidates[i], 0.0, 1.0);
                        const float2 distVector = maxCurve.evaluate(candidates[i]) - pos;
                        const float candidateDistanceSquared = dot(distVector, distVector);
                        if (candidateDistanceSquared < closestDistanceSquared)
                            closestDistanceSquared = candidateDistanceSquared;
                    }
                }

                if (!insideMajor)
                {
                    const bool minLessThanMax = minEv < maxEv;
                    float2 majorDistVector = float2(MAX_DISTANCE_SQUARED, MAX_DISTANCE_SQUARED);
                    if (majorBBoxUV > 1.0)
                    {
                        const float2 minCurveEnd = float2(minEv, 1.0) * boxScreenSpaceSize;
                        if (minLessThanMax)
                            majorDistVector = sdLineDstVec(pos, minCurveEnd, float2(maxEv, 1.0) * boxScreenSpaceSize);
                        else
                            majorDistVector = pos - minCurveEnd;
                    }
                    else
                    {
                        const float2 minCurveStart = float2(minEv, 0.0) * boxScreenSpaceSize;
                        if (minLessThanMax)
                            majorDistVector = sdLineDstVec(pos, minCurveStart, float2(maxEv, 0.0) * boxScreenSpaceSize);
                        else
                            majorDistVector = pos - minCurveStart;
                    }

                    const float majorDistSq = dot(majorDistVector, majorDistVector);
                    if (majorDistSq < closestDistanceSquared)
                        closestDistanceSquared = majorDistSq;
                }

                const float dist = sqrt(closestDistanceSquared);
                localAlpha = 1.0f - smoothstep(0.0, globals.antiAliasingFactor, dist);
            }

            LineStyle style = loadLineStyle(mainObj.styleIdx);
            uint32_t textureId = asuint(style.screenSpaceLineWidth);
            if (textureId != InvalidTextureIdx)
            {
                // For Hatch fiils we sample the first mip as we don't fill the others, because they are constant in screenspace and render as expected
                // If later on we decided that we can have different sizes here, we should do computations similar to FONT_GLYPH
                float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(frac(input.position.xy / HatchFillMSDFSceenSpaceSize), float(textureId)), 0.0).xyz;
                float msdf = nbl::hlsl::text::msdfDistance(msdfSample, MSDFPixelRange * HatchFillMSDFSceenSpaceSize / MSDFSize);
                localAlpha *= smoothstep(+globals.antiAliasingFactor / 2.0, -globals.antiAliasingFactor / 2.0f, msdf);
            }
        }
        else if (objType == ObjectType::FONT_GLYPH) 
        {
            const float2 uv = input.getFontGlyphUV();
            const uint32_t textureId = input.getFontGlyphTextureId();

            if (textureId != InvalidTextureIdx)
            {
                float mipLevel = msdfTextures.CalculateLevelOfDetail(msdfSampler, uv);
                float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(uv, float(textureId)), mipLevel);
                float msdf = nbl::hlsl::text::msdfDistance(msdfSample, input.getFontGlyphPxRange());
                /*
                    explaining "*= exp2(max(mipLevel,0.0))"
                    Each mip level has constant MSDFPixelRange
                    Which essentially makes the msdfSamples here (Harware Sampled) have different scales per mip
                    As we go up 1 mip level, the msdf distance should be multiplied by 2.0
                    While this makes total sense for NEAREST mip sampling when mipLevel is an integer and only one mip is being sampled.
                    It's a bit complex when it comes to trilinear filtering (LINEAR mip sampling), but it works in practice!
                
                    Alternatively you can think of it as doing this instead:
                    localAlpha = smoothstep(+globals.antiAliasingFactor / exp2(max(mipLevel,0.0)), 0.0, msdf);
                    Which is reducing the aa feathering as we go up the mip levels. 
                    to avoid aa feathering of the MAX_MSDF_DISTANCE_VALUE to be less than aa factor and eventually color it and cause greyed out area around the main glyph
                */
                msdf *= exp2(max(mipLevel,0.0));
            
                LineStyle style = loadLineStyle(mainObj.styleIdx);
                const float screenPxRange = input.getFontGlyphPxRange() / MSDFPixelRangeHalf;
                const float bolden = style.worldSpaceLineWidth * screenPxRange; // worldSpaceLineWidth is actually boldenInPixels, aliased TextStyle with LineStyle
                localAlpha = smoothstep(+globals.antiAliasingFactor / 2.0f + bolden, -globals.antiAliasingFactor / 2.0f + bolden, msdf);
            }
        }
        else if (objType == ObjectType::IMAGE) 
        {
            const float2 uv = input.getImageUV();
            const uint32_t textureId = input.getImageTextureId();

            if (textureId != InvalidTextureIdx)
            {
                float4 colorSample = textures[NonUniformResourceIndex(textureId)].Sample(textureSampler, float2(uv.x, uv.y));
                textureColor = colorSample.rgb;
                localAlpha = colorSample.a;
            }
        }

        uint2 fragCoord = uint2(input.position.xy);
        
        if (localAlpha <= 0)
            discard;
        
        const bool colorFromTexture = objType == ObjectType::IMAGE;
        
        // TODO[Przemek]: But make sure you're still calling this, correctly calculating alpha and texture color.
        // you can add 1 main object and push via DrawResourcesFiller like we already do for other objects (this go in the mainObjects StorageBuffer) and then set the currentMainObjectIdx to 0 here
        // having 1 main object temporarily means that all triangle meshes will be treated as a unified object in blending operations. 
        return calculateFinalColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(fragCoord, localAlpha, currentMainObjectIdx, textureColor, colorFromTexture);
    }
}
