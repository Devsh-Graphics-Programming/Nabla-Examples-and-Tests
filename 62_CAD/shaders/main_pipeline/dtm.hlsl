#ifndef _CAD_EXAMPLE_DTM_HLSL_INCLUDED_
#define _CAD_EXAMPLE_DTM_HLSL_INCLUDED_

#include "line_style.hlsl"

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

float4 calculateDTMOutlineColor(in uint outlineLineStyleIdx, in float3 v[3], in float2 fragPos)
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

float4 calculateGridDTMOutlineColor(in uint outlineLineStyleIdx, in nbl::hlsl::shapes::Line<float> outlineLineSegments[2], in float2 fragPos, in float phaseShift)
{
    LineStyle outlineStyle = loadLineStyle(outlineLineStyleIdx);
    const float outlineThickness = (outlineStyle.screenSpaceLineWidth + outlineStyle.worldSpaceLineWidth * globals.screenToWorldRatio) * 0.5f;
    const float stretch = 1.0f;

    // find distance to outline
    float minDistance = nbl::hlsl::numeric_limits<float>::max;
    if (!outlineStyle.hasStipples() || stretch == InvalidStyleStretchValue)
    {
        for (int i = 0; i < 2; ++i)
        {
            float distance = nbl::hlsl::numeric_limits<float>::max;
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float> >::sdf(outlineLineSegments[i], fragPos, outlineThickness, outlineStyle.isRoadStyleFlag);

            minDistance = min(minDistance, distance);
        }
    }
    else
    {
        for (int i = 0; i < 2; ++i)
        {
            float distance = nbl::hlsl::numeric_limits<float>::max;
            nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(outlineLineSegments[i]);
            LineStyleClipper clipper = LineStyleClipper::construct(outlineStyle, outlineLineSegments[i], arcLenCalc, phaseShift, stretch, globals.worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(outlineLineSegments[i], fragPos, outlineThickness, outlineStyle.isRoadStyleFlag, clipper);

            minDistance = min(minDistance, distance);
        }
    }

    float4 outputColor;
    outputColor.a = 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, minDistance);
    outputColor.a *= outlineStyle.color.a;
    outputColor.rgb = outlineStyle.color.rgb;

    return outputColor;
}

float4 blendUnder(in float4 dstColor, in float4 srcColor)
{
    dstColor.rgb = dstColor.rgb + (1 - dstColor.a) * srcColor.a * srcColor.rgb;
    dstColor.a = (1.0f - srcColor.a) * dstColor.a + srcColor.a;

    return dstColor;
}

E_CELL_DIAGONAL resolveGridDTMCellDiagonal(in uint32_t4 cellData)
{
    float4 cellHeights = asfloat(cellData);

    const bool4 invalidHeights = bool4(
        isInvalidGridDtmHeightValue(cellHeights.x),
        isInvalidGridDtmHeightValue(cellHeights.y),
        isInvalidGridDtmHeightValue(cellHeights.z),
        isInvalidGridDtmHeightValue(cellHeights.w)
    );

    int invalidHeightsCount = 0;
    for (int i = 0; i < 4; ++i)
        invalidHeightsCount += int(invalidHeights[i]);

    if (invalidHeightsCount == 0)
    {
        E_CELL_DIAGONAL a = getDiagonalModeFromCellCornerData(cellData.w);
        return getDiagonalModeFromCellCornerData(cellData.w);
    }

    if (invalidHeightsCount > 1)
        return INVALID;

    if (invalidHeights.x || invalidHeights.z)
        return TOP_LEFT_TO_BOTTOM_RIGHT;
    else if (invalidHeights.y || invalidHeights.w)
        return BOTTOM_LEFT_TO_TOP_RIGHT;

    return INVALID;
}

}

#endif