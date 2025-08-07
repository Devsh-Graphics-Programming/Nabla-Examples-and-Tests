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
    int32_t upperBoundHeightIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, settings.heightColorEntryCount, heightForColor), settings.heightColorEntryCount - 1u);
    int32_t lowerBoundHeightIndex = max(upperBoundHeightIndex - 1, 0);

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

float4 calculateDTMHeightColor(in DTMHeightShadingSettings settings, in float3 triangleVertices[3], in float heightDeriv, in float2 fragPos, in float height)
{
    float4 outputColor = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // HEIGHT SHADING
    const uint32_t heightMapSize = settings.heightColorEntryCount;
    float minShadingHeight = settings.heightColorMapHeights[0];
    float maxShadingHeight = settings.heightColorMapHeights[heightMapSize - 1];

    if (heightMapSize > 0)
    {
        // Do the triangle SDF:
        float2 e0 = (triangleVertices[1] - triangleVertices[0]).xy;
        float2 e1 = (triangleVertices[2] - triangleVertices[1]).xy;
        float2 e2 = (triangleVertices[0] - triangleVertices[2]).xy;
        
        float2 v0 = fragPos - triangleVertices[0].xy;
        float2 v1 = fragPos - triangleVertices[1].xy;
        float2 v2 = fragPos - triangleVertices[2].xy;

        float distanceToLine0 = dot2(v0 - e0 * clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0));
        float distanceToLine1 = dot2(v1 - e1 * clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0));
        float distanceToLine2 = dot2(v2 - e2 * clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0));

        // TODO[Optization]: We can get the sign (whether inside or outside the triangle) from the barycentric coords we already compute outside this func
        // So we can skip this part which tries to figure out which side of each triangle edge line the fragPos relies on
        float o = e0.x * e2.y - e0.y * e2.x;
        float2 d = min(min(float2(distanceToLine0, o * (v0.x * e0.y - v0.y * e0.x)),
                        float2(distanceToLine1, o * (v1.x * e1.y - v1.y * e1.x))),
                        float2(distanceToLine2, o * (v2.x * e2.y - v2.y * e2.x)));
                         
        float triangleSDF = -sqrt(d.x) * sign(d.y);
        
        // Intersect with the region between min and max height shading.
        float minHeightShadingLine = (minShadingHeight - height) / heightDeriv;
        float maxHeightShadingLine = (height - maxShadingHeight) / heightDeriv;

        float convexPolygonSdf = triangleSDF;
        convexPolygonSdf = max(convexPolygonSdf, minHeightShadingLine);
        convexPolygonSdf = max(convexPolygonSdf, maxHeightShadingLine);
        outputColor.a = 1.0f - smoothstep(0.0f, globals.antiAliasingFactor + globals.antiAliasingFactor, convexPolygonSdf);
     
        // calculate height color
        E_HEIGHT_SHADING_MODE mode = settings.determineHeightShadingMode();
        if (mode == E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS)
        {
            DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
            int upperBoundIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0u, heightMapSize, height), heightMapSize - 1u);
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
            uint32_t upperBoundHeightIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0u, heightMapSize - 1u, height), heightMapSize - 1u);
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

float calculateDTMContourSDF(in DTMContourSettings contourSettings, in LineStyle contourStyle, in float worldToScreenRatio, in float3 v[3], in float2 fragPos, in float height)
{
    float distance = nbl::hlsl::numeric_limits<float>::max;
    const float contourThickness = (contourStyle.screenSpaceLineWidth + contourStyle.worldSpaceLineWidth / worldToScreenRatio) * 0.5f;
    const float stretch = 1.0f;
    const float phaseShift = 0.0f;

    const float startHeight = contourSettings.contourLinesStartHeight;
    const float endHeight = contourSettings.contourLinesEndHeight;
    const float interval = contourSettings.contourLinesHeightInterval;
    const int maxContourLineIdx = (endHeight - startHeight) / interval;

    // TODO: it actually can output a negative number, fix
    int contourLineIdx = nbl::hlsl::_static_cast<int>((height - startHeight) / interval + 0.5f);
    contourLineIdx = clamp(contourLineIdx, 0, maxContourLineIdx);
    float contourLineHeight = startHeight + interval * contourLineIdx;

    
    // Sort so that v[0].z >= v[1].z >= v[2].z
    if (v[0].z < v[1].z)
        nbl::hlsl::swap(v[0], v[1]);
    if (v[0].z < v[2].z)
        nbl::hlsl::swap(v[0], v[2]);
    if (v[1].z < v[2].z)
        nbl::hlsl::swap(v[1], v[2]);

    int contourLinePointsIdx = 0;
    float2 contourLinePoints[2];
    for (int i = 0; i < 3; ++i)
    {
        if (contourLinePointsIdx == 2)
            break;

        int minvIdx = 0;
        int maxvIdx = 0;
        
        if (i == 0) { minvIdx = 2; maxvIdx = 0; }
        if (i == 1) { minvIdx = 1; maxvIdx = 0; }
        if (i == 2) { minvIdx = 2; maxvIdx = 1; }
        
        float3 minV = v[minvIdx];
        float3 maxV = v[maxvIdx];
        
        if (contourLineHeight >= minV.z && contourLineHeight <= maxV.z)
        {
            float interpolationVal = (contourLineHeight - minV.z) / (maxV.z - minV.z);
            contourLinePoints[contourLinePointsIdx] = lerp(minV.xy, maxV.xy, clamp(interpolationVal, 0.0f, 1.0f));
            ++contourLinePointsIdx;
        }
    }

    if (contourLinePointsIdx == 2)
    {
        nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(contourLinePoints[0], contourLinePoints[1]);

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
            LineStyleClipper clipper = LineStyleClipper::construct(contourStyle, lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, fragPos, contourThickness, contourStyle.isRoadStyleFlag, clipper);
        }
    }

    return distance;
}

float4 calculateDTMOutlineColor(in uint outlineLineStyleIdx, in float worldToScreenRatio, in float3 v[3], in float2 fragPos)
{
    float4 outputColor;

    LineStyle outlineStyle = loadLineStyle(outlineLineStyleIdx);
    const float outlineThickness = (outlineStyle.screenSpaceLineWidth + outlineStyle.worldSpaceLineWidth / worldToScreenRatio) * 0.5f;
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
            LineStyleClipper clipper = LineStyleClipper::construct(outlineStyle, lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
            distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, fragPos, outlineThickness, outlineStyle.isRoadStyleFlag, clipper);

            minDistance = min(minDistance, distance);
        }
    }

    outputColor.a = 1.0f - smoothstep(-globals.antiAliasingFactor, globals.antiAliasingFactor, minDistance);
    outputColor.a *= outlineStyle.color.a;
    outputColor.rgb = outlineStyle.color.rgb;

    return outputColor;
}

// TODO:
// It's literally sdf with a line shape
// so it should be moved somewhere else and used for every line maybe
float calculateLineSDF(in LineStyle lineStyle, in float worldToScreenRatio, in nbl::hlsl::shapes::Line<float> lineSegment, in float2 fragPos, in float phaseShift)
{
    const float outlineThickness = (lineStyle.screenSpaceLineWidth + lineStyle.worldSpaceLineWidth / worldToScreenRatio) * 0.5f;
    const float stretch = 1.0f;

    float minDistance = nbl::hlsl::numeric_limits<float>::max;
    if (!lineStyle.hasStipples() || stretch == InvalidStyleStretchValue)
    {
        float distance = nbl::hlsl::numeric_limits<float>::max;
        distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, fragPos, outlineThickness, lineStyle.isRoadStyleFlag);
        minDistance = min(minDistance, distance);
    }
    else
    {
        float distance = nbl::hlsl::numeric_limits<float>::max;
        nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);
        LineStyleClipper clipper = LineStyleClipper::construct(lineStyle, lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
        distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, fragPos, outlineThickness, lineStyle.isRoadStyleFlag, clipper);

        minDistance = min(minDistance, distance);
    }

    return minDistance;
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
        return getDiagonalModeFromCellCornerData(cellData.w);

    if (invalidHeightsCount > 1)
        return INVALID;

    if (invalidHeights.x || invalidHeights.z)
        return TOP_LEFT_TO_BOTTOM_RIGHT;
    else if (invalidHeights.y || invalidHeights.w)
        return BOTTOM_LEFT_TO_TOP_RIGHT;

    return INVALID;
}

struct GridDTMTriangle
{
    float3 vertices[3];
};

/**
* grid consists of square cells and cells are divided into two triangles:
* depending on mode it is
* either:        or:
* v2a-------v1   v0-------v2b
* |  A     / |   | \     B  |
* |     /    |   |    \     |
* |  /  B    |   |   A   \  |
* v0-------v2b   v2a-------v1
*/
struct GridDTMCell
{
    GridDTMTriangle triangleA;
    GridDTMTriangle triangleB;
    bool validA;
    bool validB;
};

struct GridDTMHeightMapData
{
    // heihts.x - bottom left texel
    // heihts.y - bottom right texel
    // heihts.z - top right texel
    // heihts.w - top left texel
    float4 heights;
    E_CELL_DIAGONAL cellDiagonal;
};

GridDTMHeightMapData retrieveGridDTMCellDataFromHeightMap(in float2 gridDimensions, in float2 cellCoords, in Texture2D<uint32_t> heightMap)
{
    GridDTMHeightMapData output;

    const float2 location = (cellCoords + float2(0.5f, 0.5f)) / gridDimensions;
    uint32_t4 cellData = heightMap.Gather(textureSampler, float2(location.x, location.y), 0);

    // printf("%u %u %u %u", cellData.x, cellData.y, cellData.z, cellData.w);

    output.heights = asfloat(cellData);
    output.cellDiagonal = dtm::resolveGridDTMCellDiagonal(cellData);
    return output;
}

GridDTMCell calculateCellTriangles(in dtm::GridDTMHeightMapData heightData, in float2 cellCoords, const float cellWidth)
{
    GridDTMCell output;

    // heightData.heihts.x - bottom left texel
    // heightData.heihts.y - bottom right texel
    // heightData.heihts.z - top right texel
    // heightData.heihts.w - top left texel
    float2 gridSpaceCellTopLeftCoords = cellCoords * cellWidth;

    if (heightData.cellDiagonal == E_CELL_DIAGONAL::TOP_LEFT_TO_BOTTOM_RIGHT)
    {
        output.triangleA.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y, heightData.heights.w);
        output.triangleA.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.y);
        output.triangleA.vertices[2] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.x);

        output.triangleB.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y, heightData.heights.w);
        output.triangleB.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.y);
        output.triangleB.vertices[2] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y, heightData.heights.z);
    }
    else
    {
        output.triangleA.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.x);
        output.triangleA.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y, heightData.heights.z);
        output.triangleA.vertices[2] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y, heightData.heights.w);

        output.triangleB.vertices[0] = float3(gridSpaceCellTopLeftCoords.x, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.x);
        output.triangleB.vertices[1] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y, heightData.heights.z);
        output.triangleB.vertices[2] = float3(gridSpaceCellTopLeftCoords.x + cellWidth, gridSpaceCellTopLeftCoords.y + cellWidth, heightData.heights.y);
    }

    output.validA = !isInvalidGridDtmHeightValue(output.triangleA.vertices[0].z) && !isInvalidGridDtmHeightValue(output.triangleA.vertices[1].z) && !isInvalidGridDtmHeightValue(output.triangleA.vertices[2].z);
    output.validB = !isInvalidGridDtmHeightValue(output.triangleB.vertices[0].z) && !isInvalidGridDtmHeightValue(output.triangleB.vertices[1].z) && !isInvalidGridDtmHeightValue(output.triangleB.vertices[2].z);

    return output;
}

}

#endif