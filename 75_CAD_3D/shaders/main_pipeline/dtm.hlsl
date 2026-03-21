#ifndef _CAD_3D_EXAMPLE_DTM_HLSL_INCLUDED_
#define _CAD_3D_EXAMPLE_DTM_HLSL_INCLUDED_

#include "common.hlsl"

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

struct HeightSegmentTransitionData
{
    float currentHeight;
    float4 currentSegmentColor;
    float boundaryHeight;
    float4 otherSegmentColor;
};

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

float getIntervalPosition(in float height, in float minHeight, in float intervalLength, in bool isCenteredShading)
{
    if (isCenteredShading)
        return ((height - minHeight) / intervalLength + 0.5f);
    else
        return ((height - minHeight) / intervalLength);
}

float32_t4 calcIntervalColor(in int intervalIndex, in DTMHeightShadingSettings settings)
{
    const float minShadingHeight = settings.heightColorMapHeights[0];
    float heightForColor = minShadingHeight + float(intervalIndex) * settings.intervalIndexToHeightMultiplier;

    DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
    int32_t upperBoundHeightIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0, settings.heightColorEntryCount, heightForColor), settings.heightColorEntryCount - 1u);
    int32_t lowerBoundHeightIndex = max(upperBoundHeightIndex - 1, 0);

    float upperBoundHeight = settings.heightColorMapHeights[upperBoundHeightIndex];
    float lowerBoundHeight = settings.heightColorMapHeights[lowerBoundHeightIndex];

    float4 upperBoundColor = settings.heightColorMapColors[upperBoundHeightIndex];
    float4 lowerBoundColor = settings.heightColorMapColors[lowerBoundHeightIndex];

    if (upperBoundHeight == lowerBoundHeight)
    {
        return upperBoundColor;
    }
    else
    {
        float interpolationVal = (heightForColor - lowerBoundHeight) / (upperBoundHeight - lowerBoundHeight);
        return lerp(lowerBoundColor, upperBoundColor, interpolationVal);
    }
}

float32_t4 calculateDTMHeightColor(in DTMHeightShadingSettings settings, in float heightDeriv, in float3 triangleVertices[3], in float2 fragPos, in float height)
{
    const uint32_t heightMapSize = settings.heightColorEntryCount;
    if(heightMapSize == 0)
        return float32_t4(0.0f, 0.0f, 0.0f, 0.0f);
    
    const E_HEIGHT_SHADING_MODE mode = settings.determineHeightShadingMode();
    if(mode == E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS)
    {
        DTMSettingsHeightsAccessor dtmHeightsAccessor = { settings };
        const int upperBoundIndex = min(nbl::hlsl::upper_bound(dtmHeightsAccessor, 0u, heightMapSize, height), heightMapSize - 1u);
        const int mapIndex = max(upperBoundIndex - 1, 0);
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

        return smoothHeightSegmentTransition(transitionInfo, heightDeriv);
    }
    else if(mode == E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS)
    {
        const float minShadingHeight = settings.heightColorMapHeights[0];
        const float intervalPosition = getIntervalPosition(height, minShadingHeight, settings.intervalLength, settings.isCenteredShading);
        const float positionWithinInterval = frac(intervalPosition);
        const int intervalIndex = nbl::hlsl::_static_cast<int>(intervalPosition);

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

        return smoothHeightSegmentTransition(transitionInfo, heightDeriv);
    }
    else if(mode == E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS)
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

        return lerp(lowerBoundColor, upperBoundColor, interpolationVal);
    }

    return float32_t4(0.0f, 0.0f, 0.0f, 0.0f);
}

}

#endif