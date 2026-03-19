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

float32_t4 calculateDTMHeightColor(in DTMHeightShadingSettings settings, in float3 triangleVertices[3], in float2 fragPos, in float height)
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

        return settings.heightColorMapColors[mapIndex];
    }
    else if(mode == E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS)
    {
        const float minShadingHeight = settings.heightColorMapHeights[0];
        const float intervalPosition = getIntervalPosition(height, minShadingHeight, settings.intervalLength, settings.isCenteredShading);
        const float positionWithinInterval = frac(intervalPosition);
        const int intervalIndex = nbl::hlsl::_static_cast<int>(intervalPosition);

        return calcIntervalColor(intervalIndex, settings);
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