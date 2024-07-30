#ifndef _FLIP_EXAMPLE_GRID_UTILS_HLSL
#define _FLIP_EXAMPLE_GRID_UTILS_HLSL

struct SGridData
{
    float gridCellSize;
    float gridInvCellSize;
    float pad0[2];

    int32_t4 particleInitMin;
    int32_t4 particleInitMax;
    int32_t4 particleInitSize;

    float32_t4 worldMin;
    float32_t4 worldMax;
    int32_t4 gridSize;
};

#ifdef __HLSL_VERSION
static const float POSITION_EPSILON = 1e-4;

void clampPosition(inout float4 position, float4 gridMin, float4 gridMax)
{
    position = clamp(position, gridMin + POSITION_EPSILON, gridMax - POSITION_EPSILON);
}

float4 gridPosToWorldPos(float4 position, SGridData data)
{
    return data.worldMin + position * data.gridCellSize;
}
#endif

#endif