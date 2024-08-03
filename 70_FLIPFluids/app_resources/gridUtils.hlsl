#ifndef _FLIP_EXAMPLE_GRID_UTILS_HLSL
#define _FLIP_EXAMPLE_GRID_UTILS_HLSL

#ifdef __HLSL_VERSION
static const uint CM_AIR = 0;
static const uint CM_FLUID = 1;
static const uint CM_SOLID = 2;

struct SGridData
{
    float gridCellSize;
    float gridInvCellSize;
    float pad0[2];

    int4 particleInitMin;
    int4 particleInitMax;
    int4 particleInitSize;

    float4 worldMin;
    float4 worldMax;
    int4 gridSize;
};

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