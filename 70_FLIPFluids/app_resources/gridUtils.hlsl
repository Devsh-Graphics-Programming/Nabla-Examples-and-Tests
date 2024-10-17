#ifndef _FLIP_EXAMPLE_GRID_UTILS_HLSL
#define _FLIP_EXAMPLE_GRID_UTILS_HLSL

#ifdef __HLSL_VERSION
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

float4 clampPosition(float4 position, float4 gridMin, float4 gridMax)
{
    return float4(clamp(position.xyz, gridMin.xyz + POSITION_EPSILON, gridMax.xyz - POSITION_EPSILON), 1);
}

int3 clampToGrid(int3 index, int4 gridSize)
{
    return clamp(index, (int3)0, gridSize.xyz - (int3)1);
}

inline uint cellIdxToFlatIdx(int3 index, int4 gridSize)
{
    uint3 idxClamp = clamp(index, (int3)0, gridSize.xyz - (int3)1);
    return idxClamp.x + idxClamp.y * gridSize.x + idxClamp.z * gridSize.x * gridSize.y;
}

inline int3 flatIdxToCellIdx(uint id, int4 gridSize)
{
    int x = id % gridSize.x;
    int y = id / gridSize.x % gridSize.y;
    int z = id / (gridSize.x * gridSize.y);
    return int3(x, y, z);
}

inline float3 cellIdxToWorldPos(int3 index, SGridData data)
{
    return data.worldMin.xyz + (index + 0.5f) * data.gridCellSize;
}

inline float3 worldPosToGridPos(float3 position, SGridData data)
{
    return (position - data.worldMin.xyz) * data.gridInvCellSize;
}

inline int3 worldPosToCellIdx(float3 position, SGridData data)
{
    return floor(worldPosToGridPos(position, data));
}

inline uint worldPosToFlatIdx(float3 position, SGridData data)
{
    return cellIdxToFlatIdx(worldPosToCellIdx(position, data), data.gridSize);
}

inline float4 gridPosToWorldPos(float4 position, SGridData data)
{
    return float4(data.worldMin.xyz + position.xyz * data.gridCellSize, 1);
}
#endif

#endif