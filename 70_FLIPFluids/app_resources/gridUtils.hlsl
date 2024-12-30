#ifndef _FLIP_EXAMPLE_GRID_UTILS_HLSL
#define _FLIP_EXAMPLE_GRID_UTILS_HLSL

struct SGridData
{
    float32_t gridCellSize;
    float32_t gridInvCellSize;

    int32_t4 particleInitMin;
    int32_t4 particleInitMax;
    int32_t4 particleInitSize;

    float32_t4 worldMin;
    float32_t4 worldMax;
    int32_t4 gridSize;
};

#ifdef __HLSL_VERSION

static const float POSITION_EPSILON = 1e-4;

float3 clampPosition(float3 position, float4 gridMin, float4 gridMax)
{
    return clamp(position, gridMin.xyz + POSITION_EPSILON, gridMax.xyz - POSITION_EPSILON);
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

inline float3 gridPosToWorldPos(float3 position, SGridData data)
{
    return data.worldMin.xyz + position * data.gridCellSize;
}

int3 flatIdxToLocalGridID(uint idx, int size)
{
    uint a = size * size;
    int3 b;
    b.z = idx / a;
    b.x = idx - b.z * a;
    b.y = b.x / size;
    b.x = b.x - b.y * size;
    return b;
}
#endif

#endif