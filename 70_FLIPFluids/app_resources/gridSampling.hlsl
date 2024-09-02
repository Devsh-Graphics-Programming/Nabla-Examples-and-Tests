#ifndef _FLIP_EXAMPLE_GRID_SAMPLING_HLSL
#define _FLIP_EXAMPLE_GRID_SAMPLING_HLSL

#include "gridUtils.hlsl"

#ifdef __HLSL_VERSION

inline float _getCellValue(int3 cellIdx, float3 S, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    uint cid = cellIdxToFlatIdx(cellIdx, gridData.gridSize);
    return cellIdx[axis] < gridData.gridSize[axis] ? gridBuffer[cid][axis] : 0.0f;
}

inline float _interpolateX(int3 cellIdx, float3 S, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    float sum = 0;
    sum += _getCellValue(cellIdx, S, gridBuffer, gridData, axis) * (1.0 - S.x);
    sum += _getCellValue(cellIdx + int3(1, 0, 0), S, gridBuffer, gridData, axis) * S.x;
    return sum;
}

inline float _interpolateY(int3 cellIdx, float3 S, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    float sum = 0;
    sum += _interpolateX(cellIdx, S, gridBuffer, gridData, axis) * (1.0 - S.y);
    sum += _interpolateX(cellIdx + int3(0, 1, 0), S, gridBuffer, gridData, axis) * S.y;
    return sum;
}

inline float _interpolateZ(int3 cellIdx, float3 S, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    float sum = 0;
    sum += _interpolateY(cellIdx, S, gridBuffer, gridData, axis) * (1.0 - S.z);
    sum += _interpolateY(cellIdx + int3(0, 0, 1), S, gridBuffer, gridData, axis) * S.z;
    return sum;
}

inline float interpolate(int3 cellIdx, float3 S, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    return _interpolateZ(cellIdx, S, gridBuffer, gridData, axis);
}

inline float _sampleVelocity(float3 pos, RWStructuredBuffer<float4> gridBuffer, SGridData gridData, uint axis)
{
    int3 cellIdx = floor(pos - 0.5f);
    float3 s = frac(pos - 0.5f);
    return interpolate(cellIdx, s, gridBuffer, gridData, axis);
}

inline float _sampleVelX(float3 pos, RWStructuredBuffer<float4> gridBuffer, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.5f, 0.0f, 0.0f);
    return _sampleVelocity(gridPos, gridBuffer, gridData, 0);
}

inline float _sampleVelY(float3 pos, RWStructuredBuffer<float4> gridBuffer, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.0f, 0.5f, 0.0f);
    return _sampleVelocity(gridPos, gridBuffer, gridData, 1);
}

inline float _sampleVelZ(float3 pos, RWStructuredBuffer<float4> gridBuffer, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.0f, 0.0f, 0.5f);
    return _sampleVelocity(gridPos, gridBuffer, gridData, 2);
}

inline float3 sampleVelocityAt(float3 pos, RWStructuredBuffer<float4> gridBuffer, SGridData gridData)
{
    float3 val;
    val.x = _sampleVelX(pos, gridBuffer, gridData);
    val.y = _sampleVelY(pos, gridBuffer, gridData);
    val.z = _sampleVelZ(pos, gridBuffer, gridData);
    return val;
}

#endif
#endif