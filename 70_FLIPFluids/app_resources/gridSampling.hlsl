#ifndef _FLIP_EXAMPLE_GRID_SAMPLING_HLSL
#define _FLIP_EXAMPLE_GRID_SAMPLING_HLSL

#include "gridUtils.hlsl"

#ifdef __HLSL_VERSION

// adapted from CUDA Cubic B-Spline Interpolation by Danny Ruitjers
// https://www.dannyruijters.nl/cubicinterpolation/
inline float4 cubic(float v)
{
    float4 n = float4(1.0f, 2.0f, 3.0f, 4.0f) - v;
    float4 s = n * n * n;
    float x = s.x;
    float y = s.y - 4.0f * s.x;
    float z = s.z - 4.0f * s.y + 6.0f * s.x;
    float w = 6.0 - x - y - z;
    return float4(x, y, z, w) * (1.0f / 6.0f);
}

inline float4 tricubicInterpolate(Texture3D<float4> grid, SamplerState _sampler, float3 coords, int4 gridSize)
{
    float4 texelSize = float4(1.0f / gridSize.xz, gridSize.xz);
    coords = coords * gridSize.xyz - 0.5f;

    float3 invCoords = frac(coords);
    coords -= invCoords;

    float4 xcubic = cubic(invCoords.x);
    float4 ycubic = cubic(invCoords.y);
    float4 zcubic = cubic(invCoords.z);

    float2 cx = coords.xx + float2(-0.5f, 1.5f);
    float2 cy = coords.yy + float2(-0.5f, 1.5f);
    float2 cz = coords.zz + float2(-0.5f, 1.5f);

    float2 sx = xcubic.xz + xcubic.yw;
    float2 sy = ycubic.xz + ycubic.yw;
    float2 sz = zcubic.xz + zcubic.yw;

    float2 offsetx = cx + xcubic.yw / sx;
    float2 offsety = cy + ycubic.yw / sy;
    float2 offsetz = cz + zcubic.yw / sz;
    offsetx /= gridSize.xx;
    offsety /= gridSize.yy;
    offsetz /= gridSize.zz;

    float4 tex0 = grid.SampleLevel(_sampler, float3(offsetx.x, offsety.x, offsetz.x), 0);
    float4 tex1 = grid.SampleLevel(_sampler, float3(offsetx.y, offsety.x, offsetz.x), 0);
    float4 tex2 = grid.SampleLevel(_sampler, float3(offsetx.x, offsety.y, offsetz.x), 0);
    float4 tex3 = grid.SampleLevel(_sampler, float3(offsetx.y, offsety.y, offsetz.x), 0);
    float4 tex4 = grid.SampleLevel(_sampler, float3(offsetx.x, offsety.x, offsetz.y), 0);
    float4 tex5 = grid.SampleLevel(_sampler, float3(offsetx.y, offsety.x, offsetz.y), 0);
    float4 tex6 = grid.SampleLevel(_sampler, float3(offsetx.x, offsety.y, offsetz.y), 0);
    float4 tex7 = grid.SampleLevel(_sampler, float3(offsetx.y, offsety.y, offsetz.y), 0);

    float gx = sx.x / (sx.x + sx.y);
    float gy = sy.x / (sy.x + sy.y);
    float gz = sz.x / (sz.x + sz.y);

    float4 x0 = lerp(tex1, tex0, gx);
    float4 x1 = lerp(tex3, tex2, gx);
    float4 x2 = lerp(tex5, tex4, gx);
    float4 x3 = lerp(tex6, tex6, gx);

    float4 y0 = lerp(x1, x0, gy);
    float4 y1 = lerp(x3, x2, gy);
    
    return lerp(y1, y0, gz);
}

inline float _sampleVelocity(float3 pos, Texture3D<float4> grid, SamplerState _sampler, SGridData gridData, uint axis)
{
    float3 coords = pos / gridData.gridSize.xyz;
    // float4 s = grid.SampleLevel(_sampler, coords, 0);
    float4 s = tricubicInterpolate(grid, _sampler, coords, gridData.gridSize);
    return s[axis];
}

inline float _sampleVelX(float3 pos, Texture3D<float4> grid, SamplerState _sampler, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.5f, 0.0f, 0.0f);
    return _sampleVelocity(gridPos, grid, _sampler, gridData, 0);
}

inline float _sampleVelY(float3 pos, Texture3D<float4> grid, SamplerState _sampler, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.0f, 0.5f, 0.0f);
    return _sampleVelocity(gridPos, grid, _sampler, gridData, 1);
}

inline float _sampleVelZ(float3 pos, Texture3D<float4> grid, SamplerState _sampler, SGridData gridData)
{
    float3 gridPos = worldPosToGridPos(pos, gridData) + float3(0.0f, 0.0f, 0.5f);
    return _sampleVelocity(gridPos, grid, _sampler, gridData, 2);
}

inline float3 sampleVelocityAt(float3 pos, Texture3D<float4> grid, SamplerState _sampler, SGridData gridData)
{
    float3 val;
    val.x = _sampleVelX(pos, grid, _sampler, gridData);
    val.y = _sampleVelY(pos, grid, _sampler, gridData);
    val.z = _sampleVelZ(pos, grid, _sampler, gridData);
    return val;
}

#endif
#endif