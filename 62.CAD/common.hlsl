
#ifndef _CAD_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_COMMON_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#endif

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    //TODO[Lucas]: another object type for a "CurveBox"
};

// Consists of multiple DrawObjects
struct MainObject
{
    uint32_t styleIdx;
    uint32_t clipProjectionIdx;
};

struct DrawObject
{
    // TODO: use struct bitfields in after DXC update and see if the invalid spirv bug still exists
    uint32_t type_subsectionIdx; // packed to uint16 into uint32
    uint32_t mainObjIndex;
    uint64_t geometryAddress;
};

struct QuadraticBezierInfo
{
    float64_t2 p[3]; // 16*3=48bytes
};

// TODO[Lucas]:
/*
You need a struct here that represents a curve which is referenced by a "CurveBox"
Which is basically the same as QuadraticBezierInfo, if the middle point is "nan" it means it's a line connected by p0 and p2
*/

// TODO[Lucas]:
/*
You need another struct here that represents a "CurveBox" which
0. have a aabb (double2 min, max) which will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
1. references two Curves by `uint64_t address` into the geometry buffer
2. It will also contain tmin,tmax for both curves (becuase we subdivide curves into smaller monotonic parts we need this info to help us discard invalid solutions)
*/

// TODO: Compute this in a compute shader from the world counterparts
//      because this struct includes NDC coordinates, the values will change based camera zoom and move
//      of course we could have the clip values to be in world units and also the matrix to transform to world instead of ndc but that requires extra computations(matrix multiplications) per vertex
struct ClipProjectionData
{
    float64_t3x3 projectionToNDC; // 72 -> because we use scalar_layout
    float32_t2 minClipNDC; // 80
    float32_t2 maxClipNDC; // 88
};

#ifndef __HLSL_VERSION
static_assert(offsetof(ClipProjectionData, projectionToNDC) == 0u);
static_assert(offsetof(ClipProjectionData, minClipNDC) == 72u);
static_assert(offsetof(ClipProjectionData, maxClipNDC) == 80u);
#endif

struct Globals
{
    ClipProjectionData defaultClipProjection; // 88
    double screenToWorldRatio; // 96
    float worldToScreenRatio; // 100
    uint32_t2 resolution; // 108
    float antiAliasingFactor; // 112
};

#ifndef __HLSL_VERSION
static_assert(offsetof(Globals, defaultClipProjection) == 0u);
static_assert(offsetof(Globals, screenToWorldRatio) == 88u);
static_assert(offsetof(Globals, worldToScreenRatio) == 96u);
static_assert(offsetof(Globals, resolution) == 100u);
static_assert(offsetof(Globals, antiAliasingFactor) == 108u);
#endif

struct LineStyle
{
    static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

    // common data
    float32_t4 color;
    float screenSpaceLineWidth;
    float worldSpaceLineWidth;
    
    // stipple pattern data
    int32_t stipplePatternSize;
    float reciprocalStipplePatternLen;
    float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
    float phaseShift;
    
    inline bool hasStipples()
    {
        return stipplePatternSize > 0 ? true : false;
    }
};

NBL_CONSTEXPR uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
NBL_CONSTEXPR uint32_t AlphaBits = 32u - MainObjectIdxBits;
NBL_CONSTEXPR uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
NBL_CONSTEXPR uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;
NBL_CONSTEXPR uint32_t InvalidClipProjectionIdx = 0xffffffff;
NBL_CONSTEXPR uint32_t UseDefaultClipProjectionIdx = InvalidClipProjectionIdx;

#ifndef __cplusplus

uint bitfieldInsert(uint base, uint insert, int offset, int bits)
{
	const uint mask = (1u << bits) - 1u;
	const uint shifted_mask = mask << offset;

	insert &= mask;
	base &= (~shifted_mask);
	base |= (insert << offset);

	return base;
}

uint bitfieldExtract(uint value, int offset, int bits)
{
	uint retval = value;
	retval >>= offset;
	return retval & ((1u<<bits) - 1u);
}

// TODO: Remove these two when we include our builtin shaders
#define nbl_hlsl_PI 3.14159265359
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#define UINT32_MAX 0xffffffffu

struct PSInput
{
    float4 position : SV_Position;
    float4 clip : SV_ClipDistance;
    [[vk::location(0)]] float4 data0 : COLOR;
    [[vk::location(1)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(2)]] nointerpolation float4 data2 : COLOR2;
    [[vk::location(3)]] nointerpolation float4 data3 : COLOR3;
        // ArcLenCalculator<float>
    [[vk::location(4)]] nointerpolation float4 data4 : COLOR4;
    
    
    // TODO[Lucas]: you will need more data here, this struct is what gets sent from vshader to fshader
    /*
        What you need to send additionally for hatches is basically
        + information about the two curves 
        + Their tmin, tmax 
            - (or you could do some curve "splitting" math in the vertex shader to transform those to the same curves with tmin=0 and tmax=1)
            - https://pomax.github.io/bezierinfo/#splitting see this for curve splitting, the derivation might be a little hard to understand but the result is simple, so focus on the result
        + Note: You'll be solving two quadratic equations for the two curves, B_y(t)=coord.y, find `t=t*` for y component of bezier equal to the "scan line"
            - after finding t* you'd find the left and right curve points by Bl_x(tl*) and Br_x(tr*) 
                where you can decide whether to fill or not based on  Bl_x(t*) < pixel.x < Br_x(t*) 
            - Notice the usage of _x and _y in the above because we SWEEP from top to bottom and our "major coordinate" is y by default. 
            - but write code that can be flexible when changing the Sweep direction (use major, minor instead of y, x)
    
        + Based on the info above, you may not need to pass the "y" component of the bezier curves, and only precomputed values for quadratic equation solving
            + that saves us computation (better to compute on each vertex than each fragment)
    */
    
    // Set functions used in vshader, get functions used in fshader
    // We have to do this because we don't have union in hlsl and this is the best way to alias
    
    // data0
    void setColor(in float4 color) { data0 = color; }
    float4 getColor() { return data0; }
    
    // data1 (w component reserved for later)
    float getLineThickness() { return asfloat(data1.x); }
    ObjectType getObjType() { return (ObjectType) data1.y; }
    uint getMainObjectIdx() { return data1.z; }
    
    void setLineThickness(float lineThickness) { data1.x = asuint(lineThickness); }
    void setObjType(ObjectType objType) { data1.y = (uint) objType; }
    void setMainObjectIdx(uint mainObjIdx) { data1.z = mainObjIdx; }
    
    // data2
    float2 getLineStart() { return data2.xy; }
    float2 getLineEnd() { return data2.zw; }
    
    void setLineStart(float2 lineStart) { data2.xy = lineStart; }
    void setLineEnd(float2 lineEnd) { data2.zw = lineEnd; }
    
    // data2 + data3.xy
    nbl::hlsl::shapes::Quadratic<float> getQuadratic()
    {
        return nbl::hlsl::shapes::Quadratic<float>::construct(data2.xy, data2.zw, data3.xy);
    }
    
    void setQuadratic(nbl::hlsl::shapes::Quadratic<float> quadratic)
    {
        data2.xy = quadratic.A;
        data2.zw = quadratic.B;
        data3.xy = quadratic.C;
    }
    
    // data3.zw + data4
    
    void setQuadraticPrecomputedArcLenData(nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator preCompData) 
    {
        data3.zw = float2(preCompData.lenA2, preCompData.AdotB);
        data4 = float4(preCompData.a, preCompData.b, preCompData.c, preCompData.b_over_4a);
    }
    
    nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator getQuadraticArcLenCalculator()
    {
        return nbl::hlsl::shapes::Quadratic<float>::ArcLenCalculator::construct(data3.z, data3.w, data4.x, data4.y, data4.z, data4.w);
    }
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<MainObject> mainObjects : register(t2);
[[vk::binding(5, 0)]] StructuredBuffer<ClipProjectionData> customClipProjections : register(t3);
#endif

#endif