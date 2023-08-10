

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    //TODO[Lucas]: another object type for a "CurveBox"
};

// Consists of multiple DrawObjects
struct MainObject
{
    // TODO[Erfan]: probably have objectType here as well?
    uint32_t styleIdx;
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
    double2 p[3]; // 16*3=48bytes
    // TODO[Przemek]: Any Data related to precomputing things for beziers will be here
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

struct Globals
{
    double4x4 viewProjection; // 128 
    double screenToWorldRatio; // 136 - TODO: make a float, no point making it a double
    uint2 resolution; // 144
    float antiAliasingFactor; // 148
    float _pad; // 152
};

struct LineStyle
{
    float4 color;
    float screenSpaceLineWidth;
    float worldSpaceLineWidth;
    // TODO[Przemek]: Anything info related to the stipple pattern will be here
    float _pad[2u];
};

//TODO: USE NBL_CONSTEXPR? in new HLSL PR for Nabla
static const uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
static const uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
static const uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;

#ifndef __cplusplus

// TODO: Remove these two when we include our builtin shaders
#define nbl_hlsl_PI 3.14159265359
#define	nbl_hlsl_FLT_EPSILON 5.96046447754e-08
#define UINT32_MAX 0xffffffffu

struct PSInput
{
    float4 position : SV_Position;
    [[vk::location(0)]] float4 data0 : COLOR;
    [[vk::location(1)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(2)]] nointerpolation float4 data2 : COLOR2;
    [[vk::location(3)]] nointerpolation float4 data3 : COLOR3;
    
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
    
    // TODO[Przemek]: We only had color and thickness to worry about before and as you can see we pass them between vertex and fragment shader (so that fragment shader doesn't have to fetch the linestyles from memory again)
    // but for cases where you found out line styles would be too large to do this kinda stuff with inter-shader memory then only pass the linestyleIdx from vertex shader to fragment shader and fetch the whole lineStyles struct in fragment shader
    // Note: Lucas is also modifying here (added data4,5,6,..) so If need be, I suggest replace a variable like set/getColor to set/getLineStyleIdx to reduce conflicts 
    
    // data0
    void setColor(in float4 color) { data0 = color; }
    float4 getColor() { return data0; }
    
    // data1 (w component reserved for later)
    float getLineThickness() { return asfloat(data1.x); }
    ObjectType getObjType() { return (ObjectType) data1.y; }
    uint getMainObjectIdx() { return data1.z; }
    
    void setLineThickness(float lineThickness) { data1.x = asuint(lineThickness); }
    void setObjType(ObjectType objType) { data1.y = (uint) objType; }
    void setMainObjectIdx(uint mainGeomIdx) { data1.z = mainGeomIdx; }
    
    // data2
    float2 getLineStart() { return data2.xy; }
    float2 getLineEnd() { return data2.zw; }
    float2 getBezierP0() { return data2.xy; }
    float2 getBezierP1() { return data2.zw; }
    
    void setLineStart(float2 lineStart) { data2.xy = lineStart; }
    void setLineEnd(float2 lineEnd) { data2.zw = lineEnd; }
    void setBezierP0(float2 p0) { data2.xy = p0; }
    void setBezierP1(float2 p1) { data2.zw = p1; }
    
    // data3 (zw reserved for later)
    float2 getBezierP2() { return data3.xy; }
    void setBezierP2(float2 p2) { data3.xy = p2; }
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<MainObject> mainObjects : register(t2);
#endif