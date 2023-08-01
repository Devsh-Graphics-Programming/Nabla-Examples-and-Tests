

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    CURVE_BOX = 2u,
};

struct DrawObject
{
    uint32_t type;
    uint32_t styleIdx;
    uint64_t address;
};

struct QuadraticBezierInfo
{
    double2 p[3]; // 16*3=48bytes
};

// Curve which is referenced by a "CurveBox"
struct Curve 
{
    // if the middle point is "nan" it means it's a line connected by p0 and p2
    double2 p[3]; // 16*3=48bytes
};

struct CurveBox 
{
    // will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
    double2 aabbMin, aabbMax; // 32
    // each addr references a Curve address into the geometry buffer
    uint64_t curveAddress1; // 40
    uint64_t curveAddress2; // 48
    // because we subdivide curves into smaller monotonic parts we need this info to help us discard invalid solutions
    double curveTmin1, curveTmax1; // 64
    double curveTmin2, curveTmax2; // 92
};

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
    float _pad[2u];
};

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
    
    // data0
    void setColor(in float4 color) { data0 = color; }
    float4 getColor() { return data0; }
    
    // data1 (w component reserved for later)
    float getLineThickness() { return asfloat(data1.x); }
    ObjectType getObjType() { return (ObjectType) data1.y; }
    uint getWriteToAlpha() { return data1.z; }
    
    void setLineThickness(float lineThickness) { data1.x = asuint(lineThickness); }
    void setObjType(ObjectType objType) { data1.y = (uint) objType; }
    void setWriteToAlpha(uint writeToAlpha) { data1.z = writeToAlpha; }
    
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
#endif