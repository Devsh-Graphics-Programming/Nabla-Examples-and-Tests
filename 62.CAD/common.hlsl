

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
    
    // TODO: Should I keep following the "COLORx" pattern?

    // Curves are split in the vertex shader based on their tmin and tmax
    // Min curve is smaller in the minor coordinate (e.g. in the default of y top to bottom sweep,
    // curveMin = smaller x / left, curveMax = bigger x / right)
    // TODO: possible optimization: passing precomputed values for solving the quadratic equation instead
    [[vk::location(4)]] nointerpolation double2 curveMin[3] : CURVE_LEFT;
    [[vk::location(5)]] nointerpolation double2 curveMax[3] : CURVE_RIGHT;
    
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

    // curveMin & curveMax
    double2 getCurveMinP0() { return curveMin[0]; }
    double2 getCurveMinP1() { return curveMin[1]; }
    double2 getCurveMinP2() { return curveMin[2]; }

    double2 getCurveMaxP0() { return curveMax[0]; }
    double2 getCurveMaxP1() { return curveMax[1]; }
    double2 getCurveMaxP2() { return curveMax[2]; }
    
    void setCurveMinP0(double2 p) { curveMin[0] = p; }
    void setCurveMinP1(double2 p) { curveMin[1] = p; }
    void setCurveMinP2(double2 p) { curveMin[2] = p; }

    void setCurveMaxP0(double2 p) { curveMax[0] = p; }
    void setCurveMaxP1(double2 p) { curveMax[1] = p; }
    void setCurveMaxP2(double2 p) { curveMax[2] = p; }
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t1);
#endif