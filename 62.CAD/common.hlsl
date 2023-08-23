

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    CURVE_BOX = 2u,
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

struct CurveBox 
{
    // will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
    double2 aabbMin, aabbMax; // 32
    double2 curveMin[3]; // 80
    double2 curveMax[3]; // 128
};

struct Globals
{
    double4x4 viewProjection; // 128 
    double screenToWorldRatio; // 136 - TODO: make a float, no point making it a double
    uint2 resolution; // 144
    float antiAliasingFactor; // 148
    int clipEnabled; // 152
    int2 _pad; // 164
    double4 clip; // 192
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
static const uint32_t AlphaBits = 32u - MainObjectIdxBits;
static const uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
static const uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;

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
    [[vk::location(0)]] float4 data0 : COLOR;
    [[vk::location(1)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(2)]] nointerpolation float4 data2 : COLOR2;
    [[vk::location(3)]] nointerpolation float4 data3 : COLOR3;
    [[vk::location(4)]] nointerpolation float4 data4 : COLOR4;
    // For curve box, has the UV within the AABB
    // UV, curve min & curve max are all 
    [[vk::location(5)]] float2 uv : UV;
    
    // A, B, C quadratic coefficients from the min & max curves,
    // swizzled to the major cordinate and with the major UV coordinate subtracted
    // These can be fed directly into the solve quadratic equation
    // curveMin.C()[major] - uv[major]
    [[vk::location(6)]] float3 minCurveQuadraticCoefficients : MIN_QUADRATIC_COEFFICIENTS;
    [[vk::location(7)]] float3 maxCurveQuadraticCoefficients : MAX_QUADRATIC_COEFFICIENTS;
    
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
    void setMainObjectIdx(uint mainObjIdx) { data1.z = mainObjIdx; }
    
    // data2
    float2 getLineStart() { return data2.xy; }
    float2 getLineEnd() { return data2.zw; }
    float2 getBezierP0() { return data2.xy; }
    float2 getBezierP1() { return data2.zw; }
    
    void setLineStart(float2 lineStart) { data2.xy = lineStart; }
    void setLineEnd(float2 lineEnd) { data2.zw = lineEnd; }
    void setBezierP0(float2 p0) { data2.xy = p0; }
    void setBezierP1(float2 p1) { data2.zw = p1; }
    
    // data3 xy
    float2 getBezierP2() { return data3.xy; }
    void setBezierP2(float2 p2) { data3.xy = p2; }

    // Curves are split in the vertex shader based on their tmin and tmax
    // Min curve is smaller in the minor coordinate (e.g. in the default of y top to bottom sweep,
    // curveMin = smaller x / left, curveMax = bigger x / right)
    // TODO: possible optimization: passing precomputed values for solving the quadratic equation instead

    // data2, data3, data4
    float2 getCurveMinA() { return data2.xy; }
    float2 getCurveMinB() { return data2.zw; }
    float2 getCurveMinC() { return data3.xy; }
    float2 getCurveMaxA() { return data3.zw; }
    float2 getCurveMaxB() { return data4.xy; }
    float2 getCurveMaxC() { return data4.zw; }
    
    void setCurveMinA(float2 p) { data2.xy = p; }
    void setCurveMinB(float2 p) { data2.zw = p; }
    void setCurveMinC(float2 p) { data3.xy = p; }
    void setCurveMaxA(float2 p) { data3.zw = p; }
    void setCurveMaxB(float2 p) { data4.xy = p; }
    void setCurveMaxC(float2 p) { data4.zw = p; }
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<MainObject> mainObjects : register(t2);
#endif