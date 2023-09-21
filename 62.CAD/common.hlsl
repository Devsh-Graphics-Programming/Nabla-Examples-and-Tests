#ifndef __cplusplus
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#endif

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
    double2 arcLen;
};

struct CurveBox 
{
    // will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
    double2 aabbMin;
    double2 aabbMax; // 32
    double2 curveMin[3]; // 80
    double2 curveMax[3]; // 128
};

struct Globals
{
    double4x4 viewProjection; // 128 
    float screenToWorldRatio; // 132
    float worldToScreenRatio; // 136
    uint2 resolution; // 144
    float antiAliasingFactor; // 148
    int _pad; // 152
};

struct LineStyle
{
    static const uint32_t STIPPLE_PATTERN_MAX_SZ = 15u;

    // common data
    float4 color;
    float screenSpaceLineWidth;
    float worldSpaceLineWidth;
    
    // stipple pattern data
    int32_t stipplePatternSize;
    float recpiprocalStipplePatternLen;
    float stipplePattern[STIPPLE_PATTERN_MAX_SZ];
    float phaseShift;
    
    inline bool hasStipples()
    {
        return stipplePatternSize > 0 ? true : false;
    }
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
    [[vk::location(5)]] float4 interp_data5 : COLOR5;
    [[vk::location(6)]] float4 interp_data6 : COLOR6;

        // ArcLenCalculator<float>

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
    
    // data3 xy
    float2 getBezierP2() { return data3.xy; }
    void setBezierP2(float2 p2) { data3.xy = p2; }

    // Curves are split in the vertex shader based on their tmin and tmax
    // Min curve is smaller in the minor coordinate (e.g. in the default of y top to bottom sweep,
    // curveMin = smaller x / left, curveMax = bigger x / right)
    // TODO: possible optimization: passing precomputed values for solving the quadratic equation instead

    // data2, data3, data4
    nbl::hlsl::shapes::QuadraticBezier<double> getCurveMinBezier() {
        return nbl::hlsl::shapes::Quadratic<double>::construct(data2.xy, data2.zw, data3.xy);
    }
    nbl::hlsl::shapes::QuadraticBezier<double> getCurveMaxBezier() {
        return nbl::hlsl::shapes::Quadratic<double>::construct(data3.zw, data4.xy, data4.zw);
    }

    void setCurveMinBezier(nbl::hlsl::shapes::QuadraticBezier<double> bezier) {
        data2.xy = bezier.A;
        data2.zw = bezier.B;
        data3.xy = bezier.C;
    }
    void setCurveMaxBezier(nbl::hlsl::shapes::QuadraticBezier<double> bezier) {
        data3.zw = bezier.A;
        data4.xy = bezier.B;
        data4.zw = bezier.C;
    }

    // interp_data5, interp_data6    

    // A, B, C quadratic coefficients from the min & max curves,
    // swizzled to the major cordinate and with the major UV coordinate subtracted
    // These can be used to solve the quadratic equation
    //
    // a, b, c = curveMin.a,b,c()[major] - uv[major]
    // 
    // SolveQuadratic:
    // det = b*b-4.f*a*c;
    // rcp = 0.5f/a;
    // detSqrt = sqrt(det)*rcp;
    // tmp = b*rcp;
    // res = float2(-detSqrt,detSqrt)-tmp;
    //
    // Collapsed version:
    // detrcp2 = det * rcp * rcp
    // brcp = b * rcp
    //
    // (In fragment shader)
    // detSqrt = sqrt(detrcp2)
    // res = float2(-detSqrt,detSqrt)-bRcp;
    float getMinCurveDetRcp2() { return interp_data5.x; }
    float getMinCurveBrcp() { return interp_data5.y; }
    float getMaxCurveDetRcp2() { return interp_data5.z; }
    float getMaxCurveBrcp() { return interp_data5.w; }
    
    void setMinCurveDetRcp2(float v) { interp_data5.x = v; }
    void setMinCurveBrcp(float v) { interp_data5.y = v; }
    void setMaxCurveDetRcp2(float v) { interp_data5.z = v; }
    void setMaxCurveBrcp(float v) { interp_data5.w = v; }
    
    // Curve box UV value along minor axis
    float getUVMinor() { return interp_data6.x; };
    void setUVMinor(float uv) { interp_data6.x = uv; }

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
#endif