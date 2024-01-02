#ifndef _CAD_EXAMPLE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_COMMON_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#endif

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    CURVE_BOX = 2u,
    POLYLINE_CONNECTOR = 3u
};

enum class MajorAxis : uint32_t
{
    MAJOR_X = 0u,
    MAJOR_Y = 1u,
};

// Consists of multiple DrawObjects
struct MainObject
{
    // TODO[Erfan]: probably have objectType here as well?
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

struct LinePointInfo
{
    float64_t2 p;
    float32_t phaseShift;
    float32_t _reserved_pad;
};

struct QuadraticBezierInfo
{
    nbl::hlsl::shapes::QuadraticBezier<float64_t> shape; // 48bytes = 3 (control points) x 16 (float64_t2)
    float32_t phaseShift;
    float32_t _reserved_pad;
};
#ifndef __HLSL_VERSION
static_assert(offsetof(QuadraticBezierInfo, phaseShift) == 48u);
#endif

struct PolylineConnector
{
    float64_t2 circleCenter;
    float32_t2 v;
    float32_t cosAngleDifferenceHalf;
    float32_t _reserved_pad;
};

// NOTE: Don't attempt to pack curveMin/Max to uints because of limited range of values, we need the logarithmic precision of floats (more precision near 0)
struct CurveBox
{
    // will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
    float64_t2 aabbMin; // 16
    float64_t2 aabbMax; // 32
    float32_t2 curveMin[3]; // 56
    float32_t2 curveMax[3]; // 80
};

#ifndef __HLSL_VERSION
static_assert(offsetof(CurveBox, aabbMin) == 0u);
static_assert(offsetof(CurveBox, aabbMax) == 16u);
static_assert(offsetof(CurveBox, curveMin[0]) == 32u);
static_assert(offsetof(CurveBox, curveMax[0]) == 56u);
static_assert(sizeof(CurveBox) == 80u);
#endif
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
    float miterLimit; // 116
    float32_t3 _padding; // 128
};

#ifndef __HLSL_VERSION
static_assert(offsetof(Globals, defaultClipProjection) == 0u);
static_assert(offsetof(Globals, screenToWorldRatio) == 88u);
static_assert(offsetof(Globals, worldToScreenRatio) == 96u);
static_assert(offsetof(Globals, resolution) == 100u);
static_assert(offsetof(Globals, antiAliasingFactor) == 108u);
static_assert(offsetof(Globals, miterLimit) == 112u);
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
    uint32_t stipplePattern[STIPPLE_PATTERN_MAX_SZ]; // packed float into uint (top two msb indicate leftIsDotPattern and rightIsDotPattern as an optimization)
    uint32_t isRoadStyleFlag; // can pack more bools here in the future

    float getStippleValue(const uint32_t ix)
    {
        const uint32_t floatValBis = 0xffffffff >> 2; // clear two msb bits reserved for something else
        return (stipplePattern[ix] & floatValBis) / float(1u << 29);
    }

    bool isLeftDot(const uint32_t ix)
    {
        // stipplePatternSize is odd by construction (pattern starts with + and ends with -)
        return (stipplePattern[ix] & (1u << 30)) > 0;
    }

    bool isRightDot(const uint32_t ix)
    {
        // stipplePatternSize is odd by construction (pattern starts with + and ends with -)
        return (stipplePattern[ix] & (1u << 31)) > 0;
    }

    bool hasStipples()
    {
        return stipplePatternSize > 0 ? true : false;
    }
};

#ifdef __cplusplus
inline bool operator==(const LineStyle& lhs, const LineStyle& rhs)
{
    const bool areParametersEqual =
        lhs.color == rhs.color &&
        lhs.screenSpaceLineWidth == rhs.screenSpaceLineWidth &&
        lhs.worldSpaceLineWidth == rhs.worldSpaceLineWidth &&
        lhs.stipplePatternSize == rhs.stipplePatternSize &&
        lhs.reciprocalStipplePatternLen == rhs.reciprocalStipplePatternLen;

    if (!areParametersEqual)
        return false;

    const bool isStipplePatternArrayEqual = (lhs.stipplePatternSize > 0) ? std::memcmp(lhs.stipplePattern, rhs.stipplePattern, sizeof(decltype(lhs.stipplePatternSize)) * lhs.stipplePatternSize) : true;

    return isStipplePatternArrayEqual;
}
#endif

NBL_CONSTEXPR uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
NBL_CONSTEXPR uint32_t AlphaBits = 32u - MainObjectIdxBits;
NBL_CONSTEXPR uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
NBL_CONSTEXPR uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;
NBL_CONSTEXPR uint32_t InvalidClipProjectionIdx = 0xffffffff;
NBL_CONSTEXPR uint32_t UseDefaultClipProjectionIdx = InvalidClipProjectionIdx;
NBL_CONSTEXPR MajorAxis SelectedMajorAxis = MajorAxis::MAJOR_Y;
// TODO: get automatic version working on HLSL
NBL_CONSTEXPR MajorAxis SelectedMinorAxis = MajorAxis::MAJOR_X; //(MajorAxis) (1 - (uint32_t) SelectedMajorAxis);

#ifdef __HLSL_VERSION

// TODO: Use these in C++ as well once nbl::hlsl::numeric_limits<uint32_t> compiles on C++
float32_t2 unpackCurveBoxUnorm(uint32_t2 value)
{
    return float32_t2(value) / float32_t(nbl::hlsl::numeric_limits<uint32_t>::max);
}

float32_t2 unpackCurveBoxSnorm(int32_t2 value)
{
    return float32_t2(value) / float32_t(nbl::hlsl::numeric_limits<int32_t>::max);
}


uint32_t2 packCurveBoxUnorm(float32_t2 value)
{
    return value * float32_t(nbl::hlsl::numeric_limits<uint32_t>::max);
}

int32_t2 packCurveBoxSnorm(float32_t2 value)
{
    return value * float32_t(nbl::hlsl::numeric_limits<int32_t>::max);
}

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

// The root we're always looking for:
// 2 * C / (-B - detSqrt)
// We send to the FS: -B, 2C, det
template<typename float_t>
struct PrecomputedRootFinder 
{
    using float2_t = vector<float_t, 2>;
    using float3_t = vector<float_t, 3>;
    
    float_t C2;
    float_t negB;
    float_t det;

    float_t computeRoots() 
    {
        return C2 / (negB - sqrt(det));
    }

    static PrecomputedRootFinder construct(float_t negB, float_t C2, float_t det)
    {
        PrecomputedRootFinder result;
        result.C2 = C2;
        result.det = det;
        result.negB = negB;
        return result;
    }

    static PrecomputedRootFinder construct(nbl::hlsl::math::equations::Quadratic<float_t> quadratic)
    {
        PrecomputedRootFinder result;
        result.C2 = quadratic.c * 2.0;
        result.negB = -quadratic.b;
        result.det = quadratic.b * quadratic.b - 4.0 * quadratic.a * quadratic.c;
        return result;
    }
};

struct PSInput
{
    float4 position : SV_Position;
    float4 clip : SV_ClipDistance;
    [[vk::location(0)]] nointerpolation uint4 data1 : COLOR1;
    [[vk::location(1)]] nointerpolation float4 data2 : COLOR2;
    [[vk::location(2)]] nointerpolation float4 data3 : COLOR3;
    [[vk::location(3)]] nointerpolation float4 data4 : COLOR4;
    // Data segments that need interpolation, mostly for hatches
    [[vk::location(5)]] float2 interp_data5 : COLOR5;
    // ArcLenCalculator<float>

    // Set functions used in vshader, get functions used in fshader
    // We have to do this because we don't have union in hlsl and this is the best way to alias
    
    // data1 (w component reserved for later)
    ObjectType getObjType() { return (ObjectType) data1.x; }
    uint getMainObjectIdx() { return data1.y; }
    float getLineThickness() { return asfloat(data1.z); }
    
    void setObjType(ObjectType objType) { data1.x = (uint) objType; }
    void setMainObjectIdx(uint mainObjIdx) { data1.y = mainObjIdx; }
    void setLineThickness(float lineThickness) { data1.z = asuint(lineThickness); }
    
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
    nbl::hlsl::math::equations::Quadratic<float> getCurveMinMinor() {
        return nbl::hlsl::math::equations::Quadratic<float>::construct(data2.x, data2.y, data2.z);
    }
    nbl::hlsl::math::equations::Quadratic<float> getCurveMaxMinor() {
        return nbl::hlsl::math::equations::Quadratic<float>::construct(data2.w, data3.x, data3.y);
    }

    void setCurveMinMinor(nbl::hlsl::math::equations::Quadratic<float> bezier) {
        data2.x = bezier.a;
        data2.y = bezier.b;
        data2.z = bezier.c;
    }
    void setCurveMaxMinor(nbl::hlsl::math::equations::Quadratic<float> bezier) {
        data2.w = bezier.a;
        data3.x = bezier.b;
        data3.y = bezier.c;
    }

    // data4
    nbl::hlsl::math::equations::Quadratic<float> getCurveMinMajor() {
        return nbl::hlsl::math::equations::Quadratic<float>::construct(data4.x, data4.y, data3.z);
    }
    nbl::hlsl::math::equations::Quadratic<float> getCurveMaxMajor() {
        return nbl::hlsl::math::equations::Quadratic<float>::construct(data4.z, data4.w, data3.w);
    }

    void setCurveMinMajor(nbl::hlsl::math::equations::Quadratic<float> bezier) {
        data4.x = bezier.a;
        data4.y = bezier.b;
        data3.z = bezier.c;
    }
    void setCurveMaxMajor(nbl::hlsl::math::equations::Quadratic<float> bezier) {
        data4.z = bezier.a;
        data4.w = bezier.b;
        data3.w = bezier.c;
    }

    // Curve box value along minor & major axis
    float getMinorBBoxUv() { return interp_data5.x; };
    void setMinorBBoxUv(float minorBBoxUv) { interp_data5.x = minorBBoxUv; }
    float getMajorBBoxUv() { return interp_data5.y; };
    void setMajorBBoxUv(float majorBBoxUv) { interp_data5.y = majorBBoxUv; }

    float2 getCurveBoxScreenSpaceSize() { return asfloat(data1.zw); }
    void setCurveBoxScreenSpaceSize(float2 aabbSize) { data1.zw = asuint(aabbSize); }

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
    
    void setQuadraticPrecomputedArcLenData(nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator preCompData) 
    {
        data3.zw = float2(preCompData.lenA2, preCompData.AdotB);
        data4 = float4(preCompData.a, preCompData.b, preCompData.c, preCompData.b_over_4a);
    }
    
    nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator getQuadraticArcLengthCalculator()
    {
        return nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator::construct(data3.z, data3.w, data4.x, data4.y, data4.z, data4.w);
    }

    // data5.x

    void setCurrentPhaseShift(float phaseShift)
    {
        interp_data5.x = phaseShift;
    }

    float getCurrentPhaseShift()
    {
        return interp_data5.x;
    }

    // POLYLINE_CONNECTOR data

    void setPolylineConnectorTrapezoidStart(float2 trapezoidStart) { data2.xy = trapezoidStart; }
    void setPolylineConnectorTrapezoidEnd(float2 trapezoidEnd) { data2.zw = trapezoidEnd; }
    void setPolylineConnectorTrapezoidShortBase(float shortBase) { data3.x = shortBase; }
    void setPolylineConnectorTrapezoidLongBase(float longBase) { data3.y = longBase; }
    void setPolylineConnectorCircleCenter(float2 C) { data3.zw = C; }

    float2 getPolylineConnectorTrapezoidStart() { return data2.xy; }
    float2 getPolylineConnectorTrapezoidEnd() { return data2.zw; }
    float getPolylineConnectorTrapezoidShortBase() { return data3.x; }
    float getPolylineConnectorTrapezoidLongBase() { return data3.y; }
    float2 getPolylineConnectorCircleCenter() { return data3.zw; }
};

[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t1);
[[vk::binding(4, 0)]] StructuredBuffer<MainObject> mainObjects : register(t2);
[[vk::binding(5, 0)]] StructuredBuffer<ClipProjectionData> customClipProjections : register(t3);

// shared by both vertex and fragment shader
struct StyleAccessor
{
    uint32_t styleIdx;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return lineStyles[styleIdx].getStippleValue(ix);
    }
};
#endif

#endif