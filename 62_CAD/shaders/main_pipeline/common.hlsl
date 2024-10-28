#ifndef _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_
#define _CAD_EXAMPLE_MAIN_PIPELINE_COMMON_HLSL_INCLUDED_

#include "../globals.hlsl"
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#endif

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    CURVE_BOX = 2u,
    POLYLINE_CONNECTOR = 3u,
    FONT_GLYPH = 4u,
    IMAGE = 5u
};

enum class MajorAxis : uint32_t
{
    MAJOR_X = 0u,
    MAJOR_Y = 1u,
};

// Consists of multiple DrawObjects
struct MainObject
{
    uint32_t styleIdx;
    uint32_t pad; // do I even need this? it's stored in structured buffer not bda
    uint64_t clipProjectionAddress;
};

struct DrawObject
{
    uint32_t type_subsectionIdx; // packed two uint16 into uint32
    uint32_t mainObjIndex;
    uint64_t geometryAddress;
};

struct LinePointInfo
{
    float64_t2 p;
    float32_t phaseShift;
    float32_t stretchValue;
};

struct QuadraticBezierInfo
{
    nbl::hlsl::shapes::QuadraticBezier<float64_t> shape; // 48bytes = 3 (control points) x 16 (float64_t2)
    float32_t phaseShift;
    float32_t stretchValue;
};
#ifndef __HLSL_VERSION
static_assert(offsetof(QuadraticBezierInfo, phaseShift) == 48u);
#endif

struct GlyphInfo
{
    float64_t2 topLeft; // 2 * 8 = 16 bytes
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (32)
    // unorm8 minU;
    // unorm8 minV;
    // uint16 textureId;
    uint32_t minUV_textureID_packed; // 4 bytes (36)
    
#ifndef __HLSL_VERSION
    GlyphInfo(float64_t2 topLeft, float32_t2 dirU, float32_t aspectRatio, uint16_t textureId, float32_t2 minUV) :
        topLeft(topLeft),
        dirU(dirU),
        aspectRatio(aspectRatio)
    {
        assert(textureId < nbl::hlsl::numeric_limits<uint16_t>::max);
        packMinUV_TextureID(minUV, textureId);
    }
#endif

    void packMinUV_TextureID(float32_t2 minUV, uint16_t textureId)
    {
        minUV_textureID_packed = textureId;
        uint32_t uPacked = (uint32_t)(clamp(minUV.x, 0.0f, 1.0f) * 255.0f);
        uint32_t vPacked = (uint32_t)(clamp(minUV.y, 0.0f, 1.0f) * 255.0f);
        minUV_textureID_packed = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(minUV_textureID_packed, uPacked, 16, 8);
        minUV_textureID_packed = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(minUV_textureID_packed, vPacked, 24, 8);
    }

    float32_t2 getMinUV()
    {
        return float32_t2(
            float32_t(nbl::hlsl::glsl::bitfieldExtract<uint32_t>(minUV_textureID_packed, 16, 8)) / 255.0,
            float32_t(nbl::hlsl::glsl::bitfieldExtract<uint32_t>(minUV_textureID_packed, 24, 8)) / 255.0
        );
    }

    uint16_t getTextureID()
    {
        return uint16_t(nbl::hlsl::glsl::bitfieldExtract<uint32_t>(minUV_textureID_packed, 0, 16));
    }
};

struct ImageObjectInfo
{
    float64_t2 topLeft; // 2 * 8 = 16 bytes (16)
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (28)
    uint32_t textureID; // 4 bytes (32)
};

static uint32_t packR11G11B10_UNORM(float32_t3 color)
{
    // Scale and convert to integers
    uint32_t r = (uint32_t)(clamp(color.r, 0.0f, 1.0f) * 2047.0f + 0.5f); // 11 bits -> 2^11 - 1 = 2047
    uint32_t g = (uint32_t)(clamp(color.g, 0.0f, 1.0f) * 2047.0f + 0.5f); // 11 bits -> 2^11 - 1 = 2047
    uint32_t b = (uint32_t)(clamp(color.b, 0.0f, 1.0f) * 1023.0f + 0.5f); // 10 bits -> 2^10 - 1 = 1023

    // Insert each component into the correct position
    uint32_t packed = r;  // R: bits 0-10
    packed = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(packed, g, 11, 11); // G: bits 11-21
    packed = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(packed, b, 22, 10); // B: bits 22-31

    return packed;
}

static float32_t3 unpackR11G11B10_UNORM(uint32_t packed)
{
    float32_t3 color;

    // Extract each component from the packed integer
    uint32_t r = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packed, 0, 11);  // R: bits 0-10
    uint32_t g = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packed, 11, 11); // G: bits 11-21
    uint32_t b = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packed, 22, 10); // B: bits 22-31

    // Convert back to float and scale to [0, 1] range
    color.r = (float32_t)(r) / 2047.0f;
    color.g = (float32_t)(g) / 2047.0f;
    color.b = (float32_t)(b) / 1023.0f;

    return color;
}

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
    float64_t2 aabbMax; // 32 , TODO: we know it's a square/box -> we save 8 bytes if we needed to store extra data
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

NBL_CONSTEXPR uint32_t InvalidRigidSegmentIndex = 0xffffffff;
NBL_CONSTEXPR float InvalidStyleStretchValue = nbl::hlsl::numeric_limits<float>::infinity;

// The color parameter is also used for styling non-curve objects such as text glyphs and hatches with solid color
struct LineStyle
{
    const static uint32_t StipplePatternMaxSize = 14u;

    // common data
    float32_t4 color;
    float screenSpaceLineWidth; // alternatively used as TextStyle::italicTiltSlope
    float worldSpaceLineWidth;  // alternatively used as TextStyle::boldInPixels
    
    // stipple pattern data
    int32_t stipplePatternSize;
    float reciprocalStipplePatternLen;
    uint32_t stipplePattern[StipplePatternMaxSize]; // packed float into uint (top two msb indicate leftIsDotPattern and rightIsDotPattern as an optimization)
    uint32_t isRoadStyleFlag;
    uint32_t rigidSegmentIdx; // TODO: can be more mem efficient with styles by packing this along other values, since stipple pattern size is bounded by StipplePatternMaxSize 

    float getStippleValue(const uint32_t ix)
    {
        const uint32_t floatValBis = 0xffffffff >> 2; // clear two msb bits reserved for something else
        return (stipplePattern[ix] & floatValBis) / float(1u << 29);
    }

    void setStippleValue(const uint32_t ix, const float val)
    {
        stipplePattern[ix] = (uint32_t)(val * (1u << 29u));
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

    void stretch(float stretch)
    {
        reciprocalStipplePatternLen /= stretch;
    }
};

#ifndef __HLSL_VERSION
inline bool operator==(const LineStyle& lhs, const LineStyle& rhs)
{
    // Compare bits of the screen space line width values, as they may have been bit cast into integers
    // for the texture IDs, and can't be compared when that results in a NaN or Infinity float
    const int comparisonResult = std::memcmp(&lhs.screenSpaceLineWidth, &rhs.screenSpaceLineWidth, sizeof(float));
    const bool areParametersEqual =
        lhs.color == rhs.color &&
        comparisonResult == 0 &&
        lhs.worldSpaceLineWidth == rhs.worldSpaceLineWidth &&
        lhs.stipplePatternSize == rhs.stipplePatternSize &&
        lhs.reciprocalStipplePatternLen == rhs.reciprocalStipplePatternLen &&
        lhs.isRoadStyleFlag == rhs.isRoadStyleFlag &&
        lhs.rigidSegmentIdx == rhs.rigidSegmentIdx;

    if (!areParametersEqual)
        return false;
    
    const bool isStipplePatternArrayEqual = (lhs.stipplePatternSize > 0) ? (std::memcmp(lhs.stipplePattern, rhs.stipplePattern, sizeof(uint32_t) * lhs.stipplePatternSize) == 0) : true;

    return isStipplePatternArrayEqual;
}
#endif

NBL_CONSTEXPR uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
NBL_CONSTEXPR uint32_t AlphaBits = 32u - MainObjectIdxBits;
NBL_CONSTEXPR uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
NBL_CONSTEXPR uint32_t InvalidStyleIdx = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;
NBL_CONSTEXPR uint64_t InvalidClipProjectionAddress = nbl::hlsl::numeric_limits<uint64_t>::max;
NBL_CONSTEXPR uint32_t InvalidTextureIdx = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR MajorAxis SelectedMajorAxis = MajorAxis::MAJOR_Y;
// TODO: get automatic version working on HLSL
NBL_CONSTEXPR MajorAxis SelectedMinorAxis = MajorAxis::MAJOR_X; //(MajorAxis) (1 - (uint32_t) SelectedMajorAxis);
NBL_CONSTEXPR float MSDFPixelRange = 4.0f;
NBL_CONSTEXPR float MSDFPixelRangeHalf = MSDFPixelRange / 2.0f;
NBL_CONSTEXPR float MSDFSize = 32.0f; 
NBL_CONSTEXPR uint32_t MSDFMips = 4; 
NBL_CONSTEXPR float HatchFillMSDFSceenSpaceSize = 8.0; 

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
    using float_t2 = vector<float_t, 2>;
    using float_t3 = vector<float_t, 3>;
    
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
    
    /* SHARED: ALL ObjectTypes */
    ObjectType getObjType() { return (ObjectType) data1.x; }
    uint getMainObjectIdx() { return data1.y; }
    
    void setObjType(ObjectType objType) { data1.x = (uint) objType; }
    void setMainObjectIdx(uint mainObjIdx) { data1.y = mainObjIdx; }
    
    /* SHARED: LINE + QUAD_BEZIER (Curve Outlines) */
    float getLineThickness() { return asfloat(data1.z); }
    float getPatternStretch() { return asfloat(data1.w); }

    void setLineThickness(float lineThickness) { data1.z = asuint(lineThickness); }
    void setPatternStretch(float stretch) { data1.w = asuint(stretch); }

    void setCurrentPhaseShift(float phaseShift)  { interp_data5.x = phaseShift; }
    float getCurrentPhaseShift() { return interp_data5.x; }

    void setCurrentWorldToScreenRatio(float worldToScreen) { interp_data5.y = worldToScreen; }
    float getCurrentWorldToScreenRatio() { return interp_data5.y; }
    
    /* LINE */
    float2 getLineStart() { return data2.xy; }
    float2 getLineEnd() { return data2.zw; }
    void setLineStart(float2 lineStart) { data2.xy = lineStart; }
    void setLineEnd(float2 lineEnd) { data2.zw = lineEnd; }
    
    /* QUAD_BEZIER */
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
    
    void setQuadraticPrecomputedArcLenData(nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator preCompData) 
    {
        data3.zw = float2(preCompData.lenA2, preCompData.AdotB);
        data4 = float4(preCompData.a, preCompData.b, preCompData.c, preCompData.b_over_4a);
    }
    nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator getQuadraticArcLengthCalculator()
    {
        return nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator::construct(data3.z, data3.w, data4.x, data4.y, data4.z, data4.w);
    }
    
    /* CURVE_BOX */
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
    float getMinorBBoxUV() { return interp_data5.x; };
    void setMinorBBoxUV(float minorBBoxUV) { interp_data5.x = minorBBoxUV; }
    float getMajorBBoxUV() { return interp_data5.y; };
    void setMajorBBoxUV(float majorBBoxUV) { interp_data5.y = majorBBoxUV; }

    float2 getCurveBoxScreenSpaceSize() { return asfloat(data1.zw); }
    void setCurveBoxScreenSpaceSize(float2 aabbSize) { data1.zw = asuint(aabbSize); }
    
    /* POLYLINE_CONNECTOR */
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
    
    /* FONT_GLYPH */
    float2 getFontGlyphUV() { return interp_data5.xy; }
    uint32_t getFontGlyphTextureId() { return asuint(data2.x); }
    float getFontGlyphPxRange() { return data2.y; }

    void setFontGlyphUV(float2 uv) { interp_data5.xy = uv; }
    void setFontGlyphTextureId(uint32_t textureId) { data2.x = asfloat(textureId); }
    void setFontGlyphPxRange(float glyphPxRange) { data2.y = glyphPxRange; }

    /* IMAGE */
    float2 getImageUV() { return interp_data5.xy; }
    uint32_t getImageTextureId() { return asuint(data2.x); }
    
    void setImageUV(float2 uv) { interp_data5.xy = uv; }
    void setImageTextureId(uint32_t textureId) { data2.x = asfloat(textureId); }
};

// Set 0 - Scene Data and Globals, buffer bindings don't change the buffers only get updated
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
[[vk::binding(1, 0)]] StructuredBuffer<DrawObject> drawObjects : register(t0);
[[vk::binding(2, 0)]] StructuredBuffer<MainObject> mainObjects : register(t1);
[[vk::binding(3, 0)]] StructuredBuffer<LineStyle> lineStyles : register(t2);

[[vk::combinedImageSampler]][[vk::binding(4, 0)]] Texture2DArray<float3> msdfTextures : register(t3);
[[vk::combinedImageSampler]][[vk::binding(4, 0)]] SamplerState msdfSampler : register(s3);

[[vk::binding(5, 0)]] SamplerState textureSampler : register(s4);
[[vk::binding(6, 0)]] Texture2D textures[128] : register(t4);

// Set 1 - Window dependant data which has higher update frequency due to multiple windows and resize need image recreation and descriptor writes
[[vk::binding(0, 1)]] globallycoherent RWTexture2D<uint> pseudoStencil : register(u0);
[[vk::binding(1, 1)]] globallycoherent RWTexture2D<uint> colorStorage : register(u1);

#endif

#endif
