#ifndef _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_
#define _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/glsl_compat/core.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
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
    double worldToScreenRatio; // 100
    uint32_t2 resolution; // 108
    float antiAliasingFactor; // 112
    float miterLimit; // 116
    float32_t2 _padding; // 128
};

#ifndef __HLSL_VERSION
static_assert(offsetof(Globals, defaultClipProjection) == 0u);
static_assert(offsetof(Globals, screenToWorldRatio) == 88u);
static_assert(offsetof(Globals, worldToScreenRatio) == 96u);
static_assert(offsetof(Globals, resolution) == 104u);
static_assert(offsetof(Globals, antiAliasingFactor) == 112u);
static_assert(offsetof(Globals, miterLimit) == 116u);
#endif

inline float64_t2 transformPointNdc(float64_t3x3 transformation, float64_t2 point2d)
{
    return mul(transformation, float64_t3(point2d, 1)).xy;
}
inline float64_t2 transformVectorNdc(float64_t3x3 transformation, float64_t2 vector2d)
{
    return mul(transformation, float64_t3(vector2d, 0)).xy;
}

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

#endif
