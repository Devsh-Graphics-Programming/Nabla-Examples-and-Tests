#ifndef _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_
#define _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_

// TODO[Erfan]: Turn off in the future, but keep enabled to test
// #define NBL_FORCE_EMULATED_FLOAT_64

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>
#include <nbl/builtin/hlsl/portable/vector_t.hlsl>
#include <nbl/builtin/hlsl/portable/matrix_t.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/tgmath.hlsl>

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#endif

using namespace nbl::hlsl;

#ifdef __HLSL_VERSION
using pfloat64_t = portable_float64_t<DeviceConfigCaps>;
using pfloat64_t2 = portable_float64_t2<DeviceConfigCaps>;
using pfloat64_t3 = portable_float64_t3<DeviceConfigCaps>;
#else
using pfloat64_t = float64_t;
using pfloat64_t2 = nbl::hlsl::vector<float64_t, 2>;
using pfloat64_t3 = nbl::hlsl::vector<float64_t, 3>;
#endif

using pfloat64_t3x3 = portable_matrix_t3x3<pfloat64_t>;

struct PushConstants
{
    uint64_t triangleMeshVerticesBaseAddress;
    uint32_t triangleMeshMainObjectIndex;
    uint32_t isDTMRendering;
};

struct WorldClipRect
{
    pfloat64_t2 minClip; // min clip of a rect in worldspace coordinates of the original space (globals.defaultProjectionToNDC)
    pfloat64_t2 maxClip; // max clip of a rect in worldspace coordinates of the original space (globals.defaultProjectionToNDC)
};

struct Pointers
{
    uint64_t lineStyles;
    uint64_t dtmSettings;
    uint64_t customProjections;
    uint64_t customClipRects;
    uint64_t mainObjects;
    uint64_t drawObjects;
    uint64_t geometryBuffer;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Pointers) == 56u);
#endif

struct Globals
{
    Pointers pointers;
    pfloat64_t3x3 defaultProjectionToNDC;
    pfloat64_t3x3 screenToWorldScaleTransform; // Pre-multiply your transform with this to scale in screen space (e.g., scale 100.0 means 100 screen pixels).
    uint32_t2 resolution;
    float antiAliasingFactor;
    uint32_t miterLimit;
    uint32_t currentlyActiveMainObjectIndex; // for alpha resolve to skip resolving activeMainObjectIdx and prep it for next submit
    float32_t _padding;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Globals) == 224u);
#endif

#ifdef __HLSL_VERSION
pfloat64_t2 transformPointNdc(NBL_CONST_REF_ARG(pfloat64_t3x3) transformation, NBL_CONST_REF_ARG(pfloat64_t2) point2d)
{
    pfloat64_t3 point3d;
    point3d.x = point2d.x;
    point3d.y = point2d.y;
    point3d.z = _static_cast<pfloat64_t>(1.0f);

    pfloat64_t3 transformationResult = nbl::hlsl::mul(transformation, point3d);

    pfloat64_t2 output;
    output.x = transformationResult.x;
    output.y = transformationResult.y;

    return output;
}
pfloat64_t2 transformVectorNdc(NBL_CONST_REF_ARG(pfloat64_t3x3) transformation, NBL_CONST_REF_ARG(pfloat64_t2) vector2d)
{
    pfloat64_t3 vector3d;
    vector3d.x = vector2d.x;
    vector3d.y = vector2d.y;
    vector3d.z = _static_cast<pfloat64_t>(0.0f);

    pfloat64_t3 transformationResult = nbl::hlsl::mul(transformation, vector3d);
    pfloat64_t2 output;
    output.x = transformationResult.x;
    output.y = transformationResult.y;

    return output;
}
#endif

enum class MainObjectType : uint32_t
{
    NONE = 0u,
    POLYLINE,
    HATCH,
    TEXT,
    STATIC_IMAGE,
    DTM,
    GRID_DTM,
    STREAMED_IMAGE,
};

enum class ObjectType : uint32_t
{
    LINE = 0u,
    QUAD_BEZIER = 1u,
    CURVE_BOX = 2u,
    POLYLINE_CONNECTOR = 3u,
    FONT_GLYPH = 4u,
    STATIC_IMAGE = 5u,
    TRIANGLE_MESH = 6u,
    GRID_DTM = 7u,
    STREAMED_IMAGE = 8u,
};

enum class MajorAxis : uint32_t
{
    MAJOR_X = 0u,
    MAJOR_Y = 1u,
};

enum TransformationType 
{
    TT_NORMAL = 0,
    TT_FIXED_SCREENSPACE_SIZE
};


// Consists of multiple DrawObjects
// [IDEA]: In GPU-driven rendering, to save mem for MainObject data fetching: many of these can be shared amongst different main objects, we could find these styles, settings, etc indices with upper_bound
// [TODO]: pack indices and members of mainObject and DrawObject + enforce max size for autosubmit --> but do it only after the mainobject definition is finalized in gpu-driven rendering work
struct MainObject
{
    uint32_t styleIdx;
    uint32_t dtmSettingsIdx;
    uint32_t customProjectionIndex;
    uint32_t customClipRectIndex;
    uint32_t transformationType; // todo pack later, it's just 2 possible values atm
};

struct DrawObject
{
    uint32_t type_subsectionIdx; // packed two uint16 into uint32
    uint32_t mainObjIndex;
    uint64_t geometryAddress;
};

// Goes into geometry buffer, needs to be aligned by 8
struct LinePointInfo
{
    pfloat64_t2 p;
    float32_t phaseShift;
    float32_t stretchValue;
};

// Goes into geometry buffer, needs to be aligned by 8
struct QuadraticBezierInfo
{
    nbl::hlsl::shapes::QuadraticBezier<pfloat64_t> shape; // 48bytes = 3 (control points) x 16 (float64_t2)
    float32_t phaseShift;
    float32_t stretchValue;
};
#ifndef __HLSL_VERSION
static_assert(offsetof(QuadraticBezierInfo, phaseShift) == 48u);
#endif

// Goes into geometry buffer, needs to be aligned by 8
struct GlyphInfo
{
    pfloat64_t2 topLeft; // 2 * 8 = 16 bytes
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (32)
    // unorm8 minU;
    // unorm8 minV;
    // uint16 textureId;
    uint32_t minUV_textureID_packed; // 4 bytes (36)
    
#ifndef __HLSL_VERSION
    GlyphInfo(pfloat64_t2  topLeft, float32_t2 dirU, float32_t aspectRatio, uint16_t textureId, float32_t2 minUV) :
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
        uint32_t uPacked = (uint32_t)(nbl::hlsl::clamp(minUV.x, 0.0f, 1.0f) * 255.0f);
        uint32_t vPacked = (uint32_t)(nbl::hlsl::clamp(minUV.y, 0.0f, 1.0f) * 255.0f);
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

// Goes into geometry buffer, needs to be aligned by 8
struct ImageObjectInfo
{
    pfloat64_t2 topLeft; // 2 * 8 = 16 bytes (16)
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (28)
    uint32_t textureID; // 4 bytes (32)
};

// Goes into geometry buffer, needs to be aligned by 8
// Currently a simple OBB like ImageObject, but later will be fullscreen with additional info about UV offset for toroidal(mirror) addressing
struct GeoreferencedImageInfo
{
    pfloat64_t2 topLeft; // 2 * 8 = 16 bytes (16)
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (28)
    uint32_t textureID; // 4 bytes (32)
};

// Goes into geometry buffer, needs to be aligned by 8
struct GridDTMInfo
{
    pfloat64_t2 topLeft; // 2 * 8 = 16 bytes (16)
    pfloat64_t2 worldSpaceExtents; // 16 bytes (32)
    uint32_t textureID; // 4 bytes (36)
    float gridCellWidth; // 4 bytes (40)
    float thicknessOfTheThickestLine; // 4 bytes (44)
    float _padding; // 4 bytes (48)
};

enum E_CELL_DIAGONAL : uint32_t
{
    TOP_LEFT_TO_BOTTOM_RIGHT = 0u,
    BOTTOM_LEFT_TO_TOP_RIGHT = 1u,
    INVALID = 2u
};

#ifndef __HLSL_VERSION

// sets last bit of data to 1 or 0 depending on diagonalMode
static void setDiagonalModeBit(float* data, E_CELL_DIAGONAL diagonalMode)
{
    if (diagonalMode == E_CELL_DIAGONAL::INVALID)
        return;

    uint32_t dataAsUint = reinterpret_cast<uint32_t&>(*data);
    constexpr uint32_t HEIGHT_VALUE_MASK = 0xFFFFFFFEu;
    dataAsUint &= HEIGHT_VALUE_MASK;
    dataAsUint |= static_cast<uint32_t>(diagonalMode);
    *data = reinterpret_cast<float&>(dataAsUint);

    uint32_t dataAsUintDbg = reinterpret_cast<uint32_t&>(*data);
}

#endif

// Top left corner holds diagonal mode info of a cell 
static E_CELL_DIAGONAL getDiagonalModeFromCellCornerData(uint32_t cellCornerData)
{
    return (cellCornerData & 0x1u) ? BOTTOM_LEFT_TO_TOP_RIGHT : TOP_LEFT_TO_BOTTOM_RIGHT;
}

static uint32_t packR11G11B10_UNORM(float32_t3 color)
{
    // Scale and convert to integers
    uint32_t r = (uint32_t)(nbl::hlsl::clamp(color.r, 0.0f, 1.0f) * 2047.0f + 0.5f); // 11 bits -> 2^11 - 1 = 2047
    uint32_t g = (uint32_t)(nbl::hlsl::clamp(color.g, 0.0f, 1.0f) * 2047.0f + 0.5f); // 11 bits -> 2^11 - 1 = 2047
    uint32_t b = (uint32_t)(nbl::hlsl::clamp(color.b, 0.0f, 1.0f) * 1023.0f + 0.5f); // 10 bits -> 2^10 - 1 = 1023

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
    pfloat64_t2 circleCenter;
    float32_t2 v; // the vector from circle center to the intersection of the line ends, it's normalized such that the radius of the circle is equal to 1
    float32_t cosAngleDifferenceHalf;
    float32_t _reserved_pad;
};

// NOTE: Don't attempt to pack curveMin/Max to uints because of limited range of values, we need the logarithmic precision of floats (more precision near 0)
// Goes into geometry buffer, needs to be aligned by 8
struct CurveBox
{
    // will get transformed in the vertex shader, and will be calculated on the cpu when generating these boxes
    pfloat64_t2 aabbMin; // 16
    pfloat64_t2 aabbMax; // 32 , TODO: we know it's a square/box -> we save 8 bytes if we needed to store extra data
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

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidRigidSegmentIndex = 0xffffffff;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float InvalidStyleStretchValue = nbl::hlsl::numeric_limits<float>::infinity;


// TODO[Przemek]: we will need something similar to LineStyles but related to heigh shading settings which is user customizable (like  stipple patterns) and requires upper_bound to figure out the color based on height value.
// We'll discuss that later or what it will be looking like and how it's gonna get passed to our shaders.

struct TriangleMeshVertex
{
    pfloat64_t2 pos;
    pfloat64_t height; // TODO: can be of type float32_t instead
};

// The color parameter is also used for styling non-curve objects such as text glyphs and hatches with solid color
struct LineStyle
{
    const static uint32_t StipplePatternMaxSize = 12u;

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

enum E_DTM_MODE
{
    OUTLINE         = 1 << 0,
    CONTOUR         = 1 << 1,
    HEIGHT_SHADING  = 1 << 2,
};

enum class E_HEIGHT_SHADING_MODE : uint32_t
{
    DISCRETE_VARIABLE_LENGTH_INTERVALS,
    DISCRETE_FIXED_LENGTH_INTERVALS,
    CONTINOUS_INTERVALS
};
    
struct DTMContourSettings
{
    uint32_t contourLineStyleIdx; // index into line styles
    float contourLinesStartHeight;
    float contourLinesEndHeight;
    float contourLinesHeightInterval;
};

struct DTMHeightShadingSettings
{
    const static uint32_t HeightColorMapMaxEntries = 16u;
    
    // height-color map
    float intervalLength;
	float intervalIndexToHeightMultiplier;
    int isCenteredShading;
    
    uint32_t heightColorEntryCount;
    float heightColorMapHeights[HeightColorMapMaxEntries];
    float32_t4 heightColorMapColors[HeightColorMapMaxEntries];
    
    E_HEIGHT_SHADING_MODE determineHeightShadingMode()
    {
        if (nbl::hlsl::isinf(intervalLength))
            return E_HEIGHT_SHADING_MODE::DISCRETE_VARIABLE_LENGTH_INTERVALS;
        if (intervalLength == 0.0f)
            return E_HEIGHT_SHADING_MODE::CONTINOUS_INTERVALS;
        return E_HEIGHT_SHADING_MODE::DISCRETE_FIXED_LENGTH_INTERVALS;
    }
};

// Documentation and explanation of variables in DTMSettingsInfo
struct DTMSettings
{
    const static uint32_t MaxContourSettings = 8u;

    uint32_t mode; // E_DTM_MODE
    
    // outline
    uint32_t outlineLineStyleIdx;

    // contour lines
    uint32_t contourSettingsCount;
    DTMContourSettings contourSettings[MaxContourSettings];

    // height shading
    DTMHeightShadingSettings heightShadingSettings;
    
    bool drawOutlineEnabled() NBL_CONST_MEMBER_FUNC { return  (mode & E_DTM_MODE::OUTLINE) != 0u; } 
    bool drawContourEnabled() NBL_CONST_MEMBER_FUNC { return (mode & E_DTM_MODE::CONTOUR) != 0u; }
    bool drawHeightShadingEnabled() NBL_CONST_MEMBER_FUNC { return (mode & E_DTM_MODE::HEIGHT_SHADING) != 0u; }
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

inline bool operator==(const DTMSettings& lhs, const DTMSettings& rhs)
{
    if (lhs.mode != rhs.mode)
        return false;

    if (lhs.drawOutlineEnabled())
    {
        if (lhs.outlineLineStyleIdx != rhs.outlineLineStyleIdx)
            return false;
    }

    if (lhs.drawContourEnabled())
    {
        if (lhs.contourSettingsCount != rhs.contourSettingsCount)
            return false;
        if (memcmp(lhs.contourSettings, rhs.contourSettings, lhs.contourSettingsCount * sizeof(DTMContourSettings)))
            return false;
    }

    if (lhs.drawHeightShadingEnabled())
    {
        if (lhs.heightShadingSettings.intervalLength != rhs.heightShadingSettings.intervalLength)
            return false;
        if (lhs.heightShadingSettings.intervalIndexToHeightMultiplier != rhs.heightShadingSettings.intervalIndexToHeightMultiplier)
            return false;
        if (lhs.heightShadingSettings.isCenteredShading != rhs.heightShadingSettings.isCenteredShading)
            return false;
        if (lhs.heightShadingSettings.heightColorEntryCount != rhs.heightShadingSettings.heightColorEntryCount)
            return false;
        
                
        if(memcmp(lhs.heightShadingSettings.heightColorMapHeights, rhs.heightShadingSettings.heightColorMapHeights, lhs.heightShadingSettings.heightColorEntryCount * sizeof(float)))
            return false;
        if(memcmp(lhs.heightShadingSettings.heightColorMapColors, rhs.heightShadingSettings.heightColorMapColors, lhs.heightShadingSettings.heightColorEntryCount * sizeof(float32_t4)))
            return false;
    }

    return true;
}
#endif

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t ImagesBindingArraySize = 128;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t AlphaBits = 32u - MainObjectIdxBits;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidStyleIdx = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidDTMSettingsIdx = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidCustomProjectionIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidCustomClipRectIndex = nbl::hlsl::numeric_limits<uint32_t>::max;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidTextureIndex = nbl::hlsl::numeric_limits<uint32_t>::max;

// Hatches
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR MajorAxis SelectedMajorAxis = MajorAxis::MAJOR_Y;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR MajorAxis SelectedMinorAxis = MajorAxis::MAJOR_X; //(MajorAxis) (1 - (uint32_t) SelectedMajorAxis);

// Text or MSDF Hatches
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float MSDFPixelRange = 4.0f;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float MSDFPixelRangeHalf = MSDFPixelRange / 2.0f;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float MSDFSize = 64.0f; 
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t MSDFMips = 4; 
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR float HatchFillMSDFSceenSpaceSize = 8.0; 

inline bool isInvalidGridDtmHeightValue(float value)
{
    return nbl::hlsl::isnan(value);
}

// Used in CPU-side only for now
struct OrientedBoundingBox2D
{
    pfloat64_t2 topLeft; // 2 * 8 = 16 bytes (16)
    float32_t2 dirU; // 2 * 4 = 8 bytes (24)
    float32_t aspectRatio; // 4 bytes (28)
};

#ifdef __HLSL_VERSION
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);

LineStyle loadLineStyle(const uint32_t index)
{
    return vk::RawBufferLoad<LineStyle>(globals.pointers.lineStyles + index * sizeof(LineStyle), 4u);
}
DTMSettings loadDTMSettings(const uint32_t index)
{
    return vk::RawBufferLoad<DTMSettings>(globals.pointers.dtmSettings + index * sizeof(DTMSettings), 4u);
}
pfloat64_t3x3 loadCustomProjection(const uint32_t index)
{
    return vk::RawBufferLoad<pfloat64_t3x3>(globals.pointers.customProjections + index * sizeof(pfloat64_t3x3), 8u);
}
WorldClipRect loadCustomClipRect(const uint32_t index)
{
    return vk::RawBufferLoad<WorldClipRect>(globals.pointers.customClipRects + index * sizeof(WorldClipRect), 8u);
}
MainObject loadMainObject(const uint32_t index)
{
    return vk::RawBufferLoad<MainObject>(globals.pointers.mainObjects + index * sizeof(MainObject), 4u);
}
DrawObject loadDrawObject(const uint32_t index)
{
    return vk::RawBufferLoad<DrawObject>(globals.pointers.drawObjects + index * sizeof(DrawObject), 8u);
}
#else
static_assert(alignof(LineStyle)==4u);
static_assert(alignof(DTMSettings)==4u);
static_assert(alignof(pfloat64_t3x3)==8u);
static_assert(alignof(WorldClipRect)==8u);
static_assert(alignof(MainObject)==4u);
static_assert(alignof(DrawObject)==8u);
#endif


#endif
