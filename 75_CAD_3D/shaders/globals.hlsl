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
using pfloat64_t4 = portable_float64_t4<DeviceConfigCaps>;
#else
using pfloat64_t = float64_t;
using pfloat64_t2 = nbl::hlsl::vector<float64_t, 2>;
using pfloat64_t3 = nbl::hlsl::vector<float64_t, 3>;
using pfloat64_t4 = nbl::hlsl::vector<float64_t, 4>;
#endif

using pfloat64_t3x3 = portable_matrix_t3x3<pfloat64_t>;
using pfloat64_t4x4 = portable_matrix_t4x4<pfloat64_t>;

enum class MainObjectType : uint32_t
{
    NONE = 0u,
    DTM,
};

struct MainObject
{
    uint32_t dtmSettingsIdx;
};

struct PushConstants
{
    uint64_t triangleMeshVerticesBaseAddress;
    uint32_t triangleMeshMainObjectIndex;
    pfloat64_t4x4 viewProjectionMatrix;
};

struct Pointers
{
    uint64_t mainObjects;
    uint64_t drawObjects;
    uint64_t geometryBuffer;
    uint64_t dtmSettings;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Pointers) == 32u);
#endif

struct Globals
{
    Pointers pointers;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Globals) == 32u);
#endif

struct DrawObject
{
    uint32_t type_subsectionIdx; // packed two uint16 into uint32
    uint32_t mainObjIndex;
    uint64_t geometryAddress;
};

struct TriangleMeshVertex
{
    pfloat64_t3 pos;
};

enum class E_HEIGHT_SHADING_MODE : uint32_t
{
    DISCRETE_VARIABLE_LENGTH_INTERVALS,
    DISCRETE_FIXED_LENGTH_INTERVALS,
    CONTINOUS_INTERVALS
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

struct DTMSettings
{
    // height shading
    DTMHeightShadingSettings heightShadingSettings;
};
    
#ifndef __HLSL_VERSION
inline bool operator==(const DTMSettings& lhs, const DTMSettings& rhs)
{

    if(true) //if (lhs.drawHeightShadingEnabled())
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

NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t MainObjectIdxBits = 24u; // It will be packed next to alpha in a texture
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t MaxIndexableMainObjects = (1u << MainObjectIdxBits) - 1u;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidMainObjectIdx = MaxIndexableMainObjects;
NBL_CONSTEXPR_INLINE_NSPC_SCOPE_VAR uint32_t InvalidDTMSettingsIdx = nbl::hlsl::numeric_limits<uint32_t>::max;

#ifdef __HLSL_VERSION
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
    
MainObject loadMainObject(const uint32_t index)
{
    return vk::RawBufferLoad<MainObject>(globals.pointers.mainObjects + index * sizeof(MainObject), 4u);
}
DTMSettings loadDTMSettings(const uint32_t index)
{
    return vk::RawBufferLoad<DTMSettings>(globals.pointers.dtmSettings + index * sizeof(DTMSettings), 4u);
}
    
#else
static_assert(alignof(MainObject)==4u);
static_assert(alignof(DTMSettings)==4u);
static_assert(alignof(pfloat64_t3x3)==8u);
static_assert(alignof(DrawObject)==8u);
#endif

#endif
