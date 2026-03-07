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
using pfloat64_t4x4 = portable_matrix_t4x4<pfloat64_t>;

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
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Pointers) == 24u);
#endif

struct Globals
{
    Pointers pointers;
    pfloat64_t4x4 defaultProjectionToNDC;
};
#ifndef __HLSL_VERSION
static_assert(sizeof(Globals) == 152u);
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

struct TriangleMeshVertex
{
    pfloat64_t3 pos;
};

#ifdef __HLSL_VERSION
[[vk::binding(0, 0)]] ConstantBuffer<Globals> globals : register(b0);
#else
static_assert(alignof(pfloat64_t3x3)==8u);
static_assert(alignof(MainObject)==4u);
static_assert(alignof(DrawObject)==8u);
#endif


#endif
