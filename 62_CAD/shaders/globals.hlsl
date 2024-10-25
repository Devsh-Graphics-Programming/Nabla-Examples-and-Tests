#ifndef _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_
#define _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>
#include <nbl/builtin/hlsl/portable/vector_t.hlsl>
#include <nbl/builtin/hlsl/portable/matrix_t.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>
#endif

using namespace nbl::hlsl;

// because we can't use jit/device_capabilities.hlsl in c++ code
#ifdef __HLSL_VERSION
using pfloat64_t = portable_float64_t<jit::device_capabilities>;
using pfloat64_t2 = portable_float64_t2<jit::device_capabilities>;
using pfloat64_t3 = portable_float64_t3<jit::device_capabilities>;
using pfloat64_t3x3 = portable_float64_t3x3<jit::device_capabilities>;
#else
using pfloat64_t = float64_t;
using pfloat64_t2 = nbl::hlsl::vector<float64_t, 2>;
using pfloat64_t3 = nbl::hlsl::vector<float64_t, 3>;
using pfloat64_t3x3 = portable_float64_t3x3<>;
#endif

// TODO: Compute this in a compute shader from the world counterparts
//      because this struct includes NDC coordinates, the values will change based camera zoom and move
//      of course we could have the clip values to be in world units and also the matrix to transform to world instead of ndc but that requires extra computations(matrix multiplications) per vertex
struct ClipProjectionData
{
    pfloat64_t3x3 projectionToNDC; // 72 -> because we use scalar_layout
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
    pfloat64_t screenToWorldRatio; // 96
    pfloat64_t worldToScreenRatio; // 100
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

// TODO[Przemek]: remove `#ifdef __HLSL_VERSION` and mul shouldn't use jit::device_caps, preferably do this with nbl::hlsl::mul  instead of portableMul64
#ifdef __HLSL_VERSION
pfloat64_t2 transformPointNdc(pfloat64_t3x3 transformation, pfloat64_t2 point2d)
{
    pfloat64_t3 point3d;
    point3d.x = point2d.x;
    point3d.y = point2d.y;
    point3d.z = _static_cast < pfloat64_t > (1.0f);

    pfloat64_t3 transformationResult = portableMul64 < pfloat64_t3x3, pfloat64_t3, jit::
    device_capabilities > (transformation, point3d);
    pfloat64_t2 output;
    output.x = transformationResult.x;
    output.y = transformationResult.y;

    return output;
}
pfloat64_t2 transformVectorNdc(pfloat64_t3x3 transformation, pfloat64_t2 vector2d)
{
    pfloat64_t3 vector3d;
    vector3d.x = vector2d.x;
    vector3d.y = vector2d.y;
    vector3d.z = _static_cast < pfloat64_t > (0.0f);

    pfloat64_t3 transformationResult = portableMul64 < pfloat64_t3x3, pfloat64_t3, jit::
    device_capabilities > (transformation, vector3d);
    pfloat64_t2 output;
    output.x = transformationResult.x;
    output.y = transformationResult.y;

    return output;
}
#endif


#endif
