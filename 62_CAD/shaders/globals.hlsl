#ifndef _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_
#define _CAD_EXAMPLE_GLOBALS_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

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

#endif
