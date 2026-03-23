#ifndef _PT_COMPUTE_TRIANGLE_METHODS_SHARED_INCLUDED_
#define _PT_COMPUTE_TRIANGLE_METHODS_SHARED_INCLUDED_

#ifndef PT_VARIANT_USE_RWMC
#error PT_VARIANT_USE_RWMC must be defined before including pt.compute.triangle.methods.shared.hlsl
#endif

#ifndef PT_VARIANT_ENABLE_LINEAR
#error PT_VARIANT_ENABLE_LINEAR must be defined before including pt.compute.triangle.methods.shared.hlsl
#endif

#ifndef PT_VARIANT_ENABLE_PERSISTENT
#error PT_VARIANT_ENABLE_PERSISTENT must be defined before including pt.compute.triangle.methods.shared.hlsl
#endif

#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#define PATH_TRACER_ENABLE_LINEAR PT_VARIANT_ENABLE_LINEAR
#define PATH_TRACER_ENABLE_PERSISTENT PT_VARIANT_ENABLE_PERSISTENT
#include "compute.render.common.hlsl"
#include "scene_triangle_light.hlsl"
#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#include "compute_render_scene_impl.hlsl"

#if PT_VARIANT_ENABLE_LINEAR
#define PATH_TRACER_ENTRYPOINT_NAME main
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_APPROX_PROJECTED_SOLID_ANGLE
#include "compute.render.linear.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainArea
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_AREA
#include "compute.render.linear.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainSolidAngle
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_SOLID_ANGLE
#include "compute.render.linear.entrypoints.hlsl"
#endif

#if PT_VARIANT_ENABLE_PERSISTENT
#define PATH_TRACER_ENTRYPOINT_NAME mainPersistent
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_APPROX_PROJECTED_SOLID_ANGLE
#include "compute.render.persistent.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainPersistentArea
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_AREA
#include "compute.render.persistent.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainPersistentSolidAngle
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_SOLID_ANGLE
#include "compute.render.persistent.entrypoints.hlsl"
#endif

#undef PATH_TRACER_ENABLE_PERSISTENT
#undef PATH_TRACER_ENABLE_LINEAR
#undef PT_VARIANT_ENABLE_PERSISTENT
#undef PT_VARIANT_ENABLE_LINEAR
#undef PT_VARIANT_USE_RWMC

#endif
