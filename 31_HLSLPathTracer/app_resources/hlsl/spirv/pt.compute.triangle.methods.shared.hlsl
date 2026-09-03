#pragma once

#if !defined(PT_VARIANT_USE_RWMC)
#error Missing triangle method compile options
#endif


#if PT_VARIANT_USE_RWMC
#define PATH_TRACER_VARIANT_USE_RWMC true
#else
#define PATH_TRACER_VARIANT_USE_RWMC false
#endif

#define PATH_TRACER_VARIANT_GEOMETRY ELG_TRIANGLE
#define PATH_TRACER_VARIANT_POLYGON_METHOD EPM_PROJECTED_SOLID_ANGLE
#define PATH_TRACER_VARIANT_ENTRYPOINT_POLYGON_METHOD PPM_APPROX_PROJECTED_SOLID_ANGLE
#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC

#include "compute.render.common.hlsl"
#include "nbl/this_example/render_variant_config.hlsl"
#include "scene_triangle_light.hlsl"
#include "compute_render_scene_impl.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainPersistent
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_APPROX_PROJECTED_SOLID_ANGLE
#include "compute.render.persistent.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainPersistentArea
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_AREA
#include "compute.render.persistent.entrypoints.hlsl"

#define PATH_TRACER_ENTRYPOINT_NAME mainPersistentSolidAngle
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD PPM_SOLID_ANGLE
#include "compute.render.persistent.entrypoints.hlsl"

#undef PATH_TRACER_VARIANT_ENTRYPOINT_POLYGON_METHOD
#undef PATH_TRACER_VARIANT_POLYGON_METHOD
#undef PATH_TRACER_VARIANT_GEOMETRY
#undef PATH_TRACER_VARIANT_USE_RWMC
#undef PATH_TRACER_USE_RWMC
