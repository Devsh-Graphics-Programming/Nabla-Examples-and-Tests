#pragma once

#if !defined(PT_VARIANT_USE_RWMC) || !defined(PT_VARIANT_SCENE_KIND) || !defined(PT_VARIANT_ENTRYPOINT_KIND) || !defined(PT_VARIANT_RUNTIME_POLYGON_METHOD)
#error Missing path tracer variant compile options
#endif

#define PT_VARIANT_SCENE_SPHERE 0
#define PT_VARIANT_SCENE_TRIANGLE 1
#define PT_VARIANT_SCENE_RECTANGLE 2
#define PT_VARIANT_ENTRYPOINT_RUNTIME_UNIFORM 0
#define PT_VARIANT_ENTRYPOINT_LINEAR 1
#define PT_VARIANT_ENTRYPOINT_PERSISTENT 2

#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#if PT_VARIANT_ENTRYPOINT_KIND == PT_VARIANT_ENTRYPOINT_RUNTIME_UNIFORM
#define PATH_TRACER_ENABLE_LINEAR 1
#define PATH_TRACER_ENABLE_PERSISTENT 1
#elif PT_VARIANT_ENTRYPOINT_KIND == PT_VARIANT_ENTRYPOINT_LINEAR
#define PATH_TRACER_ENABLE_LINEAR 1
#define PATH_TRACER_ENABLE_PERSISTENT 0
#elif PT_VARIANT_ENTRYPOINT_KIND == PT_VARIANT_ENTRYPOINT_PERSISTENT
#define PATH_TRACER_ENABLE_LINEAR 0
#define PATH_TRACER_ENABLE_PERSISTENT 1
#else
#error Unsupported PT_VARIANT_ENTRYPOINT_KIND
#endif

#include "compute.render.common.hlsl"
#if PT_VARIANT_SCENE_KIND == PT_VARIANT_SCENE_SPHERE
#include "scene_sphere_light.hlsl"
#elif PT_VARIANT_SCENE_KIND == PT_VARIANT_SCENE_TRIANGLE
#include "scene_triangle_light.hlsl"
#elif PT_VARIANT_SCENE_KIND == PT_VARIANT_SCENE_RECTANGLE
#include "scene_rectangle_light.hlsl"
#else
#error Unsupported PT_VARIANT_SCENE_KIND
#endif
#include "compute_render_scene_impl.hlsl"

#if PT_VARIANT_ENTRYPOINT_KIND == PT_VARIANT_ENTRYPOINT_RUNTIME_UNIFORM
#if PT_VARIANT_RUNTIME_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#endif
#include "compute.render.linear.entrypoints.hlsl"
#if PT_VARIANT_RUNTIME_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#endif
#include "compute.render.persistent.entrypoints.hlsl"
#elif PT_VARIANT_ENTRYPOINT_KIND == PT_VARIANT_ENTRYPOINT_LINEAR
#if PT_VARIANT_RUNTIME_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#endif
#include "compute.render.linear.entrypoints.hlsl"
#else
#if PT_VARIANT_RUNTIME_POLYGON_METHOD
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#endif
#include "compute.render.persistent.entrypoints.hlsl"
#endif
