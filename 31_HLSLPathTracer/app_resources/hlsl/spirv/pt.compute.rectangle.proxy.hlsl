#ifndef _PT_COMPUTE_RECTANGLE_INCLUDED_
#define _PT_COMPUTE_RECTANGLE_INCLUDED_
#define PATH_TRACER_USE_RWMC 0
#include "compute.render.common.hlsl"
#include "scene_rectangle_light.hlsl"
#include "compute_render_scene_impl.hlsl"
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#include "compute.render.linear.entrypoints.hlsl"
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#include "compute.render.persistent.entrypoints.hlsl"
#endif
