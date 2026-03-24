#ifndef _PT_COMPUTE_RUNTIME_UNIFORM_SHARED_INCLUDED_
#define _PT_COMPUTE_RUNTIME_UNIFORM_SHARED_INCLUDED_

#ifndef PT_VARIANT_USE_RWMC
#error PT_VARIANT_USE_RWMC must be defined before including pt.compute.runtime_uniform.shared.hlsl
#endif

#ifndef PT_VARIANT_SCENE_HEADER
#error PT_VARIANT_SCENE_HEADER must be defined before including pt.compute.runtime_uniform.shared.hlsl
#endif

#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#include "compute.render.common.hlsl"
#include PT_VARIANT_SCENE_HEADER
#include "compute_render_scene_impl.hlsl"
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#include "compute.render.linear.entrypoints.hlsl"
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#include "compute.render.persistent.entrypoints.hlsl"

#undef PT_VARIANT_SCENE_HEADER
#undef PT_VARIANT_USE_RWMC

#endif
