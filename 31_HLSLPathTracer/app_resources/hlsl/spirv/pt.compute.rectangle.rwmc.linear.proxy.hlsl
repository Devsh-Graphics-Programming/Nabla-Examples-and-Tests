#ifndef _PT_COMPUTE_RECTANGLE_RWMC_LINEAR_INCLUDED_
#define _PT_COMPUTE_RECTANGLE_RWMC_LINEAR_INCLUDED_

#define PT_VARIANT_USE_RWMC 1
#define PT_VARIANT_ENABLE_LINEAR 1
#define PT_VARIANT_ENABLE_PERSISTENT 0
#define PT_VARIANT_SCENE_HEADER "scene_rectangle_light.hlsl"
#define PT_VARIANT_ENTRYPOINT_HEADER "compute.render.linear.entrypoints.hlsl"
#define PATH_TRACER_ENTRYPOINT_POLYGON_METHOD ((NEEPolygonMethod)(pathtracer_render_variant::getRenderPushConstants().polygonMethod))
#include "pt.compute.variant.shared.hlsl"

#endif
