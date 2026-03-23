#ifndef _PT_COMPUTE_VARIANT_SHARED_INCLUDED_
#define _PT_COMPUTE_VARIANT_SHARED_INCLUDED_

#ifndef PT_VARIANT_USE_RWMC
#error PT_VARIANT_USE_RWMC must be defined before including pt.compute.variant.shared.hlsl
#endif

#ifndef PT_VARIANT_SCENE_HEADER
#error PT_VARIANT_SCENE_HEADER must be defined before including pt.compute.variant.shared.hlsl
#endif

#ifndef PT_VARIANT_ENTRYPOINT_HEADER
#error PT_VARIANT_ENTRYPOINT_HEADER must be defined before including pt.compute.variant.shared.hlsl
#endif

#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#ifdef PT_VARIANT_ENABLE_LINEAR
#define PATH_TRACER_ENABLE_LINEAR PT_VARIANT_ENABLE_LINEAR
#endif
#ifdef PT_VARIANT_ENABLE_PERSISTENT
#define PATH_TRACER_ENABLE_PERSISTENT PT_VARIANT_ENABLE_PERSISTENT
#endif
#include "compute.render.common.hlsl"
#include PT_VARIANT_SCENE_HEADER
#define PATH_TRACER_USE_RWMC PT_VARIANT_USE_RWMC
#include "compute_render_scene_impl.hlsl"
#include PT_VARIANT_ENTRYPOINT_HEADER

#undef PATH_TRACER_ENABLE_PERSISTENT
#undef PATH_TRACER_ENABLE_LINEAR
#undef PT_VARIANT_ENTRYPOINT_HEADER
#undef PT_VARIANT_SCENE_HEADER
#undef PT_VARIANT_USE_RWMC

#endif
