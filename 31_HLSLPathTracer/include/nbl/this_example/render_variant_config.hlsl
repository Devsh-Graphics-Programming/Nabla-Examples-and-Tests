#ifndef _NBL_THIS_EXAMPLE_RENDER_VARIANT_CONFIG_HLSL_INCLUDED_
#define _NBL_THIS_EXAMPLE_RENDER_VARIANT_CONFIG_HLSL_INCLUDED_


#include "nbl/this_example/render_variant_enums.hlsl"


// TODO: this file has no business being in `nbl/this_example` it depends on preprocessor macros, can't be part of PCH, should only be in HLSL
#ifndef PATH_TRACER_VARIANT_GEOMETRY
#error PATH_TRACER_VARIANT_GEOMETRY must be defined before including render_variant_config.hlsl
#endif

#ifndef PATH_TRACER_VARIANT_POLYGON_METHOD
#error PATH_TRACER_VARIANT_POLYGON_METHOD must be defined before including render_variant_config.hlsl
#endif

#ifndef PATH_TRACER_VARIANT_ENTRYPOINT_POLYGON_METHOD
#error PATH_TRACER_VARIANT_ENTRYPOINT_POLYGON_METHOD must be defined before including render_variant_config.hlsl
#endif

#ifndef PATH_TRACER_VARIANT_USE_RWMC
#error PATH_TRACER_VARIANT_USE_RWMC must be defined before including render_variant_config.hlsl
#endif

// TODO: Why isn't everything templated on this?
struct pathtracer_variant_config
{
	NBL_CONSTEXPR_STATIC_INLINE E_LIGHT_GEOMETRY Geometry = PATH_TRACER_VARIANT_GEOMETRY;
	NBL_CONSTEXPR_STATIC_INLINE E_POLYGON_METHOD PolygonMethod = PATH_TRACER_VARIANT_POLYGON_METHOD;
	NBL_CONSTEXPR_STATIC_INLINE NEEPolygonMethod EntryPointPolygonMethod = PATH_TRACER_VARIANT_ENTRYPOINT_POLYGON_METHOD;
	NBL_CONSTEXPR_STATIC_INLINE bool UseRWMC = PATH_TRACER_VARIANT_USE_RWMC;
};

#endif
