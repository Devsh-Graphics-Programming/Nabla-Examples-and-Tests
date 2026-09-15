#ifndef _NBL_THIS_EXAMPLE_RENDER_VARIANT_INFO_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_RENDER_VARIANT_INFO_HPP_INCLUDED_


#include "nbl/this_example/render_variant_enums.hlsl"


namespace nbl::this_example
{
struct SRenderVariantInfo
{
	E_POLYGON_METHOD effectiveMethod;
	E_POLYGON_METHOD pipelineMethod;
	const char* entryPoint;
};

static constexpr SRenderVariantInfo getRenderVariantInfo(const E_LIGHT_GEOMETRY geometry, const E_POLYGON_METHOD requestedMethod)
{
	const char* const defaultEntryPoint = "mainPersistent";
	switch (geometry)
	{
	case ELG_SPHERE:
		return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, defaultEntryPoint };
	case ELG_TRIANGLE:
		switch (requestedMethod)
		{
		case EPM_AREA:
			return { EPM_AREA, EPM_AREA, "mainPersistentArea" };
		case EPM_SOLID_ANGLE:
			return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, "mainPersistentSolidAngle" };
		case EPM_PROJECTED_SOLID_ANGLE:
		default:
			return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
		}
	case ELG_RECTANGLE:
		switch (requestedMethod)
		{
		case EPM_AREA:
			return { EPM_AREA, EPM_AREA, "mainPersistentArea" };
		case EPM_SOLID_ANGLE:
			return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, "mainPersistentSolidAngle" };
		case EPM_PROJECTED_SOLID_ANGLE:
		default:
			return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
		}
	default:
		return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
	}
}
}

#endif
