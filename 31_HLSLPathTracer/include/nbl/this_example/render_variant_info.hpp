#ifndef __NBL_THIS_EXAMPLE_RENDER_VARIANT_INFO_HPP_INCLUDED__
#define __NBL_THIS_EXAMPLE_RENDER_VARIANT_INFO_HPP_INCLUDED__

#include "nbl/this_example/render_variant_enums.hlsl"

namespace nbl::this_example
{
struct SRenderVariantInfo
{
	E_POLYGON_METHOD effectiveMethod;
	E_POLYGON_METHOD pipelineMethod;
	const char* entryPoint;
};

static constexpr const char* getDefaultRenderEntryPointName(const bool persistentWorkGroups)
{
	return persistentWorkGroups ? "mainPersistent" : "main";
}

static constexpr SRenderVariantInfo getRenderVariantInfo(const E_LIGHT_GEOMETRY geometry, const bool persistentWorkGroups, const E_POLYGON_METHOD requestedMethod)
{
	const char* const defaultEntryPoint = getDefaultRenderEntryPointName(persistentWorkGroups);
	switch (geometry)
	{
	case ELG_SPHERE:
		return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, defaultEntryPoint };
	case ELG_TRIANGLE:
		switch (requestedMethod)
		{
		case EPM_AREA:
			return { EPM_AREA, EPM_AREA, persistentWorkGroups ? "mainPersistentArea" : "mainArea" };
		case EPM_SOLID_ANGLE:
			return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, persistentWorkGroups ? "mainPersistentSolidAngle" : "mainSolidAngle" };
		case EPM_PROJECTED_SOLID_ANGLE:
		default:
			return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
		}
	case ELG_RECTANGLE:
		return { EPM_SOLID_ANGLE, EPM_SOLID_ANGLE, defaultEntryPoint };
	default:
		return { EPM_PROJECTED_SOLID_ANGLE, EPM_PROJECTED_SOLID_ANGLE, defaultEntryPoint };
	}
}
}

#endif
