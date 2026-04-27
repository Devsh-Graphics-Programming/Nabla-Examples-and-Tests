#ifndef _NBL_THIS_EXAMPLE_RENDER_VARIANT_STRINGS_HPP_INCLUDED_
#define _NBL_THIS_EXAMPLE_RENDER_VARIANT_STRINGS_HPP_INCLUDED_


#include <array>
#include <string>

#include "nbl/system/to_string.h"
#include "nbl/this_example/render_variant_enums.hlsl"


namespace nbl::system::impl
{
template<>
struct to_string_helper<E_LIGHT_GEOMETRY>
{
	static inline std::string __call(const E_LIGHT_GEOMETRY value)
	{
		switch (value)
		{
		case ELG_SPHERE:
			return "ELG_SPHERE";
		case ELG_TRIANGLE:
			return "ELG_TRIANGLE";
		case ELG_RECTANGLE:
			return "ELG_RECTANGLE";
		default:
			return "ERROR (geometry)";
		}
	}
};

template<>
struct to_string_helper<E_POLYGON_METHOD>
{
	static inline std::string __call(const E_POLYGON_METHOD value)
	{
		switch (value)
		{
		case EPM_AREA:
			return "Area";
		case EPM_SOLID_ANGLE:
			return "Solid Angle";
		case EPM_PROJECTED_SOLID_ANGLE:
			return "Projected Solid Angle";
		default:
			return "ERROR (method)";
		}
	}
};
}

namespace nbl::this_example
{
inline const auto& getLightGeometryNameStorage()
{
	static const auto names = std::to_array<std::string>({
		system::to_string(ELG_SPHERE),
		system::to_string(ELG_TRIANGLE),
		system::to_string(ELG_RECTANGLE)
	});
	return names;
}

inline const auto& getLightGeometryNamePointers()
{
	static const auto ptrs = [] {
		std::array<const char*, ELG_COUNT> retval = {};
		const auto& names = getLightGeometryNameStorage();
		for (size_t i = 0u; i < names.size(); ++i)
			retval[i] = names[i].c_str();
		return retval;
	}();
	return ptrs;
}

inline const auto& getPolygonMethodNameStorage()
{
	static const auto names = std::to_array<std::string>({
		system::to_string(EPM_AREA),
		system::to_string(EPM_SOLID_ANGLE),
		system::to_string(EPM_PROJECTED_SOLID_ANGLE)
	});
	return names;
}

inline const auto& getPolygonMethodNamePointers()
{
	static const auto ptrs = [] {
		std::array<const char*, EPM_COUNT> retval = {};
		const auto& names = getPolygonMethodNameStorage();
		for (size_t i = 0u; i < names.size(); ++i)
			retval[i] = names[i].c_str();
		return retval;
	}();
	return ptrs;
}
}

#endif
