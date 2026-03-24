#ifndef _NBL_THIS_EXAMPLE_BUILTIN_BUILD_SPIRV_KEYS_INCLUDED_
#define _NBL_THIS_EXAMPLE_BUILTIN_BUILD_SPIRV_KEYS_INCLUDED_

#include "nabla.h"

namespace nbl::this_example::builtin::build {
	inline constexpr const char* get_spirv_config_prefix()
	{
#if defined(CMAKE_INTDIR)
		return CMAKE_INTDIR;
#elif defined(_NBL_DEBUG) || defined(_DEBUG)
		return "Debug";
#elif defined(_NBL_RELWITHDEBINFO)
		return "RelWithDebInfo";
#elif defined(NDEBUG)
		return "Release";
#else
#error Unable to resolve path tracer SPIR-V config directory
#endif
	}

	template<nbl::core::StringLiteral Key>
	inline const nbl::core::string get_spirv_key(const nbl::video::SPhysicalDeviceLimits& limits, const nbl::video::SPhysicalDeviceFeatures& features);

	template<nbl::core::StringLiteral Key>
	inline const nbl::core::string get_spirv_key(const nbl::video::ILogicalDevice* device)
	{
		return get_spirv_key<Key>(device->getPhysicalDevice()->getLimits(), device->getEnabledFeatures());
	}
}

#define NBL_PATH_TRACER_DEFINE_SPIRV_KEY(KEY_LITERAL, FILE_LITERAL) \
namespace nbl::this_example::builtin::build { \
	template<> \
	inline const nbl::core::string get_spirv_key<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE(KEY_LITERAL)> \
	(const nbl::video::SPhysicalDeviceLimits& limits, const nbl::video::SPhysicalDeviceFeatures& features) \
	{ \
		nbl::core::string retval = FILE_LITERAL; \
		retval += ".spv"; \
		return nbl::core::string(get_spirv_config_prefix()) + "/" + retval; \
	} \
}

NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.sphere", "pt.compute.sphere");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.sphere.rwmc", "pt.compute.sphere.rwmc");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle", "pt.compute.triangle");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle.rwmc", "pt.compute.triangle.rwmc");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle.linear", "pt.compute.triangle.linear");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle.persistent", "pt.compute.triangle.persistent");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle.rwmc.linear", "pt.compute.triangle.rwmc.linear");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.triangle.rwmc.persistent", "pt.compute.triangle.rwmc.persistent");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.rectangle", "pt.compute.rectangle");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.rectangle.rwmc", "pt.compute.rectangle.rwmc");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.rectangle.rwmc.linear", "pt.compute.rectangle.rwmc.linear");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.rectangle.rwmc.persistent", "pt.compute.rectangle.rwmc.persistent");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.compute.resolve", "pt.compute.resolve");
NBL_PATH_TRACER_DEFINE_SPIRV_KEY("pt.misc", "pt.misc");

#undef NBL_PATH_TRACER_DEFINE_SPIRV_KEY

#endif
