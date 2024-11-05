#ifndef __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__

#include <nabla.h>
#include "nbl/asset/utils/CGeometryCreator.h"

#include "SimpleWindowedApplication.hpp"
#include "InputSystem.hpp"
#include "CEventCallback.hpp"

#include "CCamera.hpp"
#include "SBasicViewParameters.hlsl"

#include "geometry/creator/spirv/builtin/CArchive.h"
#include "geometry/creator/spirv/builtin/builtinResources.h"

#include "CGeomtryCreatorScene.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;
//using namespace geometrycreator;

namespace nbl::scene
{
enum ObjectType : uint8_t
{
	OT_CUBE,
	OT_SPHERE,
	OT_CYLINDER,
	OT_RECTANGLE,
	OT_DISK,
	OT_ARROW,
	OT_CONE,
	OT_ICOSPHERE,

	OT_COUNT,
	OT_UNKNOWN = std::numeric_limits<uint8_t>::max()
};

struct ObjectMeta
{
	ObjectType type = OT_UNKNOWN;
	std::string_view name = "Unknown";
};

struct ObjectDrawHookCpu
{
	nbl::core::matrix3x4SIMD model;
	nbl::asset::SBasicViewParameters viewParameters;
	ObjectMeta meta;
};

enum GeometryShader
{
	GP_BASIC = 0,
	GP_CONE,
	GP_ICO,

	GP_COUNT
};

struct ReferenceObjectCpu
{
	ObjectMeta meta;
	GeometryShader shadersType;
	nbl::asset::CGeometryCreator::return_type data;
};

struct ReferenceObjectGpu
{
	struct Bindings
	{
		nbl::asset::SBufferBinding<IGPUBuffer> vertex, index;
	};

	//nbl::core::smart_refctd_ptr<typename Types::graphics_pipeline_t> pipeline = nullptr;

	ObjectMeta meta;
	Bindings bindings;
	nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
	uint32_t indexCount = {};
};
}

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__