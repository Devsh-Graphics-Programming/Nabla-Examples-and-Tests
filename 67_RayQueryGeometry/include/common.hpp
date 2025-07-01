#ifndef _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_

#include "nbl/examples/examples.hpp"

using namespace nbl;
using namespace nbl::core;
using namespace nbl::hlsl;
using namespace nbl::system;
using namespace nbl::asset;
using namespace nbl::ui;
using namespace nbl::video;
using namespace nbl::application_templates;
using namespace nbl::examples;

#include "app_resources/common.hlsl"

namespace nbl::scene
{
enum ObjectType : uint8_t
{
	OT_CUBE,
	OT_SPHERE,
	OT_CYLINDER,
	OT_RECTANGLE,
	OT_CONE,
	OT_ICOSPHERE,

	OT_COUNT,
	OT_UNKNOWN = std::numeric_limits<uint8_t>::max()
};

static constexpr uint32_t s_smoothNormals[OT_COUNT] = { 0, 1, 1, 0, 1, 1 };

struct ObjectMeta
{
	ObjectType type = OT_UNKNOWN;
	std::string_view name = "Unknown";
};

struct ObjectDrawHookCpu
{
	nbl::core::matrix3x4SIMD model;
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
	core::smart_refctd_ptr<ICPUPolygonGeometry> data;
};

struct ReferenceObjectGpu
{
	struct Bindings
	{
		nbl::asset::SBufferBinding<IGPUBuffer> vertex, index;
	};

	ObjectMeta meta;
	Bindings bindings;
	uint32_t vertexStride;
	nbl::asset::E_INDEX_TYPE indexType = nbl::asset::E_INDEX_TYPE::EIT_UNKNOWN;
	uint32_t indexCount = {};

	const bool useIndex() const
	{
		return bindings.index.buffer && (indexType != E_INDEX_TYPE::EIT_UNKNOWN);
	}
};
}


#endif // _NBL_THIS_EXAMPLE_COMMON_H_INCLUDED_