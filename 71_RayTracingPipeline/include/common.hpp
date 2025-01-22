#ifndef __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__

#include <nabla.h>
#include "nbl/asset/utils/CGeometryCreator.h"
#include "nbl/application_templates/MonoAssetManagerAndBuiltinResourceApplication.hpp"

#include "SimpleWindowedApplication.hpp"

#include "InputSystem.hpp"
#include "CEventCallback.hpp"

#include "CCamera.hpp"

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/asset/IRayTracingPipeline.h>

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;

#include "app_resources/common.hlsl"

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

static constexpr uint32_t s_smoothNormals[OT_COUNT] = { 0, 1, 1, 0, 0, 1, 1, 1 };

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

struct ReferenceObjectCpu
{
	ObjectMeta meta;
	nbl::asset::CGeometryCreator::return_type data;
	Material material;
  core::matrix3x4SIMD transform;
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
	Material material;
  core::matrix3x4SIMD transform;

	const bool useIndex() const
	{
		return bindings.index.buffer && (indexType != E_INDEX_TYPE::EIT_UNKNOWN);
	}
};
}

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
