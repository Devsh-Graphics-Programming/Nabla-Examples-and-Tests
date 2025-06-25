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

#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"

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

struct ReferenceObjectCpu
{
	ObjectMeta meta;
	core::smart_refctd_ptr<ICPUPolygonGeometry> data;
	Material material;
  core::matrix3x4SIMD transform;

};

}

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
