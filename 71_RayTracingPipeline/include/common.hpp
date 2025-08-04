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

struct ReferenceObjectCpu
{
	core::smart_refctd_ptr<ICPUPolygonGeometry> data;
	Material material;
  core::matrix3x4SIMD transform;

};

}

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
