#ifndef __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__

#include <nabla.h>

#include <bitset>

// common api
#include "camera/CFPSCamera.hpp"
#include "SimpleWindowedApplication.hpp"
#include "InputSystem.hpp"

#include "camera/CCubeProjection.hpp"
#include "camera/CLinearProjection.hpp"
#include "camera/CPlanarProjection.hpp"

// the example's headers
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"
#include "CGeomtryCreatorScene.hpp"

using namespace nbl;
using namespace core;
using namespace hlsl;
using namespace system;
using namespace asset;
using namespace ui;
using namespace video;
using namespace scene;
using namespace geometrycreator;

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__