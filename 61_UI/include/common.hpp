#ifndef __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__
#define __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__

#include <nabla.h>

#include <bitset>

// common api
#include "camera/CFPSCamera.hpp"
#include "SimpleWindowedApplication.hpp"
#include "InputSystem.hpp"

#include "camera/ILinearProjection.hpp"
#include "camera/IQuadProjection.hpp"

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

using matrix_precision_t = float32_t;

#endif // __NBL_THIS_EXAMPLE_COMMON_H_INCLUDED__