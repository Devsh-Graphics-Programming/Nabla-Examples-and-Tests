#ifndef _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_
#define _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_

#include <cstdint>
#include "nbl/ui/ICursorControl.h"
#include "nbl/ext/ImGui/ImGui.h"
#include "imgui/imgui_internal.h"
#include "imguizmo/ImGuizmo.h"

struct TransformRequestParams
{
	float camDistance = 8.f;
	uint8_t sceneTexDescIx = ~0;
	bool useWindow = true, editTransformDecomposition = false, enableViewManipulate = false;
};

nbl::hlsl::uint16_t2 EditTransform(float* cameraView, const float* cameraProjection, float* matrix, const TransformRequestParams& params);

#endif // _NBL_THIS_EXAMPLE_TRANSFORM_H_INCLUDED_
