#ifndef _NBL_THIS_EXAMPLE_APP_RENDER_PASS_UTILITIES_HPP_
#define _NBL_THIS_EXAMPLE_APP_RENDER_PASS_UTILITIES_HPP_

#include "common.hpp"
#include "app/AppTypes.hpp"

inline asset::SViewport makeFramebufferViewport(const uint32_t width, const uint32_t height)
{
	asset::SViewport viewport = {};
	viewport.minDepth = SCameraAppFrameRuntimeDefaults::ViewportMinDepth;
	viewport.maxDepth = SCameraAppFrameRuntimeDefaults::ViewportMaxDepth;
	viewport.x = 0u;
	viewport.y = 0u;
	viewport.width = width;
	viewport.height = height;
	return viewport;
}

inline VkRect2D makeRenderArea(const uint32_t width, const uint32_t height)
{
	return {
		.offset = { 0, 0 },
		.extent = { width, height }
	};
}

inline float32_t4x4 buildInverseViewRotation(const float32_t3x4& viewMatrix)
{
	auto inverseViewRotation = hlsl::transpose(hlsl::CCameraMathUtilities::promoteAffine3x4To4x4(viewMatrix));
	const auto xyzMask = SCameraAppFrameRuntimeDefaults::InverseViewRotationXyzMask;
	inverseViewRotation[0] *= xyzMask;
	inverseViewRotation[1] *= xyzMask;
	inverseViewRotation[2] *= xyzMask;
	inverseViewRotation[3] = SCameraAppFrameRuntimeDefaults::InverseViewRotationHomogeneousRow;
	return inverseViewRotation;
}

#endif // _NBL_THIS_EXAMPLE_APP_RENDER_PASS_UTILITIES_HPP_
