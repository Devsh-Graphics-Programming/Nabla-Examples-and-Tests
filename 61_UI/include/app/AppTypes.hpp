#ifndef _NBL_THIS_EXAMPLE_APP_TYPES_HPP_
#define _NBL_THIS_EXAMPLE_APP_TYPES_HPP_

#include "common.hpp"

using planar_projections_range_t = std::vector<IPlanarProjection::CProjection>;
using planar_projection_t = CPlanarProjection<planar_projections_range_t>;

struct ImGuizmoPlanarM16InOut
{
	float32_t4x4 view, projection;
};

struct ImGuizmoModelM16InOut
{
	float32_t4x4 inTRS, outTRS, outDeltaTRS;
};

constexpr IGPUImage::SSubresourceRange TripleBufferUsedSubresourceRange =
{
	.aspectMask = IGPUImage::EAF_COLOR_BIT,
	.baseMipLevel = 0,
	.levelCount = 1,
	.baseArrayLayer = 0,
	.layerCount = 1
};

#endif // _NBL_THIS_EXAMPLE_APP_TYPES_HPP_
