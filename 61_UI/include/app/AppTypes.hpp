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

struct SWindowControlBinding final
{
	nbl::core::smart_refctd_ptr<IGPUFramebuffer> sceneFramebuffer;
	nbl::core::smart_refctd_ptr<IGPUImageView> sceneColorView;
	nbl::core::smart_refctd_ptr<IGPUImageView> sceneDepthView;
	float32_t3x4 viewMatrix = float32_t3x4(1.f);
	float32_t4x4 projectionMatrix = float32_t4x4(1.f);
	float32_t4x4 viewProjMatrix = float32_t4x4(1.f);

	uint32_t activePlanarIx = 0u;
	bool allowGizmoAxesToFlip = false;
	bool enableDebugGridDraw = true;
	bool isOrthographicProjection = false;
	float aspectRatio = 16.f / 9.f;
	bool leftHandedProjection = true;
	CGimbalInputBinder inputBinding;

	std::optional<uint32_t> boundProjectionIx = std::nullopt;
	std::optional<uint32_t> lastBoundPerspectivePresetProjectionIx = std::nullopt;
	std::optional<uint32_t> lastBoundOrthoPresetProjectionIx = std::nullopt;
	std::optional<uint32_t> inputBindingProjectionIx = std::nullopt;
	uint32_t inputBindingPlanarIx = std::numeric_limits<uint32_t>::max();

	inline void pickDefaultProjections(const planar_projections_range_t& projections)
	{
		auto init = [&](std::optional<uint32_t>& presetix, IPlanarProjection::CProjection::ProjectionType requestedType) -> void
		{
			for (uint32_t i = 0u; i < projections.size(); ++i)
			{
				const auto& params = projections[i].getParameters();
				if (params.m_type == requestedType)
				{
					presetix = i;
					break;
				}
			}

			assert(presetix.has_value());
		};

		init(lastBoundPerspectivePresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Perspective);
		init(lastBoundOrthoPresetProjectionIx = std::nullopt, IPlanarProjection::CProjection::Orthographic);
		boundProjectionIx = lastBoundPerspectivePresetProjectionIx.value();
		inputBindingProjectionIx = std::nullopt;
		inputBindingPlanarIx = std::numeric_limits<uint32_t>::max();
	}
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
