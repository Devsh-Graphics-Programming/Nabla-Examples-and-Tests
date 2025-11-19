#ifndef _RAYTRACE_COMMON_H_INCLUDED_
#define _RAYTRACE_COMMON_H_INCLUDED_


#include "common.h"


/**
Plan for lighting:

Path Guiding with Rejection Sampling
	Do path guiding with spatio-directional (directions are implicit from light IDs) acceleration structure, could be with complete disregard for light NEE.
	Obviously the budgets for directions are low, so we might need to only track important lights and group them. Should probably read the spatiotemporal reservoir sampling paper.

	Each light gets a computed OBB and we use spherical OBB sampling (not projected solid angle, but we could clip) to generate the samples.
	Then NEE does perfect spherical sampling of the bounding volume.
	
	The OBBs could be hierarchical, possibly.

	OPTIMIZATION: Could possibly shoot an AnyHit to the front of the convex hull volume, and then ClosestHit between the front and back.
	BRDF sampling just samples the BSDF analytically (or gives up and samples only the path-guiding AS), uses Closest Hit and proceeds classically.
	There's essentially 3 ways to generate samples: NEE with PGAS (discrete directions), NEE with PGAS (for all incoming lights), BSDF Analytical.
	PROS: Probably a much better sample generation strategy, might clean up a lot of noise.
	CONS: We don't know the point on the surface we are going to hit (could be any of multiple points for a concave light), so we cannot cast a fixed length ray.
	We need to cast a ray to the furthest back side of the Bounding Volume, and it cannot be an just an AnyHit ray, it needs to have a ClosestHit shader that will compare
	if the hit instanceID==lightGroupID. It can probably be optimized so that it uses a different shadow-only + light-compare SBT. So it may take a lot longer to compute a sample.
CONCLUSION:
	We'll either be generating samples:
		A) From PGAS CDF
			No special light structure, just PGAS + GAS.
		C) Spherical sampling of OBBs
			OBB List with a CDF for the whole list in PGAS, then analytical

	Do we have to do 3-way MIS?
**/


struct SLight
{
#ifdef __cplusplus
	SLight() : obb() {}
	SLight(const SLight& other) : obb(other.obb) {}
	SLight(const nbl::core::aabbox3df& bbox, const nbl::core::matrix3x4SIMD& tform) : SLight()
	{
		auto extent = bbox.getExtent();
		obb.setScale(nbl::core::vectorSIMDf(extent.X, extent.Y, extent.Z));
		obb.setTranslation(nbl::core::vectorSIMDf(bbox.MinEdge.X, bbox.MinEdge.Y, bbox.MinEdge.Z));

		obb = nbl::core::concatenateBFollowedByA(tform, obb);
	}

	inline SLight& operator=(SLight&& other) noexcept
	{
		std::swap(obb, other.obb);

		return *this;
	}

	// also known as an upper bound on lumens put into the scene
	inline float computeLuma(const nbl::core::vectorSIMDf& radiance) const
	{
		const nbl::core::vectorSIMDf rec709LumaCoeffs(0.2126f, 0.7152f, 0.0722f, 0.f);
		return nbl::core::dot(radiance, rec709LumaCoeffs).x;
	}
	// also known as an upper bound on lumens put into the scene
	inline float computeFluxBound(const float luma) const
	{
		const auto unitHemisphereArea = 2.f * nbl::core::PI<float>();

		const auto unitBoxScale = obb.getScale();
		const float obbArea = 2.f * (unitBoxScale.x * unitBoxScale.y + unitBoxScale.x * unitBoxScale.z + unitBoxScale.y * unitBoxScale.z);

		return luma * unitHemisphereArea * obbArea;
	}
#endif

	mat4x3 obb; // needs row_major qualifier
	/** TODO new and improved
	mat2x3 obb_base;
	uvec2 radianceRemainder;
	vec3 offset;
	float obb_height;
	**/
};



//
#include <nbl/builtin/glsl/re_weighted_monte_carlo/splatting.glsl>
#ifdef __cplusplus
struct alignas(16) StaticViewData_t
#else
struct StaticViewData_t
#endif
{
#ifdef __cplusplus
	uint16_t imageDimensions[2];
	uint8_t maxPathDepth;
	uint8_t noRussianRouletteDepth;
	uint16_t samplesPerPixelPerDispatch;
	uint32_t sampleSequenceStride : 31;
	uint32_t hideEnvmap : 1;
#else
	uint imageDimensions;
	uint maxPathDepth_noRussianRouletteDepth_samplesPerPixelPerDispatch;
	uint sampleSequenceStride_hideEnvmap;
#endif
	float envMapPDFNormalizationFactor;
	nbl_glsl_RWMC_CascadeParameters cascadeParams;
};
#ifndef __cplusplus
uvec2 getImageDimensions(in StaticViewData_t data)
{
	return uvec2(
		bitfieldExtract(data.imageDimensions, 0,16),
		bitfieldExtract(data.imageDimensions,16,16)
	);
}
#endif


struct RaytraceShaderCommonData_t
{
	float   rcpFramesDispatched;
	uint	frameLowDiscrepancySequenceShift;
	uint	pathDepth_rayCountWriteIx; // depth=0 if path tracing disabled
	float	textureFootprintFactor;
	// need to be at the end because of some PC -> OpenGL Uniform mapping bug
	// PERSPECTIVE
	// mat3(viewDirReconFactors)*vec3(uv,1) or hitPoint-viewDirReconFactors[3]
	// ORTHO
	// viewDirReconFactors[2]=V
	mat4x3	viewDirReconFactors;

#ifdef __cplusplus
	uint32_t getPathDepth() const
	{
		return nbl::core::bitfieldExtract(pathDepth_rayCountWriteIx,0,RAYCOUNT_SHIFT);
	}
	void setPathDepth(const uint32_t depth)
	{
		pathDepth_rayCountWriteIx = nbl::core::bitfieldInsert(pathDepth_rayCountWriteIx,depth,0,RAYCOUNT_SHIFT);
	}

	uint32_t getReadIndex() const
	{
		const uint32_t index = nbl::core::bitfieldExtract(pathDepth_rayCountWriteIx,RAYCOUNT_SHIFT,RAYCOUNT_N_BUFFERING_LOG2);
		if (index)
			return index-1;
		return RAYCOUNT_N_BUFFERING-1;
	}
	void advanceWriteIndex()
	{
		const uint32_t writeIx = nbl::core::bitfieldExtract(pathDepth_rayCountWriteIx,RAYCOUNT_SHIFT,RAYCOUNT_N_BUFFERING_LOG2);
		pathDepth_rayCountWriteIx = nbl::core::bitfieldInsert(pathDepth_rayCountWriteIx,writeIx+1,RAYCOUNT_SHIFT,RAYCOUNT_N_BUFFERING_LOG2);
	}
#endif
};

#include <nbl/builtin/glsl/re_weighted_monte_carlo/reweighting.glsl>
#endif