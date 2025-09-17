#include "Images.h"

using namespace nbl::hlsl;

smart_refctd_ptr<GeoreferencedImageStreamingState> GeoreferencedImageStreamingState::create(GeoreferencedImageParams&& _georeferencedImageParams, uint32_t TileSize)
{
	smart_refctd_ptr<GeoreferencedImageStreamingState> retVal(new GeoreferencedImageStreamingState{});
	retVal->georeferencedImageParams = std::move(_georeferencedImageParams);
	//	1. Get the displacement (will be an offset vector in world coords and world units) from the `topLeft` corner of the image to the point
	//	2. Transform this displacement vector into the coordinates in the basis {dirU, dirV} (worldspace vectors that span the sides of the image).
	//	The composition of these matrices therefore transforms any point in worldspace into uv coordinates in imagespace
	//  To reduce code complexity, instead of computing the product of these matrices, since the first is a pure displacement matrix 
	//  (non-homogenous 2x2 upper left is identity matrix) and the other is a pure rotation matrix (2x2) we can just put them together
	//  by putting the rotation in the upper left 2x2 of the result and the post-rotated displacement in the upper right 2x1.
	//  The result is also 2x3 and not 3x3 because we can drop he homogenous since the displacement yields a vector

	// 2. Change of Basis. Since {dirU, dirV} are orthogonal, the matrix to change from world coords to `span{dirU, dirV}` coords has a quite nice expression
	//    Non-uniform scaling doesn't affect this, but this has to change if we allow for shearing (basis vectors stop being orthogonal)
	const float64_t2 dirU = retVal->georeferencedImageParams.worldspaceOBB.dirU;
	const float64_t2 dirV = float64_t2(dirU.y, -dirU.x) * float64_t(retVal->georeferencedImageParams.worldspaceOBB.aspectRatio);
	const float64_t dirULengthSquared = nbl::hlsl::dot(dirU, dirU);
	const float64_t dirVLengthSquared = nbl::hlsl::dot(dirV, dirV);
	const float64_t2 firstRow = dirU / dirULengthSquared;
	const float64_t2 secondRow = dirV / dirVLengthSquared;

	const float64_t2 displacement = -retVal->georeferencedImageParams.worldspaceOBB.topLeft;
	// This is the same as multiplying the change of basis matrix by the displacement vector
	const float64_t postRotatedShiftX = nbl::hlsl::dot(firstRow, displacement);
	const float64_t postRotatedShiftY = nbl::hlsl::dot(secondRow, displacement);

	// Put them all together
	retVal->world2UV = float64_t2x3(firstRow.x, firstRow.y, postRotatedShiftX, secondRow.x, secondRow.y, postRotatedShiftY);

	// Also set the maxMipLevel - to keep stuff simple, we don't consider having less than one tile per dimension
	// If you're zoomed out enough then at that point the whole image is just sampled as one tile along that dimension
	// In pathological cases, such as images that are way bigger on one side than the other, this could cause aliasing and slow down sampling if zoomed out too much. 
	// If we were ever to observe such pathological cases, then maybe we should consider doing something else here. For example, making the loader able to handle different tile lengths per dimension
	// (so for example a 128x64 tile) but again for now it should be left as-is.
	uint32_t2 maxMipLevels = nbl::hlsl::findMSB(nbl::hlsl::roundUpToPoT(retVal->georeferencedImageParams.imageExtents / TileSize));
	retVal->maxMipLevel = nbl::hlsl::min(maxMipLevels.x, maxMipLevels.y);

	retVal->fullImageTileLength = (retVal->georeferencedImageParams.imageExtents - 1u) / TileSize + 1u;

	return retVal;
}

void GeoreferencedImageStreamingState::ensureMappedRegionCoversViewport(const GeoreferencedImageTileRange& viewportTileRange)
{
	// A base mip level of x in the current mapped region means we can handle the viewport having mip level y, with x <= y < x + 1.0
	// without needing to remap the region. When the user starts zooming in or out and the mip level of the viewport falls outside this range, we have to remap
	// the mapped region.
	const bool mipBoundaryCrossed = viewportTileRange.baseMipLevel != currentMappedRegion.baseMipLevel;

	// If we moved a huge amount in any direction, no tiles will remain resident, so we simply reset state
	// This only need be evaluated if the mip boundary was not already crossed
	const bool relativeShiftTooBig = !mipBoundaryCrossed &&
		nbl::hlsl::any
		(
			nbl::hlsl::abs(int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegion.topLeftTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
		)
		|| nbl::hlsl::any
		(
			nbl::hlsl::abs(int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegion.bottomRightTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
		);

	// If there is no overlap between previous mapped region and the next, just reset everything
	if (mipBoundaryCrossed || relativeShiftTooBig)
		remapCurrentRegion(viewportTileRange);
	// Otherwise we can get away with (at worst) sliding the mapped region along the real image, preserving the residency of the tiles that overlap between previous mapped region and the next
	else
		slideCurrentRegion(viewportTileRange);
}

void GeoreferencedImageStreamingState::remapCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
{
	// Zoomed out
	if (viewportTileRange.baseMipLevel > currentMappedRegion.baseMipLevel)
	{
		// TODO: Here we would move some mip 1 tiles to mip 0 image to save the work of reuploading them, reflect that in the tracked tiles
	}
	// Zoomed in
	else if (viewportTileRange.baseMipLevel < currentMappedRegion.baseMipLevel)
	{
		// TODO: Here we would move some mip 0 tiles to mip 1 image to save the work of reuploading them, reflect that in the tracked tiles
	}
	currentMappedRegion = viewportTileRange;
	// We can expand the currentMappedRegion to make it as big as possible, at no extra cost since we only upload tiles on demand
	// Since we use toroidal updating it's kinda the same which way we expand the region. We first try to make the extent be `gpuImageSideLengthTiles`
	currentMappedRegion.bottomRightTile = currentMappedRegion.topLeftTile + uint32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles) - uint32_t2(1, 1);
	// This extension can cause the mapped region to fall out of bounds on border cases, therefore we clamp it and extend it in the other direction
	// by the amount of tiles we removed during clamping
	const uint32_t2 excessTiles = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegion.bottomRightTile) - int32_t2(getLastTileIndex(currentMappedRegion.baseMipLevel))));
	currentMappedRegion.bottomRightTile -= excessTiles;
	// Shifting of the topLeftTile could fall out of bounds in pathological cases or at very high mip levels (zooming out too much), so we shift if possible, otherwise set it to 0
	currentMappedRegion.topLeftTile = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegion.topLeftTile) - int32_t2(excessTiles)));

	// Mark all gpu tiles as dirty
	currentMappedRegionOccupancy.resize(gpuImageSideLengthTiles);
	for (auto i = 0u; i < gpuImageSideLengthTiles; i++)
	{
		currentMappedRegionOccupancy[i].clear();
		currentMappedRegionOccupancy[i].resize(gpuImageSideLengthTiles, false);
	}
	// Reset state for gpu image so that it starts loading tiles at top left. Not really necessary.
	gpuImageTopLeft = uint32_t2(0, 0);
}

void GeoreferencedImageStreamingState::slideCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
{
	// `topLeftShift` represents how many tiles up and to the left we have to move the mapped region to fit the viewport. 
	// First we compute a vector from the current mapped region's topleft to the viewport's topleft. If this vector is positive along a dimension it means
	// the viewport's topleft is to the right or below the current mapped region's topleft, so we don't have to shift the mapped region to the left/up in that case
	const int32_t2 topLeftShift = nbl::hlsl::min(int32_t2(0, 0), int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegion.topLeftTile));
	// `bottomRightShift` represents the same as above but in the other direction.
	const int32_t2 bottomRightShift = nbl::hlsl::max(int32_t2(0, 0), int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegion.bottomRightTile));

	// The following is not necessarily equal to `gpuImageSideLengthTiles` since there can be pathological cases, as explained in the remapping method
	const uint32_t2 mappedRegionDimensions = currentMappedRegion.bottomRightTile - currentMappedRegion.topLeftTile + 1u;
	const uint32_t2 gpuImageBottomRight = (gpuImageTopLeft + mappedRegionDimensions - 1u) % gpuImageSideLengthTiles;

	// Mark dropped tiles as dirty/non-resident
	if (topLeftShift.x < 0)
	{
		// Shift left
		const uint32_t tilesToFit = -topLeftShift.x;
		for (uint32_t tile = 0; tile < tilesToFit; tile++)
		{
			// Get actual tile index with wraparound
			uint32_t tileIdx = (gpuImageBottomRight.x + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
			currentMappedRegionOccupancy[tileIdx].clear();
			currentMappedRegionOccupancy[tileIdx].resize(gpuImageSideLengthTiles, false);
		}
	}
	else if (bottomRightShift.x > 0)
	{
		//Shift right
		const uint32_t tilesToFit = bottomRightShift.x;
		for (uint32_t tile = 0; tile < tilesToFit; tile++)
		{
			// Get actual tile index with wraparound
			uint32_t tileIdx = (tile + gpuImageTopLeft.x) % gpuImageSideLengthTiles;
			currentMappedRegionOccupancy[tileIdx].clear();
			currentMappedRegionOccupancy[tileIdx].resize(gpuImageSideLengthTiles, false);
		}
	}

	if (topLeftShift.y < 0)
	{
		// Shift up
		const uint32_t tilesToFit = -topLeftShift.y;
		for (uint32_t tile = 0; tile < tilesToFit; tile++)
		{
			// Get actual tile index with wraparound
			uint32_t tileIdx = (gpuImageBottomRight.y + (gpuImageSideLengthTiles - tile)) % gpuImageSideLengthTiles;
			for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
				currentMappedRegionOccupancy[i][tileIdx] = false;
		}
	}
	else if (bottomRightShift.y > 0)
	{
		//Shift down
		const uint32_t tilesToFit = bottomRightShift.y;
		for (uint32_t tile = 0; tile < tilesToFit; tile++)
		{
			// Get actual tile index with wraparound
			uint32_t tileIdx = (tile + gpuImageTopLeft.y) % gpuImageSideLengthTiles;
			for (uint32_t i = 0u; i < gpuImageSideLengthTiles; i++)
				currentMappedRegionOccupancy[i][tileIdx] = false;
		}
	}

	// Shift the mapped region accordingly
	// A nice consequence of the mapped region being always maximally - sized is that
	// along any dimension, only a shift in one direction is necessary, so we can simply add up the shifts
	currentMappedRegion.topLeftTile = uint32_t2(int32_t2(currentMappedRegion.topLeftTile) + topLeftShift + bottomRightShift);
	currentMappedRegion.bottomRightTile = uint32_t2(int32_t2(currentMappedRegion.bottomRightTile) + topLeftShift + bottomRightShift);

	// Toroidal shift for the gpu image top left
	gpuImageTopLeft = (gpuImageTopLeft + uint32_t2(topLeftShift + bottomRightShift + int32_t(gpuImageSideLengthTiles))) % gpuImageSideLengthTiles;
}

core::vector<GeoreferencedImageStreamingState::ImageTileToGPUTileCorrespondence> GeoreferencedImageStreamingState::tilesToLoad(const GeoreferencedImageTileRange& viewportTileRange) const
{
	core::vector<ImageTileToGPUTileCorrespondence> retVal;
	for (uint32_t tileY = viewportTileRange.topLeftTile.y; tileY <= viewportTileRange.bottomRightTile.y; tileY++)
		for (uint32_t tileX = viewportTileRange.topLeftTile.x; tileX <= viewportTileRange.bottomRightTile.x; tileX++)
		{
			uint32_t2 imageTileIndex = uint32_t2(tileX, tileY);
			// Toroidal shift to find which gpu tile the image tile corresponds to
			uint32_t2 gpuImageTileIndex = ((imageTileIndex - currentMappedRegion.topLeftTile) + gpuImageTopLeft) % gpuImageSideLengthTiles;
			// Don't bother scheduling an upload if the tile is already resident
			if (!currentMappedRegionOccupancy[gpuImageTileIndex.x][gpuImageTileIndex.y])
				retVal.push_back({ imageTileIndex , gpuImageTileIndex });
		}
	return retVal;
}