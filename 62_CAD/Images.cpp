#include "Images.h"

using namespace nbl::hlsl;

ImageCleanup::ImageCleanup()
	: imagesMemorySuballocator(nullptr)
	, addr(ImagesMemorySubAllocator::InvalidAddress)
	, size(0ull)
{
}

ImageCleanup::~ImageCleanup()
{
	// printf(std::format("Actual Eviction size={}, offset={} \n", size, addr).c_str());
	if (imagesMemorySuballocator && addr != ImagesMemorySubAllocator::InvalidAddress)
		imagesMemorySuballocator->deallocate(addr, size);
}

bool GeoreferencedImageStreamingState::init(const OrientedBoundingBox2D& worldspaceOBB, const uint32_t2 fullResImageExtents, const asset::E_FORMAT sourceImageFormat, const std::filesystem::path& storagePath)
{
	this->worldspaceOBB = std::move(worldspaceOBB);
	this->fullResImageExtents = fullResImageExtents;
	this->sourceImageFormat = sourceImageFormat;
	this->storagePath = storagePath;
	//	1. Get the displacement (will be an offset vector in world coords and world units) from the `topLeft` corner of the image to the point
	//	2. Transform this displacement vector into the coordinates in the basis {dirU, dirV} (worldspace vectors that span the sides of the image).
	//	The composition of these matrices therefore transforms any point in worldspace into uv coordinates in imagespace
	//  To reduce code complexity, instead of computing the product of these matrices, since the first is a pure displacement matrix 
	//  (non-homogenous 2x2 upper left is identity matrix) and the other is a pure rotation matrix (2x2) we can just put them together
	//  by putting the rotation in the upper left 2x2 of the result and the post-rotated displacement in the upper right 2x1.
	//  The result is also 2x3 and not 3x3 because we can drop he homogenous since the displacement yields a vector

	// 2. Change of Basis. Since {dirU, dirV} are orthogonal, the matrix to change from world coords to `span{dirU, dirV}` coords has a quite nice expression
	//    Non-uniform scaling doesn't affect this, but this has to change if we allow for shearing (basis vectors stop being orthogonal)
	const float64_t2 dirU = this->worldspaceOBB.dirU;
	const float64_t2 dirV = float64_t2(dirU.y, -dirU.x) * float64_t(this->worldspaceOBB.aspectRatio);
	const float64_t dirULengthSquared = nbl::hlsl::dot(dirU, dirU);
	const float64_t dirVLengthSquared = nbl::hlsl::dot(dirV, dirV);
	const float64_t2 firstRow = dirU / dirULengthSquared;
	const float64_t2 secondRow = dirV / dirVLengthSquared;

	const float64_t2 displacement = -(this->worldspaceOBB.topLeft);
	// This is the same as multiplying the change of basis matrix by the displacement vector
	const float64_t postRotatedShiftX = nbl::hlsl::dot(firstRow, displacement);
	const float64_t postRotatedShiftY = nbl::hlsl::dot(secondRow, displacement);

	// Put them all together
	this->worldToUV = float64_t2x3(firstRow.x, firstRow.y, postRotatedShiftX, secondRow.x, secondRow.y, postRotatedShiftY);

	// Also set the maxMipLevel - to keep stuff simple, we don't consider having less than one tile per dimension
	// If you're zoomed out enough then at that point the whole image is just sampled as one tile along that dimension
	// In pathological cases, such as images that are way bigger on one side than the other, this could cause aliasing and slow down sampling if zoomed out too much. 
	// If we were ever to observe such pathological cases, then maybe we should consider doing something else here. For example, making the loader able to handle different tile lengths per dimension
	// (so for example a 128x64 tile) but again for now it should be left as-is.
	uint32_t2 maxMipLevels = nbl::hlsl::findMSB(nbl::hlsl::roundUpToPoT(this->fullResImageExtents / GeoreferencedImageTileSize));
	this->maxMipLevel = nbl::hlsl::min(maxMipLevels.x, maxMipLevels.y);

	this->fullImageTileLength = (this->fullResImageExtents - 1u) / GeoreferencedImageTileSize + 1u;

	return true;
}

void GeoreferencedImageStreamingState::updateStreamingStateForViewport(const uint32_t2 viewportExtent, const float64_t3x3& ndcToWorldMat)
{
	currentViewportTileRange = computeViewportTileRange(viewportExtent, ndcToWorldMat);
	// Slide or remap the current mapped region to ensure the viewport falls inside it
	ensureMappedRegionCoversViewport(currentViewportTileRange);
	
	const uint32_t2 lastTileIndex = getLastTileIndex(currentViewportTileRange.baseMipLevel);
	const uint32_t2 lastTileSampligOffsetMip0 = (lastTileIndex * GeoreferencedImageTileSize) << currentViewportTileRange.baseMipLevel;
	lastTileSamplingExtent = fullResImageExtents - lastTileSampligOffsetMip0;
	const uint32_t2 lastTileTargetExtentMip1 = lastTileSamplingExtent >> (currentViewportTileRange.baseMipLevel + 1);
	lastTileTargetExtent = lastTileTargetExtentMip1 << 1u;
}

core::vector<GeoreferencedImageStreamingState::ImageTileToGPUTileCorrespondence> GeoreferencedImageStreamingState::tilesToLoad() const
{
	core::vector<ImageTileToGPUTileCorrespondence> retVal;
	for (uint32_t tileY = currentViewportTileRange.topLeftTile.y; tileY <= currentViewportTileRange.bottomRightTile.y; tileY++)
		for (uint32_t tileX = currentViewportTileRange.topLeftTile.x; tileX <= currentViewportTileRange.bottomRightTile.x; tileX++)
		{
			uint32_t2 imageTileIndex = uint32_t2(tileX, tileY);
			// Toroidal shift to find which gpu tile the image tile corresponds to
			uint32_t2 gpuImageTileIndex = ((imageTileIndex - currentMappedRegionTileRange.topLeftTile) + gpuImageTopLeft) % gpuImageSideLengthTiles;
			// Don't bother scheduling an upload if the tile is already resident
			if (!currentMappedRegionOccupancy[gpuImageTileIndex.x][gpuImageTileIndex.y])
				retVal.push_back({ imageTileIndex , gpuImageTileIndex });
		}
	return retVal;
}

GeoreferencedImageInfo GeoreferencedImageStreamingState::computeGeoreferencedImageAddressingAndPositioningInfo()
{
	GeoreferencedImageInfo ret = {};

	// Figure out an obb that covers only the currently loaded tiles
	OrientedBoundingBox2D viewportEncompassingOBB = worldspaceOBB;
	// The image's worldspace dirU corresponds to `fullResImageExtents.x` texels of the image, therefore one image texel in the U direction has a worldspace span of `dirU / fullResImageExtents.x`.
	// One mip 0 tiles therefore spans `dirU * GeoreferencedImageTileSize/ fullResImageExtents.x`. A mip `n` tile spans `2^n` this amount, since each texel at that mip level spans
	// `2^n` mip texels. Therefore the dirU offset from the image wordlspace's topLeft of the tile of index `currentViewportTileRange.topLeftTile.x` at mip level `currentMappedRegion.baseMipLevel` can be calculated as
	const uint32_t oneTileTexelSpan = GeoreferencedImageTileSize << currentMappedRegionTileRange.baseMipLevel;
	viewportEncompassingOBB.topLeft += worldspaceOBB.dirU * float32_t(currentViewportTileRange.topLeftTile.x * oneTileTexelSpan) / float32_t(fullResImageExtents.x);
	// Same reasoning for offset in v direction
	const float32_t2 dirV = float32_t2(worldspaceOBB.dirU.y, -worldspaceOBB.dirU.x) * worldspaceOBB.aspectRatio;
	viewportEncompassingOBB.topLeft += dirV * float32_t(currentViewportTileRange.topLeftTile.y * oneTileTexelSpan) / float32_t(fullResImageExtents.y);

	const uint32_t2 viewportTileLength = currentViewportTileRange.bottomRightTile - currentViewportTileRange.topLeftTile + uint32_t2(1, 1);
	// If the last tile is visible, we use the fractional span for the last tile. Otherwise it's just a normal tile
	const bool2 lastTileVisible = isLastTileVisible(currentViewportTileRange.bottomRightTile);
	const uint32_t2 lastSampledImageTileTexels = { lastTileVisible.x ? lastTileSamplingExtent.x : oneTileTexelSpan, lastTileVisible.y ? lastTileSamplingExtent.y : oneTileTexelSpan };
	const uint32_t2 lastGPUImageTileTexels = { lastTileVisible.x ? lastTileTargetExtent.x : GeoreferencedImageTileSize, lastTileVisible.y ? lastTileTargetExtent.y : GeoreferencedImageTileSize };

	// Instead of grouping per tile like in the offset case, we group per texel: the same reasoning leads to a single texel at current mip level having a span of `dirU * 2^(currentMappedRegionTileRange.baseMipLevel)/ fullResImageExtents.x`
	// in the U direction. Therefore the span in worldspace of the OBB we construct is just this number multiplied by the number of image texels spanned to draw.
	// The number of texels is just `GeoreferencedImageTileSize * 2^{mipLevel}` times the number of full tiles (all but the last) + the number of texels of the last tile, which might not be a full tile if near the right boundary
	const uint32_t2 sampledImageTexels = oneTileTexelSpan * (viewportTileLength - 1u) + lastSampledImageTileTexels;
	viewportEncompassingOBB.dirU = worldspaceOBB.dirU * float32_t(sampledImageTexels.x) / float32_t(fullResImageExtents.x);
	// Simply number of image texels in the y direction divided by number of texels in the x direction.
	viewportEncompassingOBB.aspectRatio = float32_t(sampledImageTexels.y) / float32_t(sampledImageTexels.x);

	// GPU tile corresponding to the real image tile containing the viewport top left - we can let it be negative since wrapping mode is repeat, negative tiles are correct modulo `gpuImageSideLengthTiles`
	const uint32_t2 viewportTopLeftGPUTile = currentViewportTileRange.topLeftTile - currentMappedRegionTileRange.topLeftTile + gpuImageTopLeft;
	// To get the uv corresponding to the above, simply divide the tile index by the number of tiles in the GPU image.
	// However to consider a one-texel shift inward (to prevent color bleeding at the edges) we map both numerator and denominator to texel units (by multiplying with `GeoreferencedImageTileSize`) and add 
	// a single texel to the numerator
	const float32_t2 minUV = float32_t2(GeoreferencedImageTileSize * viewportTopLeftGPUTile + 1u) / float32_t(GeoreferencedImageTileSize * gpuImageSideLengthTiles);
	// If the image was perfectly partitioned into tiles, we could get the maxUV in a similar fashion to minUV: Just compute `bottomRightTile - currentMappedRegionTileRange.topLeftTile` to get a tile
	// then divide by `gpuImageSideLengthTiles` to get a coord in `(0,1)` (correct modulo `gpuImageSideLengthTiles`)
	// However the last tile might not have all `GeoreferencedImageTileSize` texels in it. Therefore maxUV computation can be separated into a UV contribution by all full tiles (all but the last) + a contribution from the last tile
	// UV contribution from full tiles will therefore be `(bottomRightTile - currentMappedRegionTileRange.topLeftTile) / gpuImageSideLengthTiles` while last tile contribution will be 
	// `lastGPUImageTileTexels / (gpuImageSideLengthTiles * GeoreferencedImageTileSize)`. We group terms below to reduce number of float ops.
	// Again we first map to texel units then subtract one to add a single texel uv shift.
	const uint32_t2 viewportBottomRightGPUTile = currentViewportTileRange.bottomRightTile - currentMappedRegionTileRange.topLeftTile + gpuImageTopLeft;
	const float32_t2 maxUV = float32_t2(GeoreferencedImageTileSize * viewportBottomRightGPUTile + lastGPUImageTileTexels - 1u) / float32_t(GeoreferencedImageTileSize * gpuImageSideLengthTiles);

	ret.minUV = minUV;
	ret.maxUV = maxUV;
	ret.topLeft = viewportEncompassingOBB.topLeft;
	ret.dirU = viewportEncompassingOBB.dirU;
	ret.aspectRatio = viewportEncompassingOBB.aspectRatio;

	return ret;
}

GeoreferencedImageTileRange GeoreferencedImageStreamingState::computeViewportTileRange(const uint32_t2 viewportExtent, const float64_t3x3& ndcToWorldMat)
{
	// These are vulkan standard, might be different in n4ce!
	constexpr static float64_t3 topLeftViewportNDC = float64_t3(-1.0, -1.0, 1.0);
	constexpr static float64_t3 topRightViewportNDC = float64_t3(1.0, -1.0, 1.0);
	constexpr static float64_t3 bottomLeftViewportNDC = float64_t3(-1.0, 1.0, 1.0);
	constexpr static float64_t3 bottomRightViewportNDC = float64_t3(1.0, 1.0, 1.0);

	// First get world coordinates for each of the viewport's corners
	const float64_t3 topLeftViewportWorld = nbl::hlsl::mul(ndcToWorldMat, topLeftViewportNDC);
	const float64_t3 topRightViewportWorld = nbl::hlsl::mul(ndcToWorldMat, topRightViewportNDC);
	const float64_t3 bottomLeftViewportWorld = nbl::hlsl::mul(ndcToWorldMat, bottomLeftViewportNDC);
	const float64_t3 bottomRightViewportWorld = nbl::hlsl::mul(ndcToWorldMat, bottomRightViewportNDC);

	// Then we get mip 0 tiles coordinates for each of them, into the image
	const float64_t2 topLeftTileLattice = transformWorldCoordsToTileCoords(topLeftViewportWorld);
	const float64_t2 topRightTileLattice = transformWorldCoordsToTileCoords(topRightViewportWorld);
	const float64_t2 bottomLeftTileLattice = transformWorldCoordsToTileCoords(bottomLeftViewportWorld);
	const float64_t2 bottomRightTileLattice = transformWorldCoordsToTileCoords(bottomRightViewportWorld);

	// Get the min and max of each lattice coordinate to get a bounding rectangle
	const float64_t2 minTop = nbl::hlsl::min(topLeftTileLattice, topRightTileLattice);
	const float64_t2 minBottom = nbl::hlsl::min(bottomLeftTileLattice, bottomRightTileLattice);
	const float64_t2 minAll = nbl::hlsl::min(minTop, minBottom);

	const float64_t2 maxTop = nbl::hlsl::max(topLeftTileLattice, topRightTileLattice);
	const float64_t2 maxBottom = nbl::hlsl::max(bottomLeftTileLattice, bottomRightTileLattice);
	const float64_t2 maxAll = nbl::hlsl::max(maxTop, maxBottom);

	// Floor them to get an integer coordinate (index) for the tiles they fall in
	int32_t2 minAllFloored = nbl::hlsl::floor(minAll);
	int32_t2 maxAllFloored = nbl::hlsl::floor(maxAll);

	// We're undoing a previous division. Could be avoided but won't restructure the code atp.
	// Here we compute how many image pixels each side of the viewport spans 
	const float64_t2 viewportSideUImageTexelsVector = float64_t(GeoreferencedImageTileSize) * (topRightTileLattice - topLeftTileLattice);
	const float64_t2 viewportSideVImageTexelsVector = float64_t(GeoreferencedImageTileSize) * (bottomLeftTileLattice - topLeftTileLattice);

	// WARNING: This assumes pixels in the image are the same size along each axis. If the image is nonuniformly scaled or sheared, I *think* it should not matter
	// (since the pixel span takes that transformation into account), BUT we have to check if we plan on allowing those
	// Compute the side vectors of the viewport in image pixel(texel) space.
	// These vectors represent how many image pixels each side of the viewport spans.
	// They correspond to the local axes of the mapped OBB (not the mapped region one, the viewport one) in texel coordinates.
	const float64_t viewportSideUImageTexels = nbl::hlsl::length(viewportSideUImageTexelsVector);
	const float64_t viewportSideVImageTexels = nbl::hlsl::length(viewportSideVImageTexelsVector);

	// Mip is decided based on max of these
	float64_t pixelRatio = nbl::hlsl::max(viewportSideUImageTexels / viewportExtent.x, viewportSideVImageTexels / viewportExtent.y);
	pixelRatio = pixelRatio < 1.0 ? 1.0 : pixelRatio;

	GeoreferencedImageTileRange retVal = {};
	// Clamp mip level so we don't consider tiles that are too small along one dimension
	// If on a pathological case this gets too expensive because the GPU starts sampling a lot, we can consider changing this, but I doubt that will happen
	retVal.baseMipLevel = nbl::hlsl::min(nbl::hlsl::findMSB(uint32_t(nbl::hlsl::floor(pixelRatio))), int32_t(maxMipLevel));

	// Current tiles are measured in mip 0. We want the result to measure mip `retVal.baseMipLevel` tiles. Each next mip level divides by 2.
	minAllFloored >>= retVal.baseMipLevel;
	maxAllFloored >>= retVal.baseMipLevel;


	// Clamp them to reasonable tile indices
	int32_t2 lastTileIndex = getLastTileIndex(retVal.baseMipLevel);
	retVal.topLeftTile = nbl::hlsl::clamp(minAllFloored, int32_t2(0, 0), lastTileIndex);
	retVal.bottomRightTile = nbl::hlsl::clamp(maxAllFloored, int32_t2(0, 0), lastTileIndex);

	return retVal;
}

void GeoreferencedImageStreamingState::ensureMappedRegionCoversViewport(const GeoreferencedImageTileRange& viewportTileRange)
{
	// A base mip level of x in the current mapped region means we can handle the viewport having mip level y, with x <= y < x + 1.0
	// without needing to remap the region. When the user starts zooming in or out and the mip level of the viewport falls outside this range, we have to remap
	// the mapped region.
	const bool mipBoundaryCrossed = viewportTileRange.baseMipLevel != currentMappedRegionTileRange.baseMipLevel;

	// If we moved a huge amount in any direction, no tiles will remain resident, so we simply reset state
	// This only need be evaluated if the mip boundary was not already crossed
	const bool relativeShiftTooBig = !mipBoundaryCrossed &&
		nbl::hlsl::any
		(
			nbl::hlsl::abs(int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegionTileRange.topLeftTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
		)
		|| nbl::hlsl::any
		(
			nbl::hlsl::abs(int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegionTileRange.bottomRightTile)) >= int32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles)
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
	if (viewportTileRange.baseMipLevel > currentMappedRegionTileRange.baseMipLevel)
	{
		// TODO: Here we would move some mip 1 tiles to mip 0 image to save the work of reuploading them, reflect that in the tracked tiles
	}
	// Zoomed in
	else if (viewportTileRange.baseMipLevel < currentMappedRegionTileRange.baseMipLevel)
	{
		// TODO: Here we would move some mip 0 tiles to mip 1 image to save the work of reuploading them, reflect that in the tracked tiles
	}
	currentMappedRegionTileRange = viewportTileRange;
	// We can expand the currentMappedRegionTileRange to make it as big as possible, at no extra cost since we only upload tiles on demand
	// Since we use toroidal updating it's kinda the same which way we expand the region. We first try to make the extent be `gpuImageSideLengthTiles`
	currentMappedRegionTileRange.bottomRightTile = currentMappedRegionTileRange.topLeftTile + uint32_t2(gpuImageSideLengthTiles, gpuImageSideLengthTiles) - uint32_t2(1, 1);
	// This extension can cause the mapped region to fall out of bounds on border cases, therefore we clamp it and extend it in the other direction
	// by the amount of tiles we removed during clamping
	const uint32_t2 excessTiles = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegionTileRange.bottomRightTile) - int32_t2(getLastTileIndex(currentMappedRegionTileRange.baseMipLevel))));
	currentMappedRegionTileRange.bottomRightTile -= excessTiles;
	// Shifting of the topLeftTile could fall out of bounds in pathological cases or at very high mip levels (zooming out too much), so we shift if possible, otherwise set it to 0
	currentMappedRegionTileRange.topLeftTile = uint32_t2(nbl::hlsl::max(int32_t2(0, 0), int32_t2(currentMappedRegionTileRange.topLeftTile) - int32_t2(excessTiles)));

	ResetTileOccupancyState();
	// Reset state for gpu image so that it starts loading tiles at top left. Not really necessary.
	gpuImageTopLeft = uint32_t2(0, 0);
}

void GeoreferencedImageStreamingState::ResetTileOccupancyState()
{
	// Mark all gpu tiles as dirty
	currentMappedRegionOccupancy.assign(gpuImageSideLengthTiles, std::vector<bool>(gpuImageSideLengthTiles, false));
}

void GeoreferencedImageStreamingState::slideCurrentRegion(const GeoreferencedImageTileRange& viewportTileRange)
{
	// `topLeftShift` represents how many tiles up and to the left we have to move the mapped region to fit the viewport. 
	// First we compute a vector from the current mapped region's topleft to the viewport's topleft. If this vector is positive along a dimension it means
	// the viewport's topleft is to the right or below the current mapped region's topleft, so we don't have to shift the mapped region to the left/up in that case
	const int32_t2 topLeftShift = nbl::hlsl::min(int32_t2(0, 0), int32_t2(viewportTileRange.topLeftTile) - int32_t2(currentMappedRegionTileRange.topLeftTile));
	// `bottomRightShift` represents the same as above but in the other direction.
	const int32_t2 bottomRightShift = nbl::hlsl::max(int32_t2(0, 0), int32_t2(viewportTileRange.bottomRightTile) - int32_t2(currentMappedRegionTileRange.bottomRightTile));

	// The following is not necessarily equal to `gpuImageSideLengthTiles` since there can be pathological cases, as explained in the remapping method
	const uint32_t2 mappedRegionDimensions = currentMappedRegionTileRange.bottomRightTile - currentMappedRegionTileRange.topLeftTile + 1u;
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
	currentMappedRegionTileRange.topLeftTile = uint32_t2(int32_t2(currentMappedRegionTileRange.topLeftTile) + topLeftShift + bottomRightShift);
	currentMappedRegionTileRange.bottomRightTile = uint32_t2(int32_t2(currentMappedRegionTileRange.bottomRightTile) + topLeftShift + bottomRightShift);

	// Toroidal shift for the gpu image top left
	gpuImageTopLeft = (gpuImageTopLeft + uint32_t2(topLeftShift + bottomRightShift + int32_t(gpuImageSideLengthTiles))) % gpuImageSideLengthTiles;
}

std::string CachedImageRecord::toString(uint64_t imageID) const
{
	auto stringifyImageState = [](ImageState state) -> std::string {
		switch (state)
		{
		case ImageState::INVALID: return "INVALID";
		case ImageState::CREATED_AND_MEMORY_BOUND: return "CREATED_AND_MEMORY_BOUND";
		case ImageState::BOUND_TO_DESCRIPTOR_SET: return "BOUND_TO_DESCRIPTOR_SET";
		case ImageState::GPU_RESIDENT_WITH_VALID_STATIC_DATA: return "GPU_RESIDENT_WITH_VALID_STATIC_DATA";
		default: return "UNKNOWN_STATE";
		}
		};

	auto stringifyImageType = [](ImageType type) -> std::string {
		switch (type)
		{
		case ImageType::INVALID: return "INVALID";
		case ImageType::STATIC: return "STATIC";
		case ImageType::GEOREFERENCED_STREAMED: return "GEOREFERENCED_STREAMED";
		default: return "UNKNOWN_TYPE";
		}
		};

	std::string result;
	if (imageID != std::numeric_limits<uint64_t>::max())
		result += std::format("  ImageID: {}\n", imageID);

	result += std::format(
		"  Type: {}\n"
		"  State: {}\n"
		"  Array Index: {}\n"
		"  Allocation Offset: {}\n"
		"  Allocation Size: {}\n"
		"  Current Layout: {}\n"
		"  Last Used Frame Index: {}\n"
		"  GPU ImageView: {}\n"
		"  CPU Image: {}\n"
		"  Georeferenced Image State: {}\n",
		stringifyImageType(type),
		stringifyImageState(state),
		arrayIndex,
		allocationOffset,
		allocationSize,
		static_cast<uint32_t>(currentLayout),
		lastUsedFrameIndex,
		gpuImageView ? "VALID" : "NULL",
		staticCPUImage ? "VALID" : "NULL",
		georeferencedImageState ? "VALID" : "NULL"
	);
	return result;
}