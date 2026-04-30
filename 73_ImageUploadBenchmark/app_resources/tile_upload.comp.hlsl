#include "common.hlsl"

[[vk::binding(0,0)]] RWTexture2D<float32_t4> dstImage;
[[vk::push_constant]] PushConstantData pc;

using namespace nbl::hlsl;

static const uint32_t TILE_SIZE = 128u;
static const uint32_t TILE_SIZE_LOG2 = 7u;
static const uint32_t TILE_SIZE_MASK = TILE_SIZE - 1u;
static const uint32_t TILE_PIXELS_LOG2 = TILE_SIZE_LOG2 * 2u;
static const uint32_t BLOCK_SIZE = 16u;
static const uint32_t BLOCK_SIZE_LOG2 = 4u;
static const uint32_t BLOCK_PIXELS_LOG2 = BLOCK_SIZE_LOG2 * 2u;
static const uint32_t BLOCKS_PER_TILE_LOG2 = TILE_SIZE_LOG2 - BLOCK_SIZE_LOG2;
static const uint32_t BLOCKS_PER_TILE = TILE_SIZE / BLOCK_SIZE;

[numthreads(128, 4, 1)]
[shader("compute")]
void SnakeStore(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t2 globalPos = ID.xy;
    const uint32_t2 tileCoord = globalPos >> TILE_SIZE_LOG2;
    const uint32_t2 localPos = globalPos & TILE_SIZE_MASK;
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t localLinearIdx = (localPos.y << TILE_SIZE_LOG2) + localPos.x;
    const uint32_t srcPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + localLinearIdx;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);
    const uint32_t2 pixelPos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + localPos;

    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + srcPixelIdx * 4u);

    dstImage[pixelPos] = unpackUnorm4x8(int32_t(packed));
}

[numthreads(128, 4, 1)]
[shader("compute")]
void SnakeLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t2 globalPos = ID.xy;
    const uint32_t2 tileCoord = globalPos >> TILE_SIZE_LOG2;
    const uint32_t2 localPos = globalPos & TILE_SIZE_MASK;
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t localLinearIdx = (localPos.y << TILE_SIZE_LOG2) + localPos.x;
    const uint32_t dstPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + localLinearIdx;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);
    const uint32_t2 pixelPos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + localPos;

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + dstPixelIdx * 4u, uint32_t(packUnorm4x8(dstImage[pixelPos])));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void MortonStore(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    const uint32_t2 globalBlock = GroupID.xy;
    const uint32_t2 threadPos = ID.xy;
    const uint32_t2 tileCoord = globalBlock >> BLOCKS_PER_TILE_LOG2;
    const uint32_t2 blockCoordInTile = globalBlock & (BLOCKS_PER_TILE - 1u);
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t blockIdxInTile = (blockCoordInTile.y << BLOCKS_PER_TILE_LOG2) + blockCoordInTile.x;
    const uint32_t localLinearIdx = (threadPos.y << BLOCK_SIZE_LOG2) + threadPos.x;
    const uint32_t srcPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + (blockIdxInTile << BLOCK_PIXELS_LOG2) + localLinearIdx;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);

    morton::code<false, 4, 2> mc;
    mc.value = uint16_t(localLinearIdx);
    const uint32_t2 mortonLocalPos = _static_cast<uint32_t2>(mc);
    const uint32_t2 pixelPos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + (blockCoordInTile << BLOCK_SIZE_LOG2) + mortonLocalPos;

    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + srcPixelIdx * 4u);
    dstImage[pixelPos] = unpackUnorm4x8(int32_t(packed));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void MortonLoad(uint32_t3 ID : SV_GroupThreadID, uint32_t3 GroupID : SV_GroupID)
{
    const uint32_t2 globalBlock = GroupID.xy;
    const uint32_t2 threadPos = ID.xy;
    const uint32_t2 tileCoord = globalBlock >> BLOCKS_PER_TILE_LOG2;
    const uint32_t2 blockCoordInTile = globalBlock & (BLOCKS_PER_TILE - 1u);
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t blockIdxInTile = (blockCoordInTile.y << BLOCKS_PER_TILE_LOG2) + blockCoordInTile.x;
    const uint32_t localLinearIdx = (threadPos.y << BLOCK_SIZE_LOG2) + threadPos.x;
    const uint32_t dstPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + (blockIdxInTile << BLOCK_PIXELS_LOG2) + localLinearIdx;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);

    morton::code<false, 4, 2> mc;
    mc.value = uint16_t(localLinearIdx);
    const uint32_t2 mortonLocalPos = _static_cast<uint32_t2>(mc);
    const uint32_t2 pixelPos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + (blockCoordInTile << BLOCK_SIZE_LOG2) + mortonLocalPos;

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + dstPixelIdx * 4u, uint32_t(packUnorm4x8(dstImage[pixelPos])));
}
