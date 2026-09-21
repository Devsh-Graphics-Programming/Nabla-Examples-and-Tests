#include "common.hlsl"

[[vk::binding(0,0)]] RWTexture2D<float32_t4> dstImage;
[[vk::push_constant]] PushConstantData pc;

using namespace nbl::hlsl;

struct ImageWriteInfo
{
    uint32_t2 writePos;
    uint32_t bufferReadOffset;
};

ImageWriteInfo GetSnakeReadWriteAddress(uint32_t2 threadID)
{
    ImageWriteInfo ret;
    const uint32_t2 globalPos = threadID.xy;
    const uint32_t2 tileCoord = globalPos >> TILE_SIZE_LOG2;
    const uint32_t2 localPos = globalPos & TILE_SIZE_MASK;
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t localLinearIdx = (localPos.y << TILE_SIZE_LOG2) + localPos.x;
    const uint32_t srcPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + localLinearIdx;
    ret.bufferReadOffset = srcPixelIdx * PIXEL_BYTE_SIZE;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);
    ret.writePos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + localPos;
    return ret;
}

[numthreads(SNAKE_WORKGROUP_SIZE_X, SNAKE_WORKGROUP_SIZE_Y, 1)]
[shader("compute")]
void SnakeLoadStore(uint32_t3 threadID : SV_DispatchThreadID)
{
    // The CPU packs each 128x128 tile linearly in row-major order. TILE_SIZE is
    // a power of two, so >>, <<, and & replace division, multiplication, and
    // modulo by TILE_SIZE/TILE_PIXELS.
    ImageWriteInfo readWriteAddress = GetSnakeReadWriteAddress(threadID.xy);
    
    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + readWriteAddress.bufferReadOffset);
    dstImage[readWriteAddress.writePos] = unpackUnorm4x8(int32_t(packed));
}

ImageWriteInfo GetMortonReadWriteAddress(uint32_t2 groupThreadID, uint32_t2 groupID)
{
    ImageWriteInfo ret;
    const uint32_t2 globalBlock = groupID.xy;
    const uint32_t2 threadPos = groupThreadID.xy;
    const uint32_t2 tileCoord = globalBlock >> BLOCKS_PER_TILE_LOG2;
    const uint32_t2 blockCoordInTile = globalBlock & (BLOCKS_PER_TILE - 1u);
    const uint32_t tileIdx = tileCoord.y * pc.tilesPerRow + tileCoord.x;
    const uint32_t blockIdxInTile = (blockCoordInTile.y << BLOCKS_PER_TILE_LOG2) + blockCoordInTile.x;
    const uint32_t localLinearIdx = (threadPos.y << BLOCK_SIZE_LOG2) + threadPos.x;
    const uint32_t srcPixelIdx = (tileIdx << TILE_PIXELS_LOG2) + (blockIdxInTile << BLOCK_PIXELS_LOG2) + localLinearIdx;
    ret.bufferReadOffset = srcPixelIdx * PIXEL_BYTE_SIZE;
    const uint32_t packedDstTile = vk::RawBufferLoad<uint32_t>(pc.dstTileLocationsAddress + tileIdx * 4u);
    const uint32_t2 dstTile = uint32_t2(packedDstTile & 0xffffu, packedDstTile >> 16u);

    morton::code<false, 4, 2> mc;
    mc.value = uint16_t(localLinearIdx);
    const uint32_t2 mortonLocalPos = _static_cast<uint32_t2>(mc);
    ret.writePos = pc.dstOffset + (dstTile << TILE_SIZE_LOG2) + (blockCoordInTile << BLOCK_SIZE_LOG2) + mortonLocalPos;
    return ret;
}

[numthreads(MORTON_WORKGROUP_SIZE_X, MORTON_WORKGROUP_SIZE_Y, 1)]
[shader("compute")]
void MortonLoadStore(uint32_t3 groupThreadID : SV_GroupThreadID, uint32_t3 groupID : SV_GroupID)
{
    // Each workgroup handles one 16x16 block inside a 128x128 tile. Reads stay
    // contiguous in that block; writes use a 4-bit Morton decode locally. The
    // tile/block dimensions are powers of two, so shifts/masks are exact here.
    ImageWriteInfo readWriteAddress = GetMortonReadWriteAddress(groupThreadID.xy, groupID.xy);
    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + readWriteAddress.bufferReadOffset);
    dstImage[readWriteAddress.writePos] = unpackUnorm4x8(int32_t(packed));
}