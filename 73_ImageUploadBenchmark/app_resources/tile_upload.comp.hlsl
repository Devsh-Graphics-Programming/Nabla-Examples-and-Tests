#include "common.hlsl"

[[vk::binding(0,0)]] RWTexture2D<float32_t4> dstImage;
[[vk::push_constant]] PushConstantData pc;

using namespace nbl::hlsl;

static const uint32_t TILE_SIZE = 128u;

[numthreads(128, 1, 1)]
[shader("compute")]
void SnakeStore(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t gIdx = ID.x;
    const uint32_t tileIdx = gIdx / (TILE_SIZE * TILE_SIZE);
    const uint32_t localIdx = gIdx % (TILE_SIZE * TILE_SIZE);
    const uint32_t2 tileBase = uint32_t2(tileIdx % pc.tilesPerRow, tileIdx / pc.tilesPerRow) * TILE_SIZE;
    const uint32_t2 localPos = uint32_t2(localIdx % TILE_SIZE, localIdx / TILE_SIZE);
    const uint32_t2 pixelPos = tileBase + localPos;

    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);
    dstImage[pixelPos] = unpackUnorm4x8(int32_t(packed));
}

[numthreads(128, 1, 1)]
[shader("compute")]
void SnakeLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t gIdx = ID.x;
    const uint32_t tileIdx = gIdx / (TILE_SIZE * TILE_SIZE);
    const uint32_t localIdx = gIdx % (TILE_SIZE * TILE_SIZE);
    const uint32_t2 tileBase = uint32_t2(tileIdx % pc.tilesPerRow, tileIdx / pc.tilesPerRow) * TILE_SIZE;
    const uint32_t2 localPos = uint32_t2(localIdx % TILE_SIZE, localIdx / TILE_SIZE);
    const uint32_t2 pixelPos = tileBase + localPos;

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + gIdx * 4u, uint32_t(packUnorm4x8(dstImage[pixelPos])));
}

[numthreads(128, 1, 1)]
[shader("compute")]
void MortonStore(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t gIdx = ID.x;
    const uint32_t tileIdx = gIdx / (TILE_SIZE * TILE_SIZE);
    const uint32_t localIdx = gIdx % (TILE_SIZE * TILE_SIZE);
    const uint32_t2 tileBase = uint32_t2(tileIdx % pc.tilesPerRow, tileIdx / pc.tilesPerRow) * TILE_SIZE;

    morton::code<false, 7, 2> mc;
    mc.value = uint16_t(localIdx);
    const uint32_t2 localPos = _static_cast<uint32_t2>(mc);
    const uint32_t2 pixelPos = tileBase + localPos;

    const uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);
    dstImage[pixelPos] = unpackUnorm4x8(int32_t(packed));
}

[numthreads(128, 1, 1)]
[shader("compute")]
void MortonLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t gIdx = ID.x;
    const uint32_t tileIdx = gIdx / (TILE_SIZE * TILE_SIZE);
    const uint32_t localIdx = gIdx % (TILE_SIZE * TILE_SIZE);
    const uint32_t2 tileBase = uint32_t2(tileIdx % pc.tilesPerRow, tileIdx / pc.tilesPerRow) * TILE_SIZE;

    morton::code<false, 7, 2> mc;
    mc.value = uint16_t(localIdx);
    const uint32_t2 localPos = _static_cast<uint32_t2>(mc);
    const uint32_t2 pixelPos = tileBase + localPos;

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + gIdx * 4u, uint32_t(packUnorm4x8(dstImage[pixelPos])));
}
