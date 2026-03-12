#include "common.hlsl"

[[vk::binding(0,0)]] RWTexture2D<float32_t4> dstImage;
[[vk::push_constant]] PushConstantData pc;

using namespace nbl::hlsl;

static const uint32_t TILE_WIDTH = 16u;
static const uint32_t TILE_HEIGHT = 8u; 

[numthreads(128, 1, 1)]
[shader("compute")]
void linearStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t  gIdx     = ID.x;
    uint32_t2 pixelPos = uint32_t2(gIdx % pc.srcWidth, gIdx / pc.srcWidth);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);

    float32_t4 rgba = float32_t4(
        float32_t((packed >>  0u) & 0xFFu) / 255.0f,
        float32_t((packed >>  8u) & 0xFFu) / 255.0f,
        float32_t((packed >> 16u) & 0xFFu) / 255.0f,
        float32_t((packed >> 24u) & 0xFFu) / 255.0f
    );

    dstImage[pc.dstOffset + pixelPos] = rgba;
}

[numthreads(128, 1, 1)]
[shader("compute")]
void linearLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 pixelPos = uint32_t2(gIdx % pc.srcWidth, gIdx / pc.srcWidth);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    float32_t4 color = dstImage[pc.dstOffset + pixelPos];

    uint32_t r = uint32_t(color.r * 255.0f + 0.5f);
    uint32_t g = uint32_t(color.g * 255.0f + 0.5f);
    uint32_t b = uint32_t(color.b * 255.0f + 0.5f);
    uint32_t a = uint32_t(color.a * 255.0f + 0.5f);
    uint32_t packed = (r << 0u) | (g << 8u) | (b << 16u) | (a << 24u);
    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + gIdx * 4u, packed);
}


uint32_t2 snakePixelPos(uint32_t gIdx, uint32_t srcWidth)
{
    static const uint32_t PIXELS_PER_TILE = TILE_WIDTH * TILE_HEIGHT;
    uint32_t tilesPerRow = srcWidth / TILE_WIDTH;

    uint32_t tileIdx = gIdx / PIXELS_PER_TILE;
    uint32_t localIdx = gIdx % PIXELS_PER_TILE;

    uint32_t tileRow = tileIdx / tilesPerRow;
    uint32_t tileCol = tileIdx % tilesPerRow;
    // Odd rows: reverse X direction 
    if (tileRow & 1u)
        tileCol = tilesPerRow - 1u - tileCol;

    uint32_t localX = localIdx % TILE_WIDTH;
    uint32_t localY = localIdx / TILE_WIDTH;

    return uint32_t2(
        tileCol * TILE_WIDTH + localX,
        tileRow * TILE_HEIGHT + localY
    );
}

[numthreads(128, 1, 1)]
[shader("compute")]
void SnakeOrderStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 pixelPos = snakePixelPos(gIdx, pc.srcWidth);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);

    float32_t4 rgba = float32_t4(
        float32_t((packed >>  0u) & 0xFFu) / 255.0f,
        float32_t((packed >>  8u) & 0xFFu) / 255.0f,
        float32_t((packed >> 16u) & 0xFFu) / 255.0f,
        float32_t((packed >> 24u) & 0xFFu) / 255.0f
    );

    dstImage[pc.dstOffset + pixelPos] = rgba;
}

[numthreads(128, 1, 1)]
[shader("compute")]
void SnakeOrderLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 pixelPos = snakePixelPos(gIdx, pc.srcWidth);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    float32_t4 color = dstImage[pc.dstOffset + pixelPos];

    uint32_t r = uint32_t(color.r * 255.0f + 0.5f);
    uint32_t g = uint32_t(color.g * 255.0f + 0.5f);
    uint32_t b = uint32_t(color.b * 255.0f + 0.5f);
    uint32_t a = uint32_t(color.a * 255.0f + 0.5f);
    uint32_t packed = (r << 0u) | (g << 8u) | (b << 16u) | (a << 24u);

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + gIdx * 4u, packed);
}
    
uint32_t mortonCompact1By1(uint32_t x)
{
    x &= 0x55555555u;
    x = (x ^ (x >> 1u))  & 0x33333333u;
    x = (x ^ (x >> 2u))  & 0x0f0f0f0fu;
    x = (x ^ (x >> 4u))  & 0x00ff00ffu;
    x = (x ^ (x >> 8u))  & 0x0000ffffu;
    return x;
}

uint32_t2 mortonDecode(uint32_t code)
{
    return uint32_t2(
        mortonCompact1By1(code),
        mortonCompact1By1(code >> 1u)
    );
}
    
void batchedTileInfo(uint32_t gIdx, uint32_t tileW, uint32_t tileH, uint32_t tilesPerRow,
    out uint32_t2 tileBase, out uint32_t localIdx)
{
    uint32_t pixelsPerTile = tileW * tileH;
    uint32_t tileIdx = gIdx / pixelsPerTile;
    localIdx = gIdx % pixelsPerTile;
    uint32_t tileCol = tileIdx % tilesPerRow;
    uint32_t tileRow = tileIdx / tilesPerRow;
    tileBase = uint32_t2(tileCol * tileW, tileRow * tileH);
}

float32_t4 unpackRGBA(uint32_t packed)
{
    return float32_t4(
        float32_t((packed >>  0u) & 0xFFu) / 255.0f,
        float32_t((packed >>  8u) & 0xFFu) / 255.0f,
        float32_t((packed >> 16u) & 0xFFu) / 255.0f,
        float32_t((packed >> 24u) & 0xFFu) / 255.0f
    );
}

[numthreads(128, 1, 1)]
[shader("compute")]
void BatchedLinearStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 tileBase;
    uint32_t localIdx;
    batchedTileInfo(gIdx, pc.srcWidth, pc.srcHeight, pc.tilesPerRow, tileBase, localIdx);

    uint32_t2 localPos = uint32_t2(localIdx % pc.srcWidth, localIdx / pc.srcWidth);
    uint32_t2 pixelPos = tileBase + localPos;

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);
    dstImage[pixelPos] = unpackRGBA(packed);
}

[numthreads(128, 1, 1)]
[shader("compute")]
void BatchedSnakeStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 tileBase;
    uint32_t localIdx;
    batchedTileInfo(gIdx, pc.srcWidth, pc.srcHeight, pc.tilesPerRow, tileBase, localIdx);

    // Snake within tile row-major with zigzag on odd tile rows
    uint32_t localTilesPerRow = pc.srcWidth / TILE_WIDTH;
    uint32_t subTileIdx = localIdx / (TILE_WIDTH * TILE_HEIGHT);
    uint32_t subLocalIdx = localIdx % (TILE_WIDTH * TILE_HEIGHT);
    uint32_t subRow = subTileIdx / localTilesPerRow;
    uint32_t subCol = subTileIdx % localTilesPerRow;
    if (subRow & 1u)
        subCol = localTilesPerRow - 1u - subCol;
    uint32_t localX = subCol * TILE_WIDTH + (subLocalIdx % TILE_WIDTH);
    uint32_t localY = subRow * TILE_HEIGHT + (subLocalIdx / TILE_WIDTH);
    uint32_t2 pixelPos = tileBase + uint32_t2(localX, localY);

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);
    dstImage[pixelPos] = unpackRGBA(packed);
}

[numthreads(128, 1, 1)]
[shader("compute")]
void BatchedMortonStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 tileBase;
    uint32_t localIdx;
    batchedTileInfo(gIdx, pc.srcWidth, pc.srcHeight, pc.tilesPerRow, tileBase, localIdx);

    uint32_t2 localPos = mortonDecode(localIdx);
    uint32_t2 pixelPos = tileBase + localPos;

    if (localPos.x >= pc.srcWidth || localPos.y >= pc.srcHeight)
        return;

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);
    dstImage[pixelPos] = unpackRGBA(packed);
}

[numthreads(128, 1, 1)]
[shader("compute")]
void MortonOrderStore(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 pixelPos = mortonDecode(gIdx);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    uint32_t packed = vk::RawBufferLoad<uint32_t>(pc.deviceBufferAddress + gIdx * 4u);

    float32_t4 rgba = float32_t4(
        float32_t((packed >>  0u) & 0xFFu) / 255.0f,
        float32_t((packed >>  8u) & 0xFFu) / 255.0f,
        float32_t((packed >> 16u) & 0xFFu) / 255.0f,
        float32_t((packed >> 24u) & 0xFFu) / 255.0f
    );

    dstImage[pc.dstOffset + pixelPos] = rgba;
}

[numthreads(128, 1, 1)]
[shader("compute")]
void MortonOrderLoad(uint32_t3 ID : SV_DispatchThreadID)
{
    uint32_t gIdx = ID.x;
    uint32_t2 pixelPos = mortonDecode(gIdx);

    if (pixelPos.x >= pc.srcWidth || pixelPos.y >= pc.srcHeight)
        return;

    float32_t4 color = dstImage[pc.dstOffset + pixelPos];

    uint32_t r = uint32_t(color.r * 255.0f + 0.5f);
    uint32_t g = uint32_t(color.g * 255.0f + 0.5f);
    uint32_t b = uint32_t(color.b * 255.0f + 0.5f);
    uint32_t a = uint32_t(color.a * 255.0f + 0.5f);
    uint32_t packed = (r << 0u) | (g << 8u) | (b << 16u) | (a << 24u);

    vk::RawBufferStore<uint32_t>(pc.deviceBufferAddress + gIdx * 4u, packed);
}