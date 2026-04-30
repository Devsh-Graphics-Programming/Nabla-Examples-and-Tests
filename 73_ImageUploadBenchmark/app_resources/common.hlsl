#include <nbl/builtin/hlsl/morton.hlsl>

struct PushConstantData
{
    uint64_t deviceBufferAddress;
    uint64_t dstTileLocationsAddress;
    uint32_t2 dstOffset;
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t tilesPerRow;
};
