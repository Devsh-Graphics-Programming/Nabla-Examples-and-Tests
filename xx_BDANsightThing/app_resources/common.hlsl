#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct SHashAppendData
{
    uint32_t isValid;
    uint32_t reservoirIdx;
    uint32_t cellIdx;
    uint32_t inCellIdx;
};

struct PushConstantData
{
	uint64_t pHashAppend;
	uint64_t pCellStorage;
	uint64_t pIndex;
	nbl::hlsl::uint16_t2 renderSize;
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 16u;

#include "nbl/builtin/hlsl/random/xoroshiro.hlsl"

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"