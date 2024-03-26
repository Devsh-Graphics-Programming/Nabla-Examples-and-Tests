#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PushConstantData
{
	uint64_t inputAddress;
    uint64_t outputAddress;
	uint32_t dataElementCount;
};

NBL_CONSTEXPR uint32_t WorkgroupSize = 256;

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"