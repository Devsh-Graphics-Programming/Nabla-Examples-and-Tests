#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/cpp_compat.hlsl"

struct PushConstantData
{
	uint64_t inputAddress;
    uint64_t outputAddress;
	uint32_t dataElementCount;
    uint32_t minimum;
};

// Yes we do have our own re-creation of C++'s STL in HLSL2021 !
#include "nbl/builtin/hlsl/limits.hlsl"