#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/type_traits.hlsl"

static const uint16_t PASSES = 2;

struct PushConstants
{
    nbl::hlsl::float32_t radius;
	uint32_t activeAxis : 2;
    uint32_t edgeWrapMode : 6;
};