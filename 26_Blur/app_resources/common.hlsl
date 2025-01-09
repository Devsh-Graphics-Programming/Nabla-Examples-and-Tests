#include "nbl/builtin/hlsl/cpp_compat.hlsl"

static const uint16_t WORKGROUP_SIZE = 256;

struct PushConstants
{
    uint32_t flip;
};