#include "nbl/builtin/hlsl/cpp_compat.hlsl"

static const uint16_t WORKGROUP_SIZE = 256;

enum EdgeWrapMode : uint32_t {
	WRAP_MODE_CLAMP_TO_BORDER,
	WRAP_MODE_CLAMP_TO_EDGE,
	WRAP_MODE_REPEAT,
	WRAP_MODE_MIRROR,

	WRAP_MODE_MAX,
};

struct PushConstants
{
    uint32_t flip;
    uint32_t radius;
    uint32_t edgeWrapMode;
};