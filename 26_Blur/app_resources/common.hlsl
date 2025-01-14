#include "nbl/builtin/hlsl/cpp_compat.hlsl"

static const uint16_t PASSES = 2;
static const uint16_t CHANNELS = 3; // TODO: set dynamically
static const uint16_t WORKGROUP_SIZE = 1024; // TODO: set dynamically, by dynamically compiling a shader (see other examples) make sure its PoT though
static const uint16_t MAX_SCANLINE = 2048; // TODO: set dynamically, by dynamically compiling a shader (see other examples)

enum EdgeWrapMode : uint32_t { // TODO: move `ISampler` enum to HLSL then alias back in `ISampler`
	WRAP_MODE_CLAMP_TO_BORDER,
	WRAP_MODE_CLAMP_TO_EDGE,
	WRAP_MODE_REPEAT,
	WRAP_MODE_MIRROR,

	WRAP_MODE_MAX,
};

struct PushConstants
{
    uint32_t radius; // TODO: float32_t
	uint16_t flip; // TODO: rename activeAxis
    uint16_t edgeWrapMode;
};