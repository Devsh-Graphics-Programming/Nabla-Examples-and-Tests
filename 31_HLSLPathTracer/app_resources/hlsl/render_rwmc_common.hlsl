#ifndef _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#define _PATHTRACER_EXAMPLE_RENDER_RWMC_COMMON_INCLUDED_
#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/rwmc/SplattingParameters.hlsl"
#include "render_common.hlsl"

struct RenderRWMCPushConstants
{
	RenderPushConstants renderPushConstants;
	int32_t packedSplattingParams;

	void setSplattingParams(const float base, const float start)
	{
		packedSplattingParams = hlsl::packHalf2x16(float32_t2(base, start));
	}

	rwmc::SplattingParameters getSplattingParams(const uint32_t cascadeCount)
	{
		const float32_t2 unpacked = hlsl::unpackHalf2x16(packedSplattingParams);
		return rwmc::SplattingParameters::create(unpacked[0], unpacked[1], cascadeCount);
	}
};

#endif
