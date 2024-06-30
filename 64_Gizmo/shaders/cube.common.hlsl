#ifndef _THIS_EXAMPLE_CUBE_COMMON_HLSL_
#define _THIS_EXAMPLE_CUBE_COMMON_HLSL_

#ifdef __HLSL_VERSION
	struct VSInput
	{
		[[vk::location(0)]] float4 position : POSITION;
		[[vk::location(1)]] float4 color : COLOR0;
	};

	struct PSInput
	{
		float4 position : SV_Position;
		float4 color    : COLOR0;
	};
#else
	namespace cube
	{
		#include "nbl/nblpack.h"
		struct VSInput
		{
			float position[4];
			float color[4];
		} PACK_STRUCT;
		#include "nbl/nblunpack.h"
	}

	static_assert(sizeof(cube::VSInput) == sizeof(float) * 4 * 2);
#endif // __HLSL_VERSION

#include "common.hlsl"

#endif // _THIS_EXAMPLE_CUBE_COMMON_HLSL_

/*
	do not remove this text, WAVE is so bad that you can get errors if no proper ending xD
*/