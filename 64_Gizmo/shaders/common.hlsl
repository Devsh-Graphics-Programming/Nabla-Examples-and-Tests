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

	struct SBasicViewParameters
	{
		float4x4 MVP;
		float3x4 MV;
		float3x3 normalMat;
		float3 padding; 
	};
#else
	#include "nbl/nblpack.h"
	struct VSInput
	{
		float position[4];
		float color[4];
	} PACK_STRUCT;
	#include "nbl/nblunpack.h"

	static_assert(sizeof(VSInput) == sizeof(float) * 4 * 2);
#endif // __HLSL_VERSION

struct PushConstants
{
	bool withGizmo;
};