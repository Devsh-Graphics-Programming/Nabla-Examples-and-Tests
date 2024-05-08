#pragma shader_stage(compute)

#include "common.hlsl"

struct Guest
{
	uint guestVal;
};

struct Base
{
	struct Nested
	{
		uint nestedVal;
	};
	uint baseVal;
	Guest guest;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Base> BaseOutput;
[[vk::binding(1, 0)]] RWStructuredBuffer<Base::Nested> NestedOutput;

[numthreads(WorkgroupSize, 1, 1)]
void main()
{
	InterlockedAdd(BaseOutput[0].baseVal, 1u);
	InterlockedAdd(BaseOutput[0].guest.guestVal, 1u);
	InterlockedAdd(NestedOutput[0].nestedVal, 1u);
}