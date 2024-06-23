#pragma wave shader_stage(compute)
#include "common.hlsl"

[[vk::constant_id(3)]] uint32_t SPEC_CONSTANT_VALUE = 10;

struct SomeType
{
    float4x4 a;
	uint32_t b[5];
	uint32_t c[10];
};

[[vk::binding(0,0)]] RWStructuredBuffer<SomeType> asdf;
[[vk::binding(1,0)]] RWByteAddressBuffer output[3];
[[vk::binding(2,0)]] RWByteAddressBuffer output2[];

[numthreads(WorkgroupSize, 1, 1)]
void main(uint32_t3 ID : SV_DispatchThreadID)
{
    const uint32_t index = ID.x;
	const uint32_t byteOffset = sizeof(uint32_t)*index;

    output[0].Store<uint32_t>(byteOffset, asdf[index].a[0][0] + asdf[index].b[2] + asdf[index].c[3] * SPEC_CONSTANT_VALUE);
    output[1].Store<uint32_t>(byteOffset, asdf[index].a[0][0] + asdf[index].b[2] + asdf[index].c[3]);
    output[2].Store<uint32_t>(byteOffset, asdf[index].a[0][0] + asdf[index].b[2] + asdf[index].c[3]);
    output2[1].Store<uint32_t>(byteOffset, asdf[index].a[0][0] + asdf[index].b[2] - asdf[index].c[3]);
}