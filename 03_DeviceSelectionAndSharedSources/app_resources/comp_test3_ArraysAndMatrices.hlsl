#pragma shader_stage(compute)

#include "common.hlsl"

struct Foo
{
	uint2 uints2[5];
	float floats[3];
	int integer;
};

struct Bar
{
	uint2 uints2[3];
	Foo foo;
	float floats[2];
	float2x2 float2x2mat;
	double4x2 double4x2mat;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<Bar> output;

[numthreads(WorkgroupSize, 1, 1)]
void main()
{
	Bar bar;
	output[0] = bar;
}