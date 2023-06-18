// REVIEW: Not sure how the register types are chosen
// u -> For buffers that will be accessed randomly by threads i.e. thread 3 might access index 16
// b -> For uniform buffers
// t -> For buffers where each thread accesses its own index

// Instead of register(...) we can also use [[vk::binding(uint)]]

#pragma shader_stage(compute)

#include "../common.glsl"
//#define NBL_GL_KHR_shader_subgroup_arithmetic

StructuredBuffer<uint> inputValue : register(t0); // read-only

struct Output {
	uint subgroupSize;
	uint output[BUFFER_DWORD_COUNT];
};

RWStructuredBuffer<Output> outand : register(u1);
RWStructuredBuffer<Output> outxor : register(u2);
RWStructuredBuffer<Output> outor : register(u3);
RWStructuredBuffer<Output> outadd : register(u4);
RWStructuredBuffer<Output> outmul : register(u5);
RWStructuredBuffer<Output> outmin : register(u6);
RWStructuredBuffer<Output> outmax : register(u7);
RWStructuredBuffer<Output> outbitcount : register(u8);

#define _NBL_HLSL_WORKGROUP_SIZE_ 256U
//groupshared uint scratch[bitfieldDWORDs + 1];
groupshared uint scratch[_NBL_HLSL_WORKGROUP_SIZE_ + 32];
#define SHARED_MEM scratch