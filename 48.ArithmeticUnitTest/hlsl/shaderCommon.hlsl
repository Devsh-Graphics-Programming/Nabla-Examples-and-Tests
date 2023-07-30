// REVIEW: Not sure how the register types are chosen
// u -> For buffers that will be accessed randomly by threads i.e. thread 3 might access index 16
// b -> For uniform buffers
// t -> For buffers where each thread accesses its own index

// Instead of register(...) we can also use [[vk::binding(uint)]]

#pragma shader_stage(compute)

#ifndef _NBL_HLSL_WORKGROUP_SIZE_
#define _NBL_HLSL_WORKGROUP_SIZE_ 256
#endif

#include "../common.glsl"

#include "nbl/builtin/hlsl/glsl_compat.hlsl"
#include "nbl/builtin/hlsl/workgroup/shared_ballot.hlsl"

// Must define all groupshared memory before including shared_memory_accessor since all the proxy structs are defined there
#define scratchSize (_NBL_HLSL_WORKGROUP_SIZE_ << 1) + (nbl::hlsl::workgroup::MaxWorkgroupSize >> 1) + 1
groupshared uint scratch[scratchSize];
#define SHARED_MEM scratch
groupshared uint broadcastScratch[uballotBitfieldCount + 1];
#define BROADCAST_MEM broadcastScratch

#include "nbl/builtin/hlsl/shared_memory_accessor.hlsl"

struct Output {
	uint subgroupSize;
	uint output[BUFFER_DWORD_COUNT];
};

StructuredBuffer<uint> inputValue : register(t0); // read-only

RWStructuredBuffer<Output> outand : register(u1);
RWStructuredBuffer<Output> outxor : register(u2);
RWStructuredBuffer<Output> outor : register(u3);
RWStructuredBuffer<Output> outadd : register(u4);
RWStructuredBuffer<Output> outmul : register(u5);
RWStructuredBuffer<Output> outmin : register(u6);
RWStructuredBuffer<Output> outmax : register(u7);
RWStructuredBuffer<Output> outbitcount : register(u8);

struct MainScratchProxy
{
	uint get(uint ix)
	{
		return SHARED_MEM[ix];
	}

	void set(uint ix, uint value)
	{
		SHARED_MEM[ix] = value;
	}
	
	uint atomicAdd(in uint ix, uint data)
	{
		return nbl::hlsl::glsl::atomicAdd(SHARED_MEM[ix], data);
	}
	
	uint atomicOr(in uint ix, uint data)
	{
		return nbl::hlsl::glsl::atomicOr(SHARED_MEM[ix], data);
	}
};

struct SharedMemory
{
	nbl::hlsl::SharedMemoryAdaptor<MainScratchProxy> main;
	nbl::hlsl::SharedMemoryAdaptor<nbl::hlsl::BroadcastScratchProxy> broadcast;
};