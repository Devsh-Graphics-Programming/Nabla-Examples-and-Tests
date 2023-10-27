#include "shaderCommon.hlsl"

#include "nbl/builtin/hlsl/workgroup/scratch_sz.hlsl"
#include "nbl/builtin/hlsl/workgroup/ballot.hlsl"
#include "nbl/builtin/hlsl/memory_accessor.hlsl"


// TODO: move to a builtin!
template<uint32_t WGSZ,uint32_t SGSZ>
struct required_scratch_size : nbl::hlsl::workgroup::impl::trunc_geom_series<WGSZ,SGSZ> {};

static const uint arithmeticSz = required_scratch_size<_NBL_WORKGROUP_SIZE_, nbl::hlsl::subgroup::MinSubgroupSize>::value;

static const uint32_t ballotSz = nbl::hlsl::workgroup::impl::uballotBitfieldCount + 1;
static const uint32_t broadcastSz = ballotSz;
static const uint32_t scratchSz = arithmeticSz + ballotSz + broadcastSz;
groupshared uint32_t scratch[scratchSz];

template<uint32_t offset>
struct ScratchProxy
{
	uint get(uint ix)
	{
		return scratch[ix + offset];
	}

	void set(uint ix, uint value)
	{
		scratch[ix + offset] = value;
	}

	uint atomicAdd(in uint ix, uint data)
	{
		return nbl::hlsl::glsl::atomicAdd(scratch[ix + offset], data);
	}

	uint atomicOr(in uint ix, uint data)
	{
		return nbl::hlsl::glsl::atomicOr(scratch[ix + offset], data);
	}

    void workgroupExecutionAndMemoryBarrier() {
        nbl::hlsl::glsl::barrier();
        nbl::hlsl::glsl::memoryBarrierShared();
    }
};

struct SharedMemory
{
	nbl::hlsl::MemoryAdaptor<ScratchProxy<0> > main;
	nbl::hlsl::MemoryAdaptor<ScratchProxy<arithmeticSz> > ballot;
	nbl::hlsl::MemoryAdaptor<ScratchProxy<arithmeticSz + ballotSz> > broadcast;

    void workgroupExecutionAndMemoryBarrier()
    {
        main.workgroupExecutionAndMemoryBarrier();
        ballot.workgroupExecutionAndMemoryBarrier();
        broadcast.workgroupExecutionAndMemoryBarrier();
    }
};
