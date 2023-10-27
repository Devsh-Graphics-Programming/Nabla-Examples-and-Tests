#include "shaderCommon.hlsl"

#include "nbl/builtin/hlsl/workgroup/scratch_sz.hlsl"

uint32_t3 gl_WorkGroupSize() {return uint32_t3(WORKGROUP_SIZE,1,1);}

static const uint32_t ArithmeticSz = nbl::hlsl::workgroup::scratch_size_arithmetic<ITEMS_PER_WG>::value;
static const uint32_t BallotSz = nbl::hlsl::workgroup::scratch_size_ballot<ITEMS_PER_WG>::value;
static const uint32_t ScratchSz = arithmeticSz+ballotSz+nbl::hlsl::workgroup::scratch_size_broadcast;

// TODO: Can we make it a static variable?
groupshared uint32_t scratch[ScratchSz];

template<uint32_t offset>
struct ScratchProxy
{
	uint32_t get(const uint32_t ix)
	{
		return scratch[ix+offset];
	}

	void set(const uint32_t ix, const uint32_t value)
	{
		scratch[ix+offset] = value;
	}

	uint32_t atomicAdd(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::glsl::atomicAdd(scratch[ix+offset], data);
	}

	uint32_t atomicOr(const uint32_t ix, const uint32_t data)
	{
		return nbl::hlsl::glsl::atomicOr(scratch[ix+offset],data);
	}

    void workgroupExecutionAndMemoryBarrier()
	{
        nbl::hlsl::glsl::barrier();
        //nbl::hlsl::glsl::memoryBarrierShared(); implied by the above
    }
};

struct SharedMemory
{
	nbl::hlsl::MemoryAdaptor<ScratchProxy<0> > arithmetic;
	nbl::hlsl::MemoryAdaptor<ScratchProxy<ArithmeticSz> > ballot;
	nbl::hlsl::MemoryAdaptor<ScratchProxy<ArithmeticSz+BallotSz> > broadcast;
};

#include "nbl/builtin/hlsl/workgroup/broadcast.hlsl"

template<class Binop>
struct operation_t// : nbl::hlsl::workgroup::OPERATION<Binop,SharedMemory>
{
	SharedMemory accessors;

	type_t operator()(NBL_CONST_REF_ARG(type_t) value)
	{
		return value;
	}
};